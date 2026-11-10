from __future__ import annotations

import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from mitosis.cache import Cache
from mitosis.config import ModelSpec, Settings
from mitosis.ledger import Ledger
from mitosis.providers import load_all
from mitosis.providers.base import FACTORIES, GenResult, PermanentError, ProviderError, TransientError

log = logging.getLogger(__name__)

RETRYABLE = (TransientError, httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)


class Breaker:
    def __init__(self, threshold: int = 3, cooldown: float = 90.0) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self.failures = 0
        self.opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def closed(self) -> bool:
        with self._lock:
            if self.opened_at is None:
                return True
            if time.monotonic() - self.opened_at > self.cooldown:
                self.opened_at = None
                self.failures = self.threshold - 1
                return True
            return False

    def success(self) -> None:
        with self._lock:
            self.failures = 0
            self.opened_at = None

    def failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.threshold:
                self.opened_at = time.monotonic()


class Member:
    def __init__(self, spec: ModelSpec, settings: Settings) -> None:
        self.spec = spec
        self.settings = settings
        self.semaphore = threading.BoundedSemaphore(max(1, spec.concurrency))
        self.breaker = Breaker()
        self._instance: Any = None
        self._lock = threading.Lock()

    @property
    def configured(self) -> bool:
        if (self.spec.provider, self.spec.kind) not in FACTORIES:
            return False
        return all(os.environ.get(name) for name in self.spec.requires)

    @property
    def instance(self) -> Any:
        with self._lock:
            if self._instance is None:
                factory = FACTORIES[(self.spec.provider, self.spec.kind)]
                self._instance = factory(self.spec, self.settings)
            return self._instance


class Chain:
    def __init__(self, kind: str, members: list[Member], registry: Registry) -> None:
        self.kind = kind
        self.members = members
        self.registry = registry

    @property
    def available(self) -> bool:
        return bool(self.members)

    @property
    def offline(self) -> bool:
        return bool(self.members) and all(m.spec.offline for m in self.members)

    @property
    def names(self) -> list[str]:
        return [m.spec.id for m in self.members]

    def json(self, system: str, prompt: str, schema: dict, images: list[Path] | None = None, max_tokens: int = 8000) -> GenResult:
        return self.call("json", system=system, prompt=prompt, schema=schema, images=images or [], max_tokens=max_tokens)

    def generate(self, prompt: str, references: list[Path], aspect, seed: int, out: Path, negative: str = "") -> GenResult:
        return self.call("generate", prompt=prompt, references=references, aspect=aspect, seed=seed, out=out, negative=negative)

    def animate(self, image: Path, prompt: str, seconds: float, aspect, seed: int, out: Path) -> GenResult:
        return self.call("animate", image=image, prompt=prompt, seconds=seconds, aspect=aspect, seed=seed, out=out)

    def speak(self, text: str, voice, out: Path) -> GenResult:
        return self.call("speak", text=text, voice=voice, out=out)

    def convert(self, audio: Path, voice, out: Path) -> GenResult:
        return self.call("convert", audio=audio, voice=voice, out=out)

    def transcribe(self, audio: Path, out: Path, hint: str | None = None) -> GenResult:
        return self.call("transcribe", audio=audio, out=out, hint=hint)

    def sync(self, video: Path, audio: Path, out: Path) -> GenResult:
        return self.call("sync", video=video, audio=audio, out=out)

    def upscale(self, video: Path, out: Path, factor: int = 2) -> GenResult:
        return self.call("upscale", video=video, out=out, factor=factor)

    def call(self, method: str, **kwargs: Any) -> GenResult:
        if not self.members:
            raise ProviderError(f"no usable model for '{self.kind}'")
        out: Path | None = kwargs.get("out")
        planned = [(member, *self._prepare(member, method, kwargs)) for member in self.members]
        for member, args, key in planned:
            hit = self.registry.cache.get(key, out)
            if hit is not None:
                saved = self._list_price(member, hit.usage)
                hit.usd = 0.0
                hit.shadow_usd = 0.0
                self.registry.ledger.record(
                    kind=self.kind,
                    provider=member.spec.provider,
                    model=member.spec.id,
                    usage=hit.usage,
                    saved_usd=saved,
                    cached=True,
                )
                return hit
        errors: list[str] = []
        for member, args, key in planned:
            if not member.breaker.closed:
                errors.append(f"{member.spec.id}: circuit open")
                continue
            result = self._attempt(member, method, args, errors)
            if result is not None:
                self.registry.cache.put(key, result)
                return result
        raise ProviderError(f"all models failed for '{self.kind}': " + " | ".join(errors))

    def _list_price(self, member: Member, usage: dict[str, float]) -> float:
        if not member.spec.offline:
            return member.spec.cost(usage)
        reference = self.registry.reference(self.kind)
        return reference.cost(usage) if reference else 0.0

    def _prepare(self, member: Member, method: str, kwargs: dict[str, Any]) -> tuple[dict[str, Any], str]:
        args = dict(kwargs)
        if "seconds" in args and self.kind == "video":
            args["seconds"] = member.spec.snap(float(args["seconds"]))
        keyed = {k: v for k, v in args.items() if k != "out"}
        key = self.registry.cache.key(self.kind, method, member.spec.id, member.spec.model, member.spec.params, keyed)
        return args, key

    def _attempt(self, member: Member, method: str, args: dict[str, Any], errors: list[str]) -> GenResult | None:
        retries = int(member.spec.params.get("retries", 2))
        for attempt in range(retries + 1):
            started = time.monotonic()
            try:
                with member.semaphore:
                    result: GenResult = getattr(member.instance, method)(**args)
            except RETRYABLE as exc:
                errors.append(f"{member.spec.id}: {exc}")
                self._fail(member, exc, started)
                if attempt < retries:
                    time.sleep(min(30.0, 1.5 * 2**attempt) + random.random())
                    continue
                return None
            except (PermanentError, ProviderError, httpx.HTTPStatusError) as exc:
                errors.append(f"{member.spec.id}: {exc}")
                self._fail(member, exc, started)
                return None
            except Exception as exc:
                log.exception("provider %s crashed", member.spec.id)
                errors.append(f"{member.spec.id}: {type(exc).__name__}: {exc}")
                self._fail(member, exc, started)
                return None
            member.breaker.success()
            result.provider = member.spec.provider
            result.model = member.spec.id
            result.seconds = round(time.monotonic() - started, 3)
            result.usd = member.spec.cost(result.usage)
            result.shadow_usd = self._list_price(member, result.usage)
            self.registry.ledger.record(
                kind=self.kind,
                provider=member.spec.provider,
                model=member.spec.id,
                usage=result.usage,
                usd=result.usd,
                shadow_usd=result.shadow_usd,
                seconds=result.seconds,
            )
            return result
        return None

    def _fail(self, member: Member, exc: Exception, started: float) -> None:
        member.breaker.failure()
        self.registry.ledger.record(
            kind=self.kind,
            provider=member.spec.provider,
            model=member.spec.id,
            seconds=round(time.monotonic() - started, 3),
            ok=False,
            error=f"{type(exc).__name__}: {exc}"[:400],
        )


class Registry:
    KINDS = ("llm", "image", "video", "speech", "voice", "transcribe", "lipsync", "upscale")

    def __init__(self, settings: Settings, ledger: Ledger, cache: Cache, offline: bool | None = None) -> None:
        load_all()
        self.settings = settings
        self.ledger = ledger
        self.cache = cache
        self.offline = settings.offline if offline is None else offline
        self._chains: dict[str, Chain] = {}
        for kind in self.KINDS:
            self._chains[kind] = Chain(kind, self._select(kind), self)

    def _select(self, kind: str) -> list[Member]:
        members = [Member(spec, self.settings) for spec in self.settings.catalog.chain(kind)]
        usable = [m for m in members if m.configured]
        if self.offline:
            return [m for m in usable if m.spec.offline]
        live = [m for m in usable if not m.spec.offline]
        if live:
            return live
        return [m for m in usable if m.spec.offline]

    def chain(self, kind: str) -> Chain:
        return self._chains[kind]

    @property
    def llm(self) -> Chain:
        return self._chains["llm"]

    @property
    def image(self) -> Chain:
        return self._chains["image"]

    @property
    def video(self) -> Chain:
        return self._chains["video"]

    @property
    def speech(self) -> Chain:
        return self._chains["speech"]

    @property
    def voice(self) -> Chain:
        return self._chains["voice"]

    @property
    def transcriber(self) -> Chain:
        return self._chains["transcribe"]

    @property
    def lipsync(self) -> Chain:
        return self._chains["lipsync"]

    @property
    def upscaler(self) -> Chain:
        return self._chains["upscale"]

    def reference(self, kind: str) -> ModelSpec | None:
        for spec in self.settings.catalog.chain(kind):
            if not spec.offline:
                return spec
        return None

    def estimate(self, kind: str, usage: dict[str, float]) -> float:
        reference = self.reference(kind)
        if reference is None:
            chain = self._chains[kind]
            reference = chain.members[0].spec if chain.members else None
        return reference.cost(usage) if reference else 0.0

    @property
    def mode(self) -> str:
        used = [c for c in self._chains.values() if c.available and c.kind in {"image", "video", "speech", "voice"}]
        if all(c.offline for c in used):
            return "offline"
        if any(c.offline for c in used):
            return "hybrid"
        return "live"

    def describe(self) -> dict[str, list[str]]:
        return {kind: chain.names for kind, chain in self._chains.items()}
