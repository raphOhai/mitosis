from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import fal_client
import httpx

from mitosis.config import ModelSpec, Settings
from mitosis.models import Aspect
from mitosis.providers.base import GenResult, PermanentError, ProviderError, TransientError, register
from mitosis.providers.remote import (
    HttpProvider,
    aspect_value,
    billed_seconds,
    dig,
    duration_value,
    env,
    map_arguments,
    media_duration,
    raise_for_status,
)

log = logging.getLogger(__name__)

QUEUE_URL = "https://queue.fal.run"
SEED_LIMIT = 2_147_483_647
PENDING_STATES = frozenset({"IN_QUEUE", "IN_PROGRESS"})
PERMANENT_ERROR_TYPES = frozenset({"content_policy_violation", "validation_error", "bad_request", "file_download_error"})
TRUTHY = frozenset({"1", "true", "yes"})

DEFAULT_FIELDS: dict[str, dict[str, str | None]] = {
    "image": {"prompt": "prompt", "images": "image_urls", "aspect": "aspect_ratio", "seed": "seed", "negative": None},
    "video": {"prompt": "prompt", "image": "image_url", "duration": "duration", "aspect": None, "seed": "seed"},
    "lipsync": {"video": "video_url", "audio": "audio_url"},
    "upscale": {"video": "video_url", "factor": "upscale_factor"},
}
DEFAULT_RESULTS: dict[str, list[str | int]] = {
    "image": ["images", 0, "url"],
    "video": ["video", "url"],
    "lipsync": ["video", "url"],
    "upscale": ["video", "url"],
}


def upload_error(exc: Exception, path: Path) -> ProviderError:
    message = f"fal upload of {path.name} failed: {type(exc).__name__}: {exc}"
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500 and status not in {408, 429}:
        return PermanentError(message)
    return TransientError(message)


def failed_status(status: dict[str, Any]) -> ProviderError:
    error_type = str(status.get("error_type") or "")
    message = f"fal request failed ({error_type or 'unknown'}): {status.get('error')}"
    if error_type in PERMANENT_ERROR_TYPES:
        return PermanentError(message)
    return TransientError(message)


class FalQueue(HttpProvider):
    service = "fal"

    def __init__(self, spec: ModelSpec, settings: Settings) -> None:
        super().__init__(spec, settings)
        self._uploads: dict[tuple[str, int, int], tuple[str, float]] = {}
        self._uploads_lock = threading.Lock()

    def base_url(self) -> str:
        return str(self.params.get("queue_url", QUEUE_URL))

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Key {env('FAL_KEY')}"}

    def check(self, response: httpx.Response) -> httpx.Response:
        if not response.is_success and response.headers.get("x-fal-needs-retry", "").lower() in TRUTHY:
            response.read()
            raise TransientError(f"fal asked for a retry (HTTP {response.status_code}): {response.text[:300]}")
        return raise_for_status(response, self.service)

    def fields(self) -> dict[str, str | None]:
        return {**DEFAULT_FIELDS.get(self.spec.kind, {}), **self.params.get("fields", {})}

    def arguments(self, values: dict[str, Any]) -> dict[str, Any]:
        return map_arguments(self.fields(), self.params.get("extra", {}), values)

    def upload(self, path: Path) -> str:
        try:
            stat = path.stat()
        except OSError as exc:
            raise PermanentError(f"input file {path} is not readable: {exc}") from exc
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        ttl = float(self.params.get("upload_ttl", 3600.0))
        with self._uploads_lock:
            cached = self._uploads.get(key)
        if cached and time.monotonic() - cached[1] < ttl:
            return cached[0]
        url = self.store(path)
        with self._uploads_lock:
            self._uploads[key] = (url, time.monotonic())
        return url

    def store(self, path: Path) -> str:
        env("FAL_KEY")
        try:
            return fal_client.upload_file(path)
        except (fal_client.FalClientError, httpx.HTTPError, OSError) as exc:
            raise upload_error(exc, path) from exc

    def run(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], str]:
        ticket = self.call_json("POST", f"/{self.spec.model}", json=arguments)
        if not isinstance(ticket, dict) or not ticket.get("status_url") or not ticket.get("response_url"):
            raise TransientError(f"fal queue returned an unexpected ticket: {str(ticket)[:300]}")
        request_id = str(ticket.get("request_id", ""))
        log.info("fal %s queued as %s", self.spec.model, request_id)
        try:
            status = self.wait(lambda: self.progress(ticket["status_url"]), f"fal request {request_id}")
        except TransientError:
            self.cancel(ticket)
            raise
        result = self.call_json("GET", ticket["response_url"])
        if status.get("error"):
            raise failed_status(status)
        if not isinstance(result, dict):
            raise PermanentError(f"fal returned an unexpected result: {str(result)[:300]}")
        return result, request_id

    def progress(self, status_url: str) -> dict[str, Any] | None:
        status = self.call_json("GET", status_url)
        state = status.get("status") if isinstance(status, dict) else None
        if state in PENDING_STATES:
            return None
        if state == "COMPLETED":
            return status
        raise TransientError(f"fal reported unexpected status {state!r}")

    def cancel(self, ticket: dict[str, Any]) -> None:
        url = ticket.get("cancel_url")
        if not url:
            return
        try:
            self.call("PUT", url)
        except ProviderError as exc:
            log.warning("could not cancel fal request %s: %s", ticket.get("request_id"), exc)

    def media_url(self, result: dict[str, Any]) -> str:
        flags = result.get("has_nsfw_concepts")
        if isinstance(flags, list) and any(flags):
            raise PermanentError("fal safety checker flagged the generated media")
        url = dig(result, self.params.get("result", DEFAULT_RESULTS[self.spec.kind]))
        if not isinstance(url, str) or not url:
            raise PermanentError(f"fal result has no media URL: {json.dumps(result)[:300]}")
        return url

    def meta(self, request_id: str, result: dict[str, Any]) -> dict[str, Any]:
        return {"endpoint": self.spec.model, "request_id": request_id, "seed": result.get("seed")}


@register("fal", "image")
class FalImage(FalQueue):
    def generate(
        self,
        prompt: str,
        references: list[Path],
        aspect: Aspect,
        seed: int,
        out: Path,
        negative: str = "",
    ) -> GenResult:
        if not references:
            raise PermanentError(f"{self.spec.model} needs at least one hero reference image")
        limit = int(self.params.get("max_references", 4))
        urls = [self.upload(path) for path in references[:limit]]
        mapping = self.fields()
        values: dict[str, Any] = {
            "prompt": prompt,
            "images": urls if mapping.get("images") else None,
            "image": urls[0] if mapping.get("image") else None,
            "aspect": aspect_value(self.params.get("aspects"), aspect),
            "seed": seed % SEED_LIMIT,
            "negative": negative,
        }
        result, request_id = self.run(self.arguments(values))
        self.save(self.media_url(result), out)
        return GenResult(path=out, usage={"image": 1}, meta=self.meta(request_id, result))


@register("fal", "video")
class FalVideo(FalQueue):
    def animate(
        self,
        image: Path,
        prompt: str,
        seconds: float,
        aspect: Aspect,
        seed: int,
        out: Path,
    ) -> GenResult:
        billed = billed_seconds(seconds)
        values: dict[str, Any] = {
            "prompt": prompt,
            "image": self.upload(image),
            "duration": duration_value(self.params, billed),
            "aspect": aspect_value(self.params.get("aspects"), aspect),
            "seed": seed % SEED_LIMIT,
        }
        result, request_id = self.run(self.arguments(values))
        self.save(self.media_url(result), out)
        duration = media_duration(out) or float(billed)
        return GenResult(path=out, duration=duration, usage={"second": float(billed)}, meta=self.meta(request_id, result))


@register("fal", "lipsync")
class FalLipSync(FalQueue):
    def sync(self, video: Path, audio: Path, out: Path) -> GenResult:
        values = {"video": self.upload(video), "audio": self.upload(audio)}
        result, request_id = self.run(self.arguments(values))
        self.save(self.media_url(result), out)
        duration = media_duration(out) or media_duration(audio) or 0.0
        return GenResult(path=out, duration=duration, usage={"second": duration}, meta=self.meta(request_id, result))


@register("fal", "upscale")
class FalUpscale(FalQueue):
    def upscale(self, video: Path, out: Path, factor: int = 2) -> GenResult:
        source_seconds = media_duration(video) or 0.0
        values = {"video": self.upload(video), "factor": factor}
        result, request_id = self.run(self.arguments(values))
        self.save(self.media_url(result), out)
        duration = media_duration(out) or source_seconds
        return GenResult(
            path=out,
            duration=duration,
            usage={"second": source_seconds or duration},
            meta=self.meta(request_id, result),
        )
