from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

from mitosis.models import LedgerEntry

_variant: ContextVar[str | None] = ContextVar("variant", default=None)
_stage: ContextVar[str | None] = ContextVar("stage", default=None)
_attempt: ContextVar[int] = ContextVar("attempt", default=1)


class Ledger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._entries: list[LedgerEntry] = []
        self._lock = threading.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")

    @contextmanager
    def scope(self, variant: str | None = None, stage: str | None = None, attempt: int | None = None) -> Iterator[None]:
        tokens = []
        if variant is not None:
            tokens.append((_variant, _variant.set(variant)))
        if stage is not None:
            tokens.append((_stage, _stage.set(stage)))
        if attempt is not None:
            tokens.append((_attempt, _attempt.set(attempt)))
        try:
            yield
        finally:
            for var, token in reversed(tokens):
                var.reset(token)

    def record(self, **fields) -> LedgerEntry:
        entry = LedgerEntry(
            ts=time.time(),
            variant=fields.pop("variant", _variant.get()),
            stage=fields.pop("stage", _stage.get()),
            attempt=fields.pop("attempt", _attempt.get()),
            **fields,
        )
        with self._lock:
            self._entries.append(entry)
            if self.path:
                with self.path.open("a") as handle:
                    handle.write(entry.model_dump_json() + "\n")
        return entry

    @property
    def entries(self) -> list[LedgerEntry]:
        with self._lock:
            return list(self._entries)

    def for_variant(self, variant: str) -> tuple[float, float, float]:
        rows = [e for e in self.entries if e.variant == variant]
        return (
            round(sum(e.usd for e in rows), 4),
            round(sum(e.shadow_usd for e in rows), 4),
            round(sum(e.seconds for e in rows), 2),
        )

    def summary(self) -> dict:
        rows = self.entries
        by_stage: dict[str, float] = defaultdict(float)
        by_provider: dict[str, float] = defaultdict(float)
        for row in rows:
            spend = row.usd or row.shadow_usd
            by_stage[row.stage or "other"] += spend
            by_provider[f"{row.provider}:{row.model}"] += spend
        return {
            "usd": round(sum(r.usd for r in rows), 4),
            "shadow_usd": round(sum(r.shadow_usd for r in rows), 4),
            "seconds": round(sum(r.seconds for r in rows), 2),
            "cache_hits": sum(1 for r in rows if r.cached),
            "saved_usd": round(sum(r.saved_usd for r in rows), 4),
            "failures": sum(1 for r in rows if not r.ok),
            "by_stage": {k: round(v, 4) for k, v in sorted(by_stage.items())},
            "by_provider": {k: round(v, 4) for k, v in sorted(by_provider.items(), key=lambda kv: -kv[1])},
        }

    @staticmethod
    def read(path: Path) -> list[LedgerEntry]:
        return [LedgerEntry(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
