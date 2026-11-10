from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from mitosis.providers.base import GenResult


class Cache:
    def __init__(self, root: Path, enabled: bool = True) -> None:
        self.root = root
        self.enabled = enabled
        self._digests: dict[tuple[str, int, int], str] = {}
        self._lock = threading.Lock()
        root.mkdir(parents=True, exist_ok=True)

    def digest(self, path: Path) -> str:
        stat = path.stat()
        marker = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        with self._lock:
            if marker in self._digests:
                return self._digests[marker]
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                sha.update(block)
        value = sha.hexdigest()
        with self._lock:
            self._digests[marker] = value
        return value

    def normalize(self, value: Any) -> Any:
        if isinstance(value, Path):
            return {"file": self.digest(value)} if value.exists() else {"path": str(value)}
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, BaseModel):
            return self.normalize(value.model_dump(mode="json"))
        if isinstance(value, dict):
            return {str(k): self.normalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
        if isinstance(value, (list, tuple)):
            return [self.normalize(v) for v in value]
        if isinstance(value, float):
            return round(value, 4)
        return value

    def key(self, *parts: Any) -> str:
        payload = json.dumps(self.normalize(list(parts)), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def _slot(self, key: str) -> Path:
        slot = self.root / key[:2]
        slot.mkdir(parents=True, exist_ok=True)
        return slot

    def get(self, key: str, out: Path | None = None) -> GenResult | None:
        if not self.enabled:
            return None
        meta = self._slot(key) / f"{key}.json"
        if not meta.exists():
            return None
        record = json.loads(meta.read_text())
        stored = record.pop("_file", None)
        result = GenResult(**record)
        if stored:
            source = self._slot(key) / stored
            if not source.exists():
                return None
            target = out or source
            if out is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    target.unlink()
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
            result.path = target
        result.cached = True
        return result

    def put(self, key: str, result: GenResult) -> None:
        if not self.enabled:
            return
        slot = self._slot(key)
        record = result.model_dump(mode="json", exclude={"path", "cached"})
        if result.path and Path(result.path).exists():
            stored = f"{key}{Path(result.path).suffix}"
            shutil.copy2(result.path, slot / stored)
            record["_file"] = stored
        (slot / f"{key}.json").write_text(json.dumps(record))
