from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator

from mitosis.models import Event

log = logging.getLogger("mitosis")

Listener = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._listeners: list[Listener] = []
        self._lock = threading.Lock()
        self.history: list[Event] = []

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def emit(self, type: str, **fields) -> Event:
        event = Event(type=type, **fields)
        with self._lock:
            self.history.append(event)
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:
                log.exception("event listener failed")
        return event

    @contextmanager
    def stage(self, name: str, message: str = "") -> Iterator[None]:
        started = time.monotonic()
        self.emit("stage", stage=name, status="start", message=message)
        try:
            yield
        except Exception as exc:
            self.emit("error", stage=name, status="error", message=str(exc))
            raise
        self.emit("stage", stage=name, status="done", data={"seconds": round(time.monotonic() - started, 2)})
