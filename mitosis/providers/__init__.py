from __future__ import annotations

import importlib
import logging

log = logging.getLogger(__name__)

MODULES = ("offline", "anthropic_llm", "fal", "elevenlabs", "runway", "higgsfield", "comfyui")


def load_all() -> None:
    for name in MODULES:
        try:
            importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as exc:
            log.debug("provider module %s unavailable: %s", name, exc)
