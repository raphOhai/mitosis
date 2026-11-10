from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mitosis.providers.base import LanguageModel, ProviderError

log = logging.getLogger(__name__)


def strict_object(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}


def string(enum: list[str] | None = None) -> dict[str, Any]:
    return {"type": "string", "enum": enum} if enum else {"type": "string"}


def array(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


def ask_json(
    llm: LanguageModel | None,
    system: str,
    prompt: str,
    schema: dict[str, Any],
    images: list[Path] | None = None,
    max_tokens: int = 4000,
) -> dict[str, Any] | None:
    if llm is None:
        return None
    try:
        result = llm.json(system=system, prompt=prompt, schema=schema, images=list(images or []), max_tokens=max_tokens)
    except ProviderError as exc:
        log.warning("llm call failed, using deterministic path: %s", exc)
        return None
    data = result.data
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            log.warning("llm returned non-json text, using deterministic path")
            return None
    if not isinstance(data, dict):
        log.warning("llm returned %s instead of an object, using deterministic path", type(data).__name__)
        return None
    return data
