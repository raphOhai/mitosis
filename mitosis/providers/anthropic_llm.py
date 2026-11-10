from __future__ import annotations

import base64
import copy
import json
import logging
import threading
from pathlib import Path
from typing import Any

import anthropic

from mitosis.config import ModelSpec, Settings
from mitosis.providers.base import GenResult, PermanentError, Provider, TransientError, register
from mitosis.providers.remote import encode_image

log = logging.getLogger(__name__)

MAX_IMAGE_EDGE = 1568
STREAMING_THRESHOLD = 16000
FALLBACK_BETA = "server-side-fallback-2026-07-01"
RETRYABLE_STATUS = frozenset({408, 409, 429})
SCHEMA_MAPS = ("properties", "$defs", "definitions", "patternProperties")
SCHEMA_LISTS = ("anyOf", "allOf", "oneOf", "prefixItems")
SCHEMA_NODES = ("items", "additionalProperties", "not", "contains")
TRANSIENT_ERRORS = (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APITimeoutError, anthropic.APIConnectionError)
PERMANENT_ERRORS = (
    anthropic.BadRequestError,
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
    anthropic.UnprocessableEntityError,
    anthropic.RequestTooLargeError,
)


def image_block(path: Path, max_edge: int = MAX_IMAGE_EDGE) -> dict[str, Any]:
    raw, media_type = encode_image(path, max_edge=max_edge)
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": base64.standard_b64encode(raw).decode("ascii")},
    }


def is_object_schema(node: dict[str, Any]) -> bool:
    declared = node.get("type")
    types = declared if isinstance(declared, list) else [declared]
    return "object" in types or "properties" in node


def close_objects(node: Any) -> None:
    if not isinstance(node, dict):
        return
    if is_object_schema(node):
        node.setdefault("additionalProperties", False)
    for key in SCHEMA_MAPS:
        children = node.get(key)
        if isinstance(children, dict):
            for child in children.values():
                close_objects(child)
    for key in SCHEMA_LISTS:
        children = node.get(key)
        if isinstance(children, list):
            for child in children:
                close_objects(child)
    for key in SCHEMA_NODES:
        close_objects(node.get(key))


def closed_schema(schema: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(schema)
    close_objects(result)
    return result


def fallback_setting(value: Any) -> str | list[dict[str, str]] | None:
    if value is True:
        return "default"
    if isinstance(value, str) and value:
        return value
    if isinstance(value, list) and value:
        return [{"model": str(model)} for model in value]
    return None


def error_text(exc: anthropic.AnthropicError) -> str:
    request_id = getattr(exc, "request_id", None)
    suffix = f" (request {request_id})" if request_id else ""
    return f"anthropic {type(exc).__name__}: {exc}{suffix}"[:600]


def ensure_complete(message: Any) -> None:
    reason = getattr(message, "stop_reason", None)
    if reason == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        explanation = getattr(details, "explanation", None) if details else None
        raise PermanentError(f"claude refused the request (category={category}): {explanation or 'no explanation'}")
    if reason == "max_tokens":
        raise TransientError("claude hit max_tokens before finishing the JSON")


def parse_json(message: Any) -> Any:
    text = next((block.text for block in message.content if getattr(block, "type", None) == "text"), None)
    if text is None:
        raise TransientError(f"claude returned no text block (stop_reason={getattr(message, 'stop_reason', None)})")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise TransientError(f"claude returned invalid JSON: {exc}") from exc


def usage_of(message: Any) -> dict[str, float]:
    usage = message.usage
    totals = {
        "input_mtok": (getattr(usage, "input_tokens", 0) or 0) / 1e6,
        "output_mtok": (getattr(usage, "output_tokens", 0) or 0) / 1e6,
    }
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    if cache_write:
        totals["cache_write_mtok"] = cache_write / 1e6
    if cache_read:
        totals["cache_read_mtok"] = cache_read / 1e6
    return totals


@register("anthropic", "llm")
class AnthropicLLM(Provider):
    def __init__(self, spec: ModelSpec, settings: Settings) -> None:
        super().__init__(spec, settings)
        self.client: Any = None
        self._lock = threading.Lock()

    def sdk(self) -> Any:
        with self._lock:
            if self.client is None:
                try:
                    self.client = anthropic.Anthropic(
                        timeout=self.spec.timeout,
                        max_retries=int(self.spec.params.get("sdk_retries", 1)),
                    )
                except anthropic.AnthropicError as exc:
                    raise PermanentError(error_text(exc)) from exc
            return self.client

    def json(
        self,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        images: list[Path] | None = None,
        max_tokens: int = 8000,
    ) -> GenResult:
        request = self.build_request(system, prompt, schema, images or [], max_tokens)
        message = self.send(request, stream=max_tokens > STREAMING_THRESHOLD)
        ensure_complete(message)
        served_by = getattr(message, "model", None) or self.spec.model
        if served_by != self.spec.model:
            log.info("claude request answered by fallback model %s", served_by)
        return GenResult(
            data=parse_json(message),
            usage=usage_of(message),
            meta={
                "stop_reason": message.stop_reason,
                "served_by": served_by,
                "request_id": getattr(message, "_request_id", None),
            },
        )

    def build_request(
        self,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        images: list[Path],
        max_tokens: int,
    ) -> dict[str, Any]:
        max_edge = int(self.spec.params.get("max_image_edge", MAX_IMAGE_EDGE))
        content = [image_block(path, max_edge) for path in images]
        content.append({"type": "text", "text": prompt})
        request: dict[str, Any] = {
            "model": self.spec.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "output_config": {
                "format": {"type": "json_schema", "schema": closed_schema(schema)},
                "effort": self.spec.params.get("effort", "medium"),
            },
        }
        fallbacks = fallback_setting(self.spec.params.get("fallbacks", True))
        if fallbacks is not None:
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = fallbacks
        return request

    def send(self, request: dict[str, Any], stream: bool) -> Any:
        client = self.sdk()
        endpoint = client.beta.messages if "betas" in request else client.messages
        try:
            if stream:
                with endpoint.stream(**request) as events:
                    return events.get_final_message()
            return endpoint.create(**request)
        except TRANSIENT_ERRORS as exc:
            raise TransientError(error_text(exc)) from exc
        except PERMANENT_ERRORS as exc:
            raise PermanentError(error_text(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code in RETRYABLE_STATUS or exc.status_code >= 500:
                raise TransientError(error_text(exc)) from exc
            raise PermanentError(error_text(exc)) from exc
        except anthropic.APIError as exc:
            raise TransientError(error_text(exc)) from exc
        except anthropic.AnthropicError as exc:
            raise PermanentError(error_text(exc)) from exc
