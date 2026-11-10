from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import httpx

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
    media_type,
    raise_for_status,
)

log = logging.getLogger(__name__)

API_URL = "https://api.higgsfield.ai"
UPLOAD_PATH = "/files/generate-upload-url"
PENDING_STATES = frozenset({"queued", "in_progress"})
SEED_LIMIT = 2_147_483_647
DEFAULT_FIELDS: dict[str, str | None] = {
    "prompt": "prompt",
    "image": "image_url",
    "duration": "duration",
    "aspect": "aspect_ratio",
    "seed": None,
}
DEFAULT_RESULT: list[str | int] = ["video", "url"]


@register("higgsfield", "video")
class HiggsfieldVideo(HttpProvider):
    service = "higgsfield"

    def base_url(self) -> str:
        return str(self.params.get("api_url", API_URL))

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Key {env('HIGGSFIELD_API_KEY')}:{env('HIGGSFIELD_API_SECRET')}"}

    def animate(
        self,
        image: Path,
        prompt: str,
        seconds: float,
        aspect: Aspect,
        seed: int,
        out: Path,
    ) -> GenResult:
        duration = billed_seconds(seconds)
        fields = {**DEFAULT_FIELDS, **self.params.get("fields", {})}
        values = {
            "prompt": prompt,
            "image": self.upload(image),
            "duration": duration_value(self.params, duration),
            "aspect": aspect_value(self.params.get("aspects"), aspect),
            "seed": seed % SEED_LIMIT,
        }
        arguments = map_arguments(fields, self.params.get("extra", {}), values)
        ticket = self.call_json(
            "POST",
            f"/{self.spec.model.strip('/')}",
            json=arguments,
            headers={"Idempotency-Key": str(uuid.uuid4())},
        )
        request_id = str(ticket.get("request_id") or "") if isinstance(ticket, dict) else ""
        if not request_id:
            raise TransientError(f"higgsfield did not return a request id: {str(ticket)[:300]}")
        status_url = ticket.get("status_url") or f"/requests/{request_id}/status"
        log.info("higgsfield %s queued as %s", self.spec.model, request_id)
        try:
            status = self.wait(lambda: self.progress(status_url, request_id), f"higgsfield request {request_id}")
        except TransientError:
            self.cancel(ticket.get("cancel_url") or f"/requests/{request_id}/cancel", request_id)
            raise
        url = dig(status, self.params.get("result", DEFAULT_RESULT))
        if not isinstance(url, str) or not url:
            raise PermanentError(f"higgsfield request {request_id} completed without a video URL")
        self.save(url, out)
        return GenResult(
            path=out,
            duration=media_duration(out) or float(duration),
            usage={"second": float(duration)},
            meta={"request_id": request_id, "endpoint": self.spec.model},
        )

    def upload(self, path: Path) -> str:
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise PermanentError(f"input file {path} is not readable: {exc}") from exc
        content_type = media_type(path)
        slot = self.call_json("POST", UPLOAD_PATH, json={"content_type": content_type})
        if not isinstance(slot, dict) or not slot.get("upload_url") or not slot.get("public_url"):
            raise TransientError(f"higgsfield returned an unusable upload slot: {str(slot)[:300]}")
        headers = {"Content-Type": content_type, **(slot.get("upload_headers") or {})}
        try:
            response = self.downloads.put(slot["upload_url"], content=content, headers=headers)
        except httpx.TransportError as exc:
            raise TransientError(f"higgsfield upload of {path.name} failed: {type(exc).__name__}: {exc}") from exc
        raise_for_status(response, self.service)
        return str(slot["public_url"])

    def progress(self, status_url: str, request_id: str) -> dict[str, Any] | None:
        status = self.call_json("GET", status_url)
        state = str(status.get("status", "")).lower() if isinstance(status, dict) else ""
        if state in PENDING_STATES:
            return None
        if state == "completed":
            return status
        detail = status.get("error") if isinstance(status, dict) else None
        if state == "nsfw":
            raise PermanentError(f"higgsfield request {request_id} was blocked by moderation")
        if state == "canceled":
            raise PermanentError(f"higgsfield request {request_id} was canceled")
        if state == "failed":
            raise TransientError(f"higgsfield request {request_id} failed: {detail or 'no detail'}")
        raise TransientError(f"higgsfield request {request_id} reported unexpected status {state!r}")

    def cancel(self, cancel_url: str, request_id: str) -> None:
        try:
            self.call("POST", cancel_url)
        except ProviderError as exc:
            log.warning("could not cancel higgsfield request %s: %s", request_id, exc)
