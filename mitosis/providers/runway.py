from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mitosis.models import Aspect
from mitosis.providers.base import GenResult, PermanentError, ProviderError, TransientError, register
from mitosis.providers.remote import HttpProvider, aspect_value, billed_seconds, data_uri, env, media_duration

log = logging.getLogger(__name__)

API_URL = "https://api.dev.runwayml.com"
API_VERSION = "2024-11-06"
DATA_URI_BYTES = 3_000_000
SEED_LIMIT = 4_294_967_295
PROMPT_LIMIT = 1000
PENDING_STATES = frozenset({"PENDING", "THROTTLED", "RUNNING"})
RETRYABLE_FAILURES = ("INTERNAL", "THIRD_PARTY", "INPUT_PREPROCESSING.INTERNAL")
DEFAULT_RATIOS = {"9:16": "720:1280", "4:5": "832:1104", "1:1": "960:960"}


def task_failure(task: dict[str, Any]) -> ProviderError:
    code = str(task.get("failureCode") or "")
    message = f"runway task {task.get('id')} failed ({code or 'no code'}): {task.get('failure') or 'no detail'}"
    if not code or code.startswith(RETRYABLE_FAILURES):
        return TransientError(message)
    return PermanentError(message)


@register("runway", "video")
class RunwayVideo(HttpProvider):
    service = "runway"

    def base_url(self) -> str:
        return str(self.params.get("api_url", API_URL))

    def auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {env('RUNWAYML_API_SECRET')}",
            "X-Runway-Version": str(self.params.get("api_version", API_VERSION)),
        }

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
        body = self.request_body(image, prompt, duration, aspect, seed)
        created = self.call_json("POST", "/v1/image_to_video", json=body)
        task_id = created.get("id") if isinstance(created, dict) else None
        if not task_id:
            raise TransientError(f"runway did not return a task id: {str(created)[:300]}")
        log.info("runway %s task %s started", self.spec.model, task_id)
        try:
            task = self.wait(lambda: self.progress(task_id), f"runway task {task_id}")
        except TransientError:
            self.cancel(task_id)
            raise
        self.save(task["output"][0], out)
        return GenResult(
            path=out,
            duration=media_duration(out) or float(duration),
            usage={"second": float(duration)},
            meta={"task_id": task_id, "model": self.spec.model},
        )

    def request_body(self, image: Path, prompt: str, duration: int, aspect: Aspect, seed: int) -> dict[str, Any]:
        ratios = {**DEFAULT_RATIOS, **self.params.get("ratios", {})}
        body: dict[str, Any] = {
            "model": self.spec.model,
            "promptImage": [{"uri": data_uri(image, max_bytes=DATA_URI_BYTES), "position": "first"}],
            "ratio": aspect_value(ratios, aspect),
            "duration": duration,
            "seed": seed % SEED_LIMIT,
        }
        if prompt.strip():
            body["promptText"] = prompt.strip()[:PROMPT_LIMIT]
        body.update(self.params.get("extra", {}))
        return body

    def progress(self, task_id: str) -> dict[str, Any] | None:
        task = self.call_json("GET", f"/v1/tasks/{task_id}")
        status = task.get("status") if isinstance(task, dict) else None
        if status in PENDING_STATES:
            return None
        if status == "SUCCEEDED":
            outputs = task.get("output") or []
            if not outputs:
                raise TransientError(f"runway task {task_id} succeeded without output")
            return task
        if status == "FAILED":
            raise task_failure(task)
        if status == "CANCELLED":
            raise PermanentError(f"runway task {task_id} was cancelled")
        raise TransientError(f"runway task {task_id} reported unexpected status {status!r}")

    def cancel(self, task_id: str) -> None:
        try:
            self.call("DELETE", f"/v1/tasks/{task_id}")
        except ProviderError as exc:
            log.warning("could not cancel runway task %s: %s", task_id, exc)
