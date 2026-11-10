from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import math
import mimetypes
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import httpx
from PIL import Image, UnidentifiedImageError

from mitosis.config import ModelSpec, Settings
from mitosis.models import Aspect
from mitosis.providers.base import PermanentError, Provider, ProviderError, TransientError

log = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_STATUS = frozenset({408, 409, 425, 429})
CONNECT_TIMEOUT = 15.0
MAX_POLL_INTERVAL = 15.0
IMAGE_MEDIA_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "GIF": "image/gif"}
ALPHA_MODES = frozenset({"RGBA", "LA", "PA"})


def env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise PermanentError(f"environment variable {name} is not set")
    return value


def describe(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text.strip()[:400]
    return json.dumps(body, ensure_ascii=False)[:400]


def raise_for_status(response: httpx.Response, service: str) -> httpx.Response:
    if response.is_success:
        return response
    response.read()
    status = response.status_code
    message = f"{service} returned HTTP {status}: {describe(response)}"
    if status in RETRYABLE_STATUS or status >= 500:
        raise TransientError(message)
    raise PermanentError(message)


def payload(response: httpx.Response, service: str) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise TransientError(f"{service} sent a non-JSON body: {response.text[:200]}") from exc


def dig(data: Any, path: Sequence[str | int]) -> Any:
    current = data
    for step in path:
        if isinstance(current, Mapping) and step in current:
            current = current[step]
        elif isinstance(current, list) and isinstance(step, int) and -len(current) <= step < len(current):
            current = current[step]
        else:
            return None
    return current


def poll(probe: Callable[[], T | None], timeout: float, interval: float, what: str) -> T:
    deadline = time.monotonic() + timeout
    delay = interval
    while True:
        outcome = probe()
        if outcome is not None:
            return outcome
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TransientError(f"{what} still unfinished after {timeout:.0f}s")
        time.sleep(min(delay, remaining))
        delay = min(delay * 1.5, MAX_POLL_INTERVAL)


def write_data_uri(uri: str, out: Path) -> Path:
    header, _, encoded = uri.partition(",")
    if ";base64" not in header:
        raise PermanentError("inline result is not base64 encoded")
    try:
        out.write_bytes(base64.b64decode(encoded, validate=True))
    except (binascii.Error, ValueError) as exc:
        raise PermanentError(f"inline result is not valid base64: {exc}") from exc
    return out


def download(client: httpx.Client, url: str, out: Path, service: str) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if url.startswith("data:"):
        return write_data_uri(url, out)
    partial = out.with_name(f".{out.name}.part")
    try:
        with client.stream("GET", url) as response:
            raise_for_status(response, service)
            with partial.open("wb") as handle:
                for chunk in response.iter_bytes(1 << 20):
                    handle.write(chunk)
    except httpx.TransportError as exc:
        partial.unlink(missing_ok=True)
        raise TransientError(f"{service} download of {url} failed: {type(exc).__name__}: {exc}") from exc
    except (ProviderError, OSError):
        partial.unlink(missing_ok=True)
        raise
    if partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        raise TransientError(f"{service} returned an empty file from {url}")
    partial.replace(out)
    return out


def binary(name: str) -> str:
    return os.environ.get(f"MITOSIS_{name.upper()}") or shutil.which(name) or name


def media_duration(path: Path) -> float | None:
    command = [
        binary("ffprobe"),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
        return round(float(completed.stdout.strip()), 3)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        log.warning("could not probe duration of %s: %s", path, exc)
        return None


def to_wav(source: Path, out: Path, sample_rate: int = 48000, input_args: Sequence[str] = ()) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        binary("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *input_args,
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(out),
    ]
    try:
        subprocess.run(command, capture_output=True, text=True, timeout=300, check=True)
    except subprocess.CalledProcessError as exc:
        raise PermanentError(f"ffmpeg could not convert {source.name}: {exc.stderr.strip()[-300:]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise TransientError(f"ffmpeg timed out converting {source.name}") from exc
    except OSError as exc:
        raise PermanentError(f"ffmpeg is unavailable: {exc}") from exc
    return out


def media_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def has_alpha(image: Image.Image) -> bool:
    return image.mode in ALPHA_MODES or (image.mode == "P" and "transparency" in image.info)


def reencode(image: Image.Image, prefer_png: bool) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    if prefer_png and has_alpha(image):
        image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue(), "image/png"
    image.convert("RGB").save(buffer, format="JPEG", quality=90, optimize=True)
    return buffer.getvalue(), "image/jpeg"


def encode_image(path: Path, max_edge: int | None = None, max_bytes: int | None = None) -> tuple[bytes, str]:
    try:
        raw = path.read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            image_format = (image.format or "").upper()
            within_edge = max_edge is None or max(image.size) <= max_edge
            within_bytes = max_bytes is None or len(raw) <= max_bytes
            if within_edge and within_bytes and image_format in IMAGE_MEDIA_TYPES:
                return raw, IMAGE_MEDIA_TYPES[image_format]
            image.load()
            canvas = image.copy()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise PermanentError(f"cannot read image {path}: {exc}") from exc
    if max_edge is not None and max(canvas.size) > max_edge:
        canvas.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return reencode(canvas, prefer_png=max_bytes is None)


def data_uri(path: Path, max_bytes: int | None = None) -> str:
    raw, mime = encode_image(path, max_bytes=max_bytes)
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def aspect_value(mapping: Mapping[str, Any] | None, aspect: Aspect) -> Any:
    if mapping and aspect.value in mapping:
        return mapping[aspect.value]
    return aspect.value



def map_arguments(fields: Mapping[str, str | None], fixed: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    arguments = dict(fixed)
    for name, value in values.items():
        field = fields.get(name)
        if field and value is not None and value != "":
            arguments[field] = value
    return arguments


def billed_seconds(seconds: float) -> int:
    return max(1, math.ceil(seconds - 1e-6))


def duration_value(params: Mapping[str, Any], seconds: int) -> str | int:
    if params.get("duration_numeric", False):
        return seconds
    return str(params.get("duration_template", "{seconds}")).format(seconds=seconds)


class HttpProvider(Provider):
    service = "http"
    transport: httpx.BaseTransport | None = None

    def __init__(self, spec: ModelSpec, settings: Settings) -> None:
        super().__init__(spec, settings)
        self._clients: dict[str, httpx.Client] = {}
        self._clients_lock = threading.Lock()

    def base_url(self) -> str:
        return ""

    def auth_headers(self) -> dict[str, str]:
        return {}

    @property
    def params(self) -> dict[str, Any]:
        return self.spec.params

    @property
    def api(self) -> httpx.Client:
        return self._client("api", self.base_url, self.auth_headers)

    @property
    def downloads(self) -> httpx.Client:
        return self._client("downloads", lambda: "", lambda: {})

    def _client(self, name: str, base_url: Callable[[], str], headers: Callable[[], dict[str, str]]) -> httpx.Client:
        with self._clients_lock:
            client = self._clients.get(name)
            if client is None:
                client = httpx.Client(
                    base_url=base_url(),
                    headers=headers(),
                    timeout=httpx.Timeout(float(self.params.get("http_timeout", 120.0)), connect=CONNECT_TIMEOUT),
                    transport=self.transport,
                    follow_redirects=True,
                )
                self._clients[name] = client
            return client

    def check(self, response: httpx.Response) -> httpx.Response:
        return raise_for_status(response, self.service)

    def call(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self.api.request(method, url, **kwargs)
        except httpx.TransportError as exc:
            raise TransientError(f"{self.service} {method} {url} failed: {type(exc).__name__}: {exc}") from exc
        return self.check(response)

    def call_json(self, method: str, url: str, **kwargs: Any) -> Any:
        return payload(self.call(method, url, **kwargs), self.service)

    def save(self, url: str, out: Path) -> Path:
        return download(self.downloads, url, out, self.service)

    def wait(self, probe: Callable[[], T | None], what: str) -> T:
        return poll(probe, self.spec.timeout, float(self.params.get("poll_interval", 3.0)), what)
