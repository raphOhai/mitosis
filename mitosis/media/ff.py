from __future__ import annotations

import json
import logging
import re
import shlex
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

STDERR_TAIL_LINES = 25
FILTER_LEVEL_SPECIALS = re.compile(r"([\\'\[\],;])")


class FFmpegError(RuntimeError):
    pass


@lru_cache(maxsize=8)
def binary(name: str) -> str:
    return shutil.which(name) or name


def tail(text: str, lines: int = STDERR_TAIL_LINES) -> str:
    kept = [line for line in text.strip().splitlines() if line.strip()]
    return "\n".join(kept[-lines:])


def run(args: list[str], timeout: float = 600.0) -> subprocess.CompletedProcess:
    program = Path(args[0]).name
    log.debug("exec %s", shlex.join(args))
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise FFmpegError(f"{program} is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"{program} timed out after {timeout:.0f}s") from exc
    if completed.returncode != 0:
        raise FFmpegError(f"{program} exited with code {completed.returncode}: {tail(completed.stderr)}")
    return completed


def ffmpeg(*args: str, timeout: float = 600.0) -> subprocess.CompletedProcess:
    return run([binary("ffmpeg"), "-hide_banner", "-nostdin", "-y", *args], timeout=timeout)


def ffprobe_json(path: Path) -> dict:
    source = Path(path)
    if not source.exists():
        raise FFmpegError(f"cannot probe missing file {source}")
    completed = run(
        [binary("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(source)],
        timeout=60.0,
    )
    try:
        return json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"ffprobe returned invalid JSON for {source}") from exc


def escape_filter_value(value: str) -> str:
    option_level = value.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
    return FILTER_LEVEL_SPECIALS.sub(r"\\\1", option_level)


def seconds(value: float) -> str:
    return f"{max(0.0, value):.6f}"


@lru_cache(maxsize=1)
def available_filters() -> frozenset[str]:
    try:
        listing = run([binary("ffmpeg"), "-hide_banner", "-filters"], timeout=30.0).stdout
    except FFmpegError:
        return frozenset()
    names = set()
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) >= 3 and "->" in parts[2]:
            names.add(parts[1])
    return frozenset(names)


def has_filter(name: str) -> bool:
    return name in available_filters()
