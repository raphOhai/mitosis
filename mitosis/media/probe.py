from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path
from typing import Any

from mitosis.media import ff
from mitosis.models import MediaInfo

log = logging.getLogger(__name__)


def parse_rate(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number >= 0 else None


def to_int(value: Any) -> int | None:
    number = to_float(value)
    return int(number) if number is not None else None


def first_stream(streams: list[dict], kind: str) -> dict | None:
    for stream in streams:
        if stream.get("codec_type") != kind:
            continue
        if kind == "video" and stream.get("disposition", {}).get("attached_pic"):
            continue
        return stream
    return None


def rotation(stream: dict) -> int:
    for side_data in stream.get("side_data_list", []) or []:
        if "rotation" in side_data:
            return int(round(float(side_data["rotation"])))
    tag = stream.get("tags", {}).get("rotate")
    return int(tag) if tag and tag.lstrip("-").isdigit() else 0


def display_size(stream: dict) -> tuple[int, int]:
    width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
    if abs(rotation(stream)) % 180 == 90:
        return height, width
    return width, height


def stream_fps(stream: dict) -> float:
    return parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate"))


def container_duration(data: dict) -> float:
    declared = to_float(data.get("format", {}).get("duration"))
    if declared:
        return declared
    durations = [to_float(s.get("duration")) for s in data.get("streams", [])]
    known = [d for d in durations if d]
    if known:
        return max(known)
    log.warning("no duration reported for %s", data.get("format", {}).get("filename"))
    return 0.0


def probe(path: Path) -> MediaInfo:
    source = Path(path)
    data = ff.ffprobe_json(source)
    streams = data.get("streams", [])
    video = first_stream(streams, "video")
    audio = first_stream(streams, "audio")
    if video is None and audio is None:
        raise ff.FFmpegError(f"{source} has no audio or video stream")
    width, height = display_size(video) if video else (0, 0)
    return MediaInfo(
        path=source,
        width=width,
        height=height,
        fps=round(stream_fps(video), 4) if video else 0.0,
        duration=round(container_duration(data), 6),
        has_audio=audio is not None,
        sample_rate=to_int(audio.get("sample_rate")) if audio else None,
        video_codec=video.get("codec_name") if video else None,
        audio_codec=audio.get("codec_name") if audio else None,
        pix_fmt=video.get("pix_fmt") if video else None,
        bitrate=to_int(data.get("format", {}).get("bit_rate")),
    )


def duration(path: Path) -> float:
    return round(container_duration(ff.ffprobe_json(Path(path))), 6)
