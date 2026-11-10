from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from mitosis.config import Brand
from mitosis.models import Check, GateInput

log = logging.getLogger(__name__)

DURATION_TOLERANCE = 0.2
FPS_TOLERANCE = 0.5
BLACK_MIN_SECONDS = 0.3
BLACK_PIXEL_THRESHOLD = 0.10
FREEZE_NOISE_DB = -55
HOOK_FREEZE_SECONDS = 0.8
BODY_FREEZE_SECONDS = 1.5
TRUE_PEAK_HEADROOM = 0.5
SPEECH_START_LIMIT = 0.5
SPEECH_FLOOR_BELOW_TARGET = 14.0
SPEECH_MIN_SILENCE = 0.12
LEADING_SILENCE_EPSILON = 0.05
SCAN_WIDTH = 320
PROBE_TIMEOUT = 60.0
SCAN_TIMEOUT = 600.0
TOOL_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin")
CODEC_NAMES = {"libx264": "h264", "h264_videotoolbox": "h264", "libx265": "hevc", "hevc_videotoolbox": "hevc"}

BLACK_PATTERN = re.compile(r"black_start:\s*(?P<start>-?[\d.]+)\s+black_end:\s*(?P<end>-?[\d.]+)")
FREEZE_PATTERN = re.compile(r"lavfi\.freezedetect\.freeze_(?P<kind>start|end):\s*(?P<value>-?[\d.]+)")
SILENCE_PATTERN = re.compile(r"silence_(?P<kind>start|end):\s*(?P<value>-?[\d.]+)")
INTEGRATED_PATTERN = re.compile(r"\bI:\s+(?P<value>-?(?:[\d.]+|inf))\s+LUFS")
PEAK_PATTERN = re.compile(r"\bPeak:\s+(?P<value>-?(?:[\d.]+|inf))\s+dBFS")
RANGE_PATTERN = re.compile(r"\bLRA:\s+(?P<value>-?(?:[\d.]+|inf))\s+LU\b")


class ToolMissing(RuntimeError):
    pass


class RenderUnreadable(RuntimeError):
    pass


class ScanFailed(RuntimeError):
    pass


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    @property
    def length(self) -> float:
        return max(0.0, self.end - self.start)

    def overlap(self, low: float, high: float) -> float:
        return max(0.0, min(self.end, high) - max(self.start, low))

    def label(self) -> str:
        return f"{self.start:.2f}-{self.end:.2f}s"


@dataclass(frozen=True)
class RenderFacts:
    width: int
    height: int
    fps: float
    video_codec: str
    pix_fmt: str
    audio_codec: str | None
    sample_rate: int | None
    duration: float
    size: int

    @property
    def has_audio(self) -> bool:
        return self.audio_codec is not None


@dataclass(frozen=True)
class VideoScan:
    black: list[Interval]
    frozen: list[Interval]


@dataclass(frozen=True)
class AudioScan:
    integrated: float | None
    true_peak: float | None
    loudness_range: float | None
    silences: list[Interval]


@lru_cache(maxsize=None)
def binary(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    for directory in TOOL_DIRS:
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise ToolMissing(f"{name} not found on PATH or in {', '.join(TOOL_DIRS)}")


def run_tool(args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, errors="replace", timeout=timeout, check=False)


def tail(text: str, lines: int = 4) -> str:
    return " / ".join(line.strip() for line in text.strip().splitlines()[-lines:])


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def as_rate(value: Any) -> float:
    text = str(value or "")
    if "/" in text:
        numerator, _, denominator = text.partition("/")
        bottom = as_float(denominator)
        return as_float(numerator) / bottom if bottom else 0.0
    return as_float(text)


def probe_render(video: Path) -> RenderFacts:
    if not video.is_file():
        raise RenderUnreadable(f"{video.name} does not exist")
    command = [binary("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(video)]
    try:
        completed = run_tool(command, PROBE_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        raise RenderUnreadable(f"ffprobe timed out after {PROBE_TIMEOUT:.0f}s") from exc
    if completed.returncode != 0:
        raise RenderUnreadable(f"ffprobe failed: {tail(completed.stderr)}")
    try:
        data = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RenderUnreadable(f"ffprobe returned invalid JSON: {exc}") from exc
    return facts_from_probe(data, video)


def facts_from_probe(data: dict[str, Any], video: Path) -> RenderFacts:
    streams = data.get("streams") or []
    container = data.get("format") or {}
    picture = next(
        (s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")),
        None,
    )
    if picture is None:
        raise RenderUnreadable(f"{video.name} has no video stream")
    sound = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fps = as_rate(picture.get("avg_frame_rate")) or as_rate(picture.get("r_frame_rate"))
    stream_seconds = max((as_float(s.get("duration")) for s in streams), default=0.0)
    size = int(as_float(container.get("size"))) or video.stat().st_size
    sample_rate = int(as_float(sound.get("sample_rate"))) if sound else 0
    return RenderFacts(
        width=int(picture.get("width") or 0),
        height=int(picture.get("height") or 0),
        fps=fps,
        video_codec=str(picture.get("codec_name") or ""),
        pix_fmt=str(picture.get("pix_fmt") or ""),
        audio_codec=str(sound.get("codec_name") or "unknown") if sound else None,
        sample_rate=sample_rate or None,
        duration=as_float(container.get("duration")) or stream_seconds,
        size=size,
    )


def pair_events(events: list[tuple[str, float]], end_of_stream: float) -> list[Interval]:
    intervals: list[Interval] = []
    opened: float | None = None
    for kind, value in events:
        if kind == "start":
            opened = value
        elif kind == "end" and opened is not None:
            intervals.append(Interval(opened, value))
            opened = None
    if opened is not None:
        intervals.append(Interval(opened, max(opened, end_of_stream)))
    return intervals


def parse_video_scan(log_text: str, duration: float) -> VideoScan:
    black = [Interval(float(m["start"]), float(m["end"])) for m in BLACK_PATTERN.finditer(log_text)]
    freezes = [(m["kind"], float(m["value"])) for m in FREEZE_PATTERN.finditer(log_text)]
    return VideoScan(black=black, frozen=pair_events(freezes, duration))


def parse_audio_scan(log_text: str, duration: float) -> AudioScan:
    summary = log_text[log_text.rfind("Summary:") :] if "Summary:" in log_text else ""
    integrated = INTEGRATED_PATTERN.search(summary)
    peak = PEAK_PATTERN.search(summary)
    loudness_range = RANGE_PATTERN.search(summary)
    silences = [(m["kind"], float(m["value"])) for m in SILENCE_PATTERN.finditer(log_text)]
    return AudioScan(
        integrated=float(integrated["value"]) if integrated else None,
        true_peak=float(peak["value"]) if peak else None,
        loudness_range=float(loudness_range["value"]) if loudness_range else None,
        silences=pair_events(silences, duration),
    )


def scan(command: list[str], label: str) -> str:
    try:
        completed = run_tool(command, SCAN_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        raise ScanFailed(f"{label} scan timed out after {SCAN_TIMEOUT:.0f}s") from exc
    if completed.returncode != 0:
        raise ScanFailed(f"{label} scan failed: {tail(completed.stderr)}")
    return completed.stderr


def scan_video(video: Path, duration: float) -> VideoScan:
    filters = (
        f"scale={SCAN_WIDTH}:-2,"
        f"blackdetect=d={BLACK_MIN_SECONDS}:pix_th={BLACK_PIXEL_THRESHOLD},"
        f"freezedetect=n={FREEZE_NOISE_DB}dB:d={HOOK_FREEZE_SECONDS}"
    )
    command = [
        binary("ffmpeg"), "-hide_banner", "-nostats", "-nostdin", "-loglevel", "info",
        "-i", str(video), "-map", "0:v:0", "-vf", filters, "-an", "-f", "null", "-",
    ]
    return parse_video_scan(scan(command, "video"), duration)


def scan_audio(video: Path, duration: float, brand: Brand) -> AudioScan:
    floor = brand.loudness.target_lufs - SPEECH_FLOOR_BELOW_TARGET
    graph = (
        "[0:a:0]asplit=2[loud][voice];"
        "[loud]ebur128=peak=true:framelog=quiet[metered];"
        f"[voice]highpass=f=150,lowpass=f=4000,silencedetect=n={floor:.1f}dB:d={SPEECH_MIN_SILENCE}[gated]"
    )
    command = [
        binary("ffmpeg"), "-hide_banner", "-nostats", "-nostdin", "-loglevel", "info",
        "-i", str(video), "-filter_complex", graph,
        "-map", "[metered]", "-f", "null", "-",
        "-map", "[gated]", "-f", "null", "-",
    ]
    return parse_audio_scan(scan(command, "audio"), duration)


def expected_codec(brand: Brand) -> str:
    return CODEC_NAMES.get(brand.video.codec, brand.video.codec)


def resolution_check(facts: RenderFacts, inp: GateInput) -> Check:
    width, height = inp.aspect.size
    passed = (facts.width, facts.height) == (width, height)
    detail = f"{facts.width}x{facts.height}" + ("" if passed else f", expected {width}x{height} for {inp.aspect.value}")
    return Check(name="resolution", passed=passed, retry="none", detail=detail)


def frame_rate_check(facts: RenderFacts, brand: Brand) -> Check:
    target = float(brand.video.fps)
    passed = abs(facts.fps - target) <= FPS_TOLERANCE
    return Check(
        name="frame_rate",
        passed=passed,
        severity="warn",
        score=round(facts.fps, 3),
        threshold=target,
        detail=f"{facts.fps:.3f} fps, expected {target:g}",
    )


def video_codec_check(facts: RenderFacts, brand: Brand) -> Check:
    codec, pix_fmt = expected_codec(brand), brand.video.pix_fmt
    problems = []
    if facts.video_codec != codec:
        problems.append(f"codec {facts.video_codec or 'unknown'} (expected {codec})")
    if facts.pix_fmt != pix_fmt:
        problems.append(f"pixel format {facts.pix_fmt or 'unknown'} (expected {pix_fmt})")
    detail = "; ".join(problems) if problems else f"{facts.video_codec} {facts.pix_fmt}"
    return Check(name="video_codec", passed=not problems, retry="none", detail=detail)


def audio_stream_check(facts: RenderFacts, brand: Brand) -> Check:
    if not facts.has_audio:
        return Check(name="audio_stream", passed=False, retry="render", detail="no audio stream")
    if facts.audio_codec != "aac":
        return Check(name="audio_stream", passed=False, retry="render", detail=f"audio codec {facts.audio_codec}, expected aac")
    rate = brand.video.sample_rate
    if facts.sample_rate != rate:
        return Check(
            name="audio_stream",
            passed=False,
            severity="warn",
            detail=f"aac at {facts.sample_rate or 'unknown'} Hz, expected {rate} Hz",
        )
    return Check(name="audio_stream", passed=True, detail=f"aac {rate} Hz")


def duration_check(facts: RenderFacts, inp: GateInput) -> Check:
    drift = facts.duration - inp.expected_seconds
    passed = abs(drift) <= DURATION_TOLERANCE
    return Check(
        name="duration",
        passed=passed,
        retry="render",
        score=round(facts.duration, 3),
        threshold=round(inp.expected_seconds, 3),
        detail=f"{facts.duration:.2f}s vs expected {inp.expected_seconds:.2f}s ({drift:+.2f}s, tolerance ±{DURATION_TOLERANCE}s)",
    )


def file_size_check(facts: RenderFacts, brand: Brand) -> Check:
    limit = brand.video.max_bytes
    passed = facts.size < limit
    return Check(
        name="file_size",
        passed=passed,
        retry="none",
        score=float(facts.size),
        threshold=float(limit),
        detail=f"{facts.size / 1_000_000:.1f} MB (limit {limit / 1_000_000:.0f} MB)",
    )


def black_check(scan_result: VideoScan, inp: GateInput) -> Check:
    segments = [segment for segment in scan_result.black if segment.length > BLACK_MIN_SECONDS - 1e-3]
    if not segments:
        return Check(name="black_frames", passed=True, detail=f"no black segment longer than {BLACK_MIN_SECONDS}s")
    in_hook = any(segment.overlap(0.0, inp.hook_seconds) > 0.0 for segment in segments)
    return Check(
        name="black_frames",
        passed=False,
        retry="shot" if in_hook else "render",
        score=round(max(segment.length for segment in segments), 3),
        threshold=BLACK_MIN_SECONDS,
        detail="black at " + ", ".join(segment.label() for segment in segments),
    )


def freeze_checks(scan_result: VideoScan, inp: GateInput) -> list[Check]:
    hook_end = inp.hook_seconds
    stalled_hook = [f for f in scan_result.frozen if f.overlap(0.0, hook_end) > HOOK_FREEZE_SECONDS]
    stalled_body = [f for f in scan_result.frozen if f.overlap(hook_end, math.inf) > BODY_FREEZE_SECONDS]
    hook = Check(
        name="frozen_hook",
        passed=not stalled_hook,
        retry="shot",
        threshold=HOOK_FREEZE_SECONDS,
        score=round(max((f.overlap(0.0, hook_end) for f in stalled_hook), default=0.0), 3),
        detail=(
            "frozen video in hook at " + ", ".join(f.label() for f in stalled_hook)
            if stalled_hook
            else f"no freeze over {HOOK_FREEZE_SECONDS}s in the first {hook_end:.2f}s"
        ),
    )
    body = Check(
        name="frozen_body",
        passed=not stalled_body,
        severity="warn",
        threshold=BODY_FREEZE_SECONDS,
        score=round(max((f.overlap(hook_end, math.inf) for f in stalled_body), default=0.0), 3),
        detail=(
            "frozen video at " + ", ".join(f.label() for f in stalled_body)
            if stalled_body
            else f"no freeze over {BODY_FREEZE_SECONDS}s after the hook"
        ),
    )
    return [hook, body]


def loudness_checks(scan_result: AudioScan, brand: Brand) -> list[Check]:
    spec = brand.loudness
    integrated = scan_result.integrated
    if integrated is None:
        level = Check(name="loudness", passed=False, retry="none", detail="integrated loudness could not be measured")
    else:
        level = Check(
            name="loudness",
            passed=abs(integrated - spec.target_lufs) <= spec.tolerance,
            retry="none",
            score=integrated if math.isfinite(integrated) else -99.0,
            threshold=spec.target_lufs,
            detail=f"{integrated:.1f} LUFS integrated, target {spec.target_lufs:.1f} ±{spec.tolerance:.1f}",
        )
    ceiling = spec.true_peak + TRUE_PEAK_HEADROOM
    peak = scan_result.true_peak
    if peak is None:
        limit = Check(name="true_peak", passed=False, retry="none", detail="true peak could not be measured")
    else:
        limit = Check(
            name="true_peak",
            passed=peak <= ceiling,
            retry="none",
            score=peak if math.isfinite(peak) else -99.0,
            threshold=ceiling,
            detail=f"{peak:.1f} dBTP, ceiling {ceiling:.1f} dBTP",
        )
    return [level, limit]


def speech_onset(silences: list[Interval], duration: float) -> float:
    leading = next((s for s in sorted(silences, key=lambda s: s.start) if s.start <= LEADING_SILENCE_EPSILON), None)
    return min(leading.end, duration) if leading else 0.0


def speech_start_check(scan_result: AudioScan, duration: float) -> Check:
    onset = speech_onset(scan_result.silences, duration)
    if onset >= duration - 1e-3:
        return Check(name="speech_start", passed=False, severity="warn", score=round(onset, 3), threshold=SPEECH_START_LIMIT, detail="no speech detected")
    return Check(
        name="speech_start",
        passed=onset <= SPEECH_START_LIMIT,
        severity="warn",
        score=round(onset, 3),
        threshold=SPEECH_START_LIMIT,
        detail=f"speech starts at {onset:.2f}s (limit {SPEECH_START_LIMIT}s)",
    )


def video_scan_checks(future: Future[VideoScan], inp: GateInput) -> list[Check]:
    try:
        result = future.result()
    except ScanFailed as exc:
        return [Check(name="video_scan", passed=False, retry="render", detail=str(exc))]
    return [black_check(result, inp), *freeze_checks(result, inp)]


def audio_scan_checks(future: Future[AudioScan] | None, facts: RenderFacts, brand: Brand) -> list[Check]:
    if future is None:
        detail = "no audio stream to measure"
        return [
            Check(name="loudness", passed=False, skipped=True, detail=detail),
            Check(name="speech_start", passed=False, skipped=True, severity="warn", detail=detail),
        ]
    try:
        result = future.result()
    except ScanFailed as exc:
        return [Check(name="audio_scan", passed=False, retry="render", detail=str(exc))]
    return [*loudness_checks(result, brand), speech_start_check(result, facts.duration)]


def check_render(inp: GateInput, brand: Brand) -> list[Check]:
    video = Path(inp.video)
    try:
        facts = probe_render(video)
    except RenderUnreadable as exc:
        log.warning("render %s unreadable: %s", video, exc)
        return [Check(name="render_probe", passed=False, retry="render", detail=str(exc))]
    checks = [
        resolution_check(facts, inp),
        frame_rate_check(facts, brand),
        video_codec_check(facts, brand),
        audio_stream_check(facts, brand),
        duration_check(facts, inp),
        file_size_check(facts, brand),
    ]
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="qa-scan") as pool:
        video_future = pool.submit(scan_video, video, facts.duration)
        audio_future = pool.submit(scan_audio, video, facts.duration, brand) if facts.has_audio else None
        checks.extend(video_scan_checks(video_future, inp))
        checks.extend(audio_scan_checks(audio_future, facts, brand))
    return checks
