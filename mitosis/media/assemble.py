from __future__ import annotations

import logging
import os
from pathlib import Path

from mitosis.config import Brand
from mitosis.media import ff, frames, probe, reframe
from mitosis.models import Aspect, MediaInfo, RenderJob, Segment

log = logging.getLogger(__name__)

AUDIO_RATE = 48000
RETIME_RANGE = (0.8, 1.25)
RETIME_EPSILON = 1e-3
RENDER_TIMEOUT = 3600.0


def frame_budget(seconds: list[float], fps: int) -> list[int]:
    budget: list[int] = []
    elapsed = 0.0
    for length in seconds:
        first = round(elapsed * fps)
        elapsed += length
        budget.append(max(1, round(elapsed * fps) - first))
    return budget


def source_window(segment: Segment, info: MediaInfo) -> tuple[float, float]:
    end = info.duration if segment.end is None else min(segment.end, info.duration or segment.end)
    length = end - segment.start
    if length <= 0:
        raise ValueError(f"segment of {segment.path} has no content between {segment.start:.3f}s and {end:.3f}s")
    return segment.start, length


def timing_filters(length: float, seconds: float) -> tuple[list[str], float]:
    speed = length / seconds
    if abs(speed - 1.0) <= RETIME_EPSILON:
        return [], length
    if RETIME_RANGE[0] <= speed <= RETIME_RANGE[1]:
        return [f"setpts=PTS*{seconds / length:.6f}"], seconds
    if speed > RETIME_RANGE[1]:
        return [f"trim=duration={seconds:.6f}"], seconds
    return [], length


def reframe_filter(segment: Segment, info: MediaInfo, aspect: Aspect) -> str:
    if segment.reframe == "shots":
        return reframe.shots_filter(info, segment.shots, aspect, segment.start)
    return reframe.cover_filter(info.width, info.height, aspect)


def segment_chain(index: int, segment: Segment, info: MediaInfo, length: float, aspect: Aspect, fps: int, frames_out: int) -> str:
    timing, produced = timing_filters(length, segment.seconds)
    shortfall = max(0.0, segment.seconds - produced) + 2.0 / fps
    steps = [
        "setpts=PTS-STARTPTS",
        reframe_filter(segment, info, aspect),
        *timing,
        f"fps={fps}",
        f"tpad=stop_mode=clone:stop_duration={shortfall:.6f}",
        f"trim=end_frame={frames_out}",
        "setpts=PTS-STARTPTS",
        "format=yuv420p",
    ]
    return f"[{index}:v]" + ",".join(steps) + f"[v{index}]"


def subtitle_filter(path: Path) -> str:
    return f"ass=filename={ff.escape_filter_value(str(Path(path).resolve()))}"


def build_graph(job: RenderJob, infos: list[MediaInfo], lengths: list[float], fps: int, total: float) -> str:
    budget = frame_budget([segment.seconds for segment in job.segments], fps)
    chains = [
        segment_chain(index, segment, info, length, job.aspect, fps, frames_out)
        for index, (segment, info, length, frames_out) in enumerate(zip(job.segments, infos, lengths, budget))
    ]
    labels = "".join(f"[v{index}]" for index in range(len(job.segments)))
    video_tail = [f"concat=n={len(job.segments)}:v=1:a=0"] if len(job.segments) > 1 else ["null"]
    if job.subtitles is not None:
        video_tail.append(subtitle_filter(job.subtitles))
    chains.append(labels + ",".join(video_tail) + "[vout]")
    samples = round(total * AUDIO_RATE)
    chains.append(
        f"[{len(job.segments)}:a]aresample={AUDIO_RATE},aformat=channel_layouts=stereo,"
        f"apad=whole_len={samples},atrim=end_sample={samples}[aout]"
    )
    return ";".join(chains)


def encoder_args(brand: Brand, total: float) -> list[str]:
    spec = brand.video
    return [
        "-c:v", spec.codec,
        "-preset", spec.preset,
        "-crf", str(spec.crf),
        "-pix_fmt", spec.pix_fmt,
        "-r", str(spec.fps),
        "-g", str(spec.fps * 2),
        "-c:a", "aac",
        "-b:a", spec.audio_bitrate,
        "-ar", str(spec.sample_rate),
        "-ac", "2",
        "-t", f"{total:.6f}",
        "-movflags", "+faststart",
    ]


def render(job: RenderJob, brand: Brand) -> Path:
    if not job.segments:
        raise ValueError("a render needs at least one segment")
    if job.subtitles is not None and not Path(job.subtitles).exists():
        raise FileNotFoundError(f"subtitles not found: {job.subtitles}")
    fps = brand.video.fps
    total = sum(segment.seconds for segment in job.segments)
    infos = [segment.info or probe.probe(segment.path) for segment in job.segments]
    inputs: list[str] = []
    lengths: list[float] = []
    for segment, info in zip(job.segments, infos):
        start, length = source_window(segment, info)
        lengths.append(length)
        inputs += ["-ss", ff.seconds(start), "-t", ff.seconds(length), "-i", str(segment.path)]
    inputs += ["-i", str(job.audio)]
    target = Path(job.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.stem}.partial{target.suffix}")
    try:
        ff.ffmpeg(
            "-loglevel", "error",
            *inputs,
            "-filter_complex", build_graph(job, infos, lengths, fps, total),
            "-map", "[vout]",
            "-map", "[aout]",
            *encoder_args(brand, total),
            str(partial),
            timeout=RENDER_TIMEOUT,
        )
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    log.info("rendered %s (%s, %.2fs)", target.name, job.aspect.value, total)
    return target


def thumbnail(video: Path, out: Path, t: float = 0.6, width: int = 360) -> Path:
    return frames.extract_frame(video, t, out, width=width)
