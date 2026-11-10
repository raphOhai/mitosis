from __future__ import annotations

import importlib.util
import json
import logging
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

from mitosis.config import LoudnessSpec
from mitosis.media import ff, probe
from mitosis.models import AudioPart, Loudness, Stems

log = logging.getLogger(__name__)

MIX_RATE = 48000
SILENCE_FLOOR_LUFS = -70.0
PEAK_FLOOR_DBTP = -99.0
MIN_SPEECH_SECONDS = 0.05
EDGE_FADE_IN = 0.012
EDGE_FADE_OUT = 0.05
BED_FADE_IN = 0.4
BED_FADE_OUT = 1.2
DUCK_DEPTH_DB = 10.0
VOICE_REFERENCE_LUFS = -20.0
MAX_MATCH_GAIN_DB = 30.0
LIMITER_HEADROOM_DB = 1.0
CONDITIONING_ROUNDS = 4
LIMITER_RATE = 192000
DEMUCS_MODEL = "htdemucs"
DEMUCS_TIMEOUT = 1800.0
SILENCE_START = re.compile(r"silence_start:\s*(-?[0-9.]+)")
SILENCE_END = re.compile(r"silence_end:\s*(-?[0-9.]+)")
INTEGRATED = re.compile(r"\bI:\s+(-?[0-9.]+|-?inf)\s+LUFS")


def db_to_linear(db: float) -> float:
    return 10 ** (db / 20)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def fit_length(samples: int) -> str:
    return f"apad=whole_len={samples},atrim=end_sample={samples}"


def extract_audio(video: Path, out: Path, start: float = 0.0, end: float | None = None, rate: int = 48000) -> Path:
    source, target = Path(video), Path(out)
    info = probe.probe(source)
    stop = info.duration if end is None else end
    length = stop - start
    if length <= 0:
        raise ValueError(f"empty audio window [{start:.3f}, {stop:.3f}] in {source}")
    samples = round(length * rate)
    target.parent.mkdir(parents=True, exist_ok=True)
    if info.has_audio:
        inputs = ["-ss", ff.seconds(start), "-t", ff.seconds(length + 0.25), "-i", str(source), "-map", "0:a:0"]
    else:
        log.info("%s has no audio stream, writing %.2fs of silence", source.name, length)
        inputs = ["-f", "lavfi", "-i", f"anullsrc=r={rate}:cl=mono"]
    ff.ffmpeg(
        "-loglevel", "error",
        *inputs,
        "-vn", "-sn", "-dn", "-map_metadata", "-1",
        "-af", f"aresample={rate},aformat=sample_fmts=s16:channel_layouts=mono,{fit_length(samples)}",
        "-c:a", "pcm_s16le",
        str(target),
    )
    return target


def demucs_installed() -> bool:
    try:
        return importlib.util.find_spec("demucs") is not None
    except (ImportError, ValueError):
        return False


def to_wav(source: Path, target: Path, channels: int) -> Path:
    ff.ffmpeg("-loglevel", "error", "-i", str(source), "-ar", str(MIX_RATE), "-ac", str(channels), "-c:a", "pcm_s16le", str(target))
    return target


def separate_stems(audio: Path, out_dir: Path) -> Stems | None:
    if not demucs_installed():
        log.info("demucs is not installed, skipping stem separation")
        return None
    source, folder = Path(audio), Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", DEMUCS_MODEL, "-o", str(folder), str(source)]
    try:
        ff.run(command, timeout=DEMUCS_TIMEOUT)
        produced = folder / DEMUCS_MODEL / source.stem
        stems = Stems(
            vocals=to_wav(produced / "vocals.wav", folder / "vocals.wav", channels=1),
            accompaniment=to_wav(produced / "no_vocals.wav", folder / "accompaniment.wav", channels=2),
        )
    except ff.FFmpegError as exc:
        log.warning("stem separation failed for %s: %s", source.name, exc)
        return None
    finally:
        shutil.rmtree(folder / DEMUCS_MODEL, ignore_errors=True)
    log.info("separated vocals and accompaniment for %s", source.name)
    return stems


def silences(stderr: str, total: float) -> list[tuple[float, float]]:
    found: list[tuple[float, float]] = []
    opened: float | None = None
    for line in stderr.splitlines():
        start = SILENCE_START.search(line)
        if start:
            opened = clamp(float(start.group(1)), 0.0, total)
            continue
        end = SILENCE_END.search(line)
        if end:
            found.append((opened if opened is not None else 0.0, clamp(float(end.group(1)), 0.0, total)))
            opened = None
    if opened is not None:
        found.append((opened, total))
    return found


def speech_intervals(audio: Path, noise_db: float = -35.0, min_silence: float = 0.18) -> list[tuple[float, float]]:
    source = Path(audio)
    total = probe.duration(source)
    completed = ff.ffmpeg(
        "-loglevel", "info",
        "-i", str(source),
        "-vn",
        "-af", f"silencedetect=noise={noise_db:.2f}dB:d={min_silence:.3f}",
        "-f", "null", "-",
    )
    intervals: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in silences(completed.stderr, total):
        if start - cursor >= MIN_SPEECH_SECONDS:
            intervals.append((round(cursor, 3), round(start, 3)))
        cursor = max(cursor, end)
    if total - cursor >= MIN_SPEECH_SECONDS:
        intervals.append((round(cursor, 3), round(total, 3)))
    return intervals


def parse_number(value: object) -> float:
    try:
        return float(str(value).strip())
    except ValueError:
        return float("nan")


def loudnorm_stats(path: Path) -> dict[str, float]:
    completed = ff.ffmpeg("-loglevel", "info", "-i", str(path), "-vn", "-af", "loudnorm=print_format=json", "-f", "null", "-")
    text = completed.stderr
    opening, closing = text.rfind("{"), text.rfind("}")
    if opening < 0 or closing < opening:
        raise ff.FFmpegError(f"loudnorm printed no statistics for {path}")
    try:
        raw = json.loads(text[opening : closing + 1])
    except json.JSONDecodeError as exc:
        raise ff.FFmpegError(f"unreadable loudnorm statistics for {path}") from exc
    return {key: parse_number(value) for key, value in raw.items() if key != "normalization_type"}


def integrated_lufs(path: Path) -> float:
    completed = ff.ffmpeg("-loglevel", "info", "-i", str(path), "-vn", "-af", "ebur128=framelog=quiet", "-f", "null", "-")
    readings = INTEGRATED.findall(completed.stderr)
    if not readings:
        raise ff.FFmpegError(f"ebur128 printed no integrated loudness for {path}")
    return max(finite_or(parse_number(readings[-1]), SILENCE_FLOOR_LUFS), SILENCE_FLOOR_LUFS)


def finite_or(value: float, fallback: float) -> float:
    return value if math.isfinite(value) else fallback


def measure_loudness(path: Path) -> Loudness:
    stats = loudnorm_stats(Path(path))
    return Loudness(
        integrated=round(max(finite_or(stats["input_i"], SILENCE_FLOOR_LUFS), SILENCE_FLOOR_LUFS), 2),
        true_peak=round(max(finite_or(stats["input_tp"], PEAK_FLOOR_DBTP), PEAK_FLOOR_DBTP), 2),
        lra=round(max(finite_or(stats["input_lra"], 0.0), 0.0), 2),
    )


def is_silent(stats: dict[str, float]) -> bool:
    integrated = stats.get("input_i", float("-inf"))
    return not math.isfinite(integrated) or integrated <= SILENCE_FLOOR_LUFS


def matching_gain(part: AudioPart) -> float:
    level = integrated_lufs(part.path)
    if level <= SILENCE_FLOOR_LUFS:
        return part.gain_db
    return clamp(VOICE_REFERENCE_LUFS - level, -MAX_MATCH_GAIN_DB, MAX_MATCH_GAIN_DB) + part.gain_db


def part_chain(index: int, part: AudioPart, gain_db: float) -> str:
    placement = (
        f"adelay=delays={round(part.offset * MIX_RATE)}S:all=1"
        if part.offset >= 0
        else f"atrim=start={-part.offset:.6f},asetpts=PTS-STARTPTS"
    )
    return (
        f"[{index}:a]aresample={MIX_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo,"
        f"volume={gain_db:.3f}dB,{placement}[p{index}]"
    )


def render_voice(parts: list[AudioPart], total: float, samples: int, target: Path) -> Path:
    edges = f"afade=t=in:st=0:d={EDGE_FADE_IN},afade=t=out:st={max(0.0, total - EDGE_FADE_OUT):.6f}:d={EDGE_FADE_OUT}"
    if not parts:
        ff.ffmpeg(
            "-loglevel", "error",
            "-f", "lavfi", "-i", f"anullsrc=r={MIX_RATE}:cl=stereo",
            "-af", fit_length(samples),
            "-c:a", "pcm_f32le", str(target),
        )
        return target
    inputs = [arg for part in parts for arg in ("-i", str(part.path))]
    chains = [part_chain(index, part, matching_gain(part)) for index, part in enumerate(parts)]
    labels = "".join(f"[p{index}]" for index in range(len(parts)))
    chains.append(f"{labels}amix=inputs={len(parts)}:normalize=0:duration=longest,{fit_length(samples)},{edges}[voice]")
    ff.ffmpeg(
        "-loglevel", "error",
        *inputs,
        "-filter_complex", ";".join(chains),
        "-map", "[voice]",
        "-c:a", "pcm_f32le", str(target),
    )
    return target


def render_bed(bed: Path, total: float, samples: int, bed_offset: float, target: Path) -> Path:
    length = probe.duration(bed)
    if length <= 0:
        raise ff.FFmpegError(f"music bed {bed} has no duration")
    seek = bed_offset % length if bed_offset >= 0 else 0.0
    lead = f"adelay=delays={round(-bed_offset * MIX_RATE)}S:all=1," if bed_offset < 0 else ""
    fade_in = min(BED_FADE_IN, total / 4)
    fade_out = min(BED_FADE_OUT, total / 4)
    ff.ffmpeg(
        "-loglevel", "error",
        "-stream_loop", "-1",
        "-ss", ff.seconds(seek),
        "-i", str(bed),
        "-vn",
        "-af",
        (
            f"aresample={MIX_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo,{lead}{fit_length(samples)},"
            f"afade=t=in:st=0:d={fade_in:.3f},afade=t=out:st={max(0.0, total - fade_out):.6f}:d={fade_out:.3f}"
        ),
        "-c:a", "pcm_f32le", str(target),
    )
    return target


def duck_bed(voice: Path, bed: Path, samples: int, spec: LoudnessSpec, target: Path) -> Path:
    voice_lufs, bed_lufs = integrated_lufs(voice), integrated_lufs(bed)
    reference = spec.target_lufs if voice_lufs <= SILENCE_FLOOR_LUFS else voice_lufs
    bed_gain = 0.0 if bed_lufs <= SILENCE_FLOOR_LUFS else clamp(reference + spec.bed_gain_db - bed_lufs, -60.0, 30.0)
    threshold = clamp(db_to_linear(reference - DUCK_DEPTH_DB), 0.001, 1.0)
    graph = ";".join(
        [
            f"[1:a]volume={bed_gain:.3f}dB[bed]",
            "[0:a]asplit=2[voice][key]",
            f"[bed][key]sidechaincompress=threshold={threshold:.5f}:ratio=6:attack=20:release=400:makeup=1[ducked]",
            f"[voice][ducked]amix=inputs=2:normalize=0:duration=first,{fit_length(samples)}[mix]",
        ]
    )
    ff.ffmpeg(
        "-loglevel", "error",
        "-i", str(voice), "-i", str(bed),
        "-filter_complex", graph,
        "-map", "[mix]",
        "-c:a", "pcm_f32le", str(target),
    )
    log.debug("bed gain %.2f dB under voice at %.2f LUFS", bed_gain, reference)
    return target


def peaks_fit(stats: dict[str, float], spec: LoudnessSpec) -> bool:
    return stats["input_tp"] + spec.target_lufs - stats["input_i"] <= spec.true_peak


def condition_peaks(premix: Path, stats: dict[str, float], spec: LoudnessSpec, scratch: Path) -> tuple[Path, dict[str, float]]:
    gain = spec.target_lufs - stats["input_i"]
    ceiling_db = spec.true_peak - LIMITER_HEADROOM_DB
    current, current_stats = premix, stats
    for attempt in range(CONDITIONING_ROUNDS):
        conditioned = scratch / f"conditioned_{attempt}.wav"
        ceiling = clamp(db_to_linear(ceiling_db), 0.0625, 1.0)
        ff.ffmpeg(
            "-loglevel", "error",
            "-i", str(premix),
            "-af",
            (
                f"volume={gain:.3f}dB,aresample={LIMITER_RATE},"
                f"alimiter=limit={ceiling:.5f}:attack=5:release=80:level=0:latency=1,aresample={MIX_RATE}"
            ),
            "-c:a", "pcm_f32le", str(conditioned),
        )
        current, current_stats = conditioned, loudnorm_stats(conditioned)
        if peaks_fit(current_stats, spec):
            break
        overshoot = current_stats["input_tp"] + spec.target_lufs - current_stats["input_i"] - spec.true_peak
        gain += spec.target_lufs - current_stats["input_i"]
        ceiling_db -= max(overshoot, 0.5)
    return current, current_stats


def loudnorm_filter(stats: dict[str, float], spec: LoudnessSpec, lra: float) -> str:
    return (
        f"loudnorm=I={clamp(spec.target_lufs, -70.0, -5.0):.2f}"
        f":TP={clamp(spec.true_peak, -9.0, 0.0):.2f}"
        f":LRA={clamp(lra, 1.0, 50.0):.2f}"
        f":measured_I={clamp(stats['input_i'], -99.0, 0.0):.2f}"
        f":measured_TP={clamp(finite_or(stats['input_tp'], -99.0), -99.0, 99.0):.2f}"
        f":measured_LRA={clamp(finite_or(stats['input_lra'], 0.0), 0.01, 99.0):.2f}"
        f":measured_thresh={clamp(finite_or(stats['input_thresh'], -70.0), -99.0, 0.0):.2f}"
        f":offset={clamp(finite_or(stats['target_offset'], 0.0), -99.0, 99.0):.2f}"
        ":linear=true"
    )


def finalize(source: Path, target: Path, samples: int, normalization: str | None) -> None:
    chain = [normalization] if normalization else []
    chain += [f"aresample={MIX_RATE}", "aformat=sample_fmts=s16:channel_layouts=stereo", fit_length(samples)]
    ff.ffmpeg(
        "-loglevel", "error",
        "-i", str(source),
        "-af", ",".join(chain),
        "-ar", str(MIX_RATE), "-ac", "2",
        "-c:a", "pcm_s16le", str(target),
    )


def normalize(premix: Path, target: Path, samples: int, spec: LoudnessSpec, scratch: Path) -> None:
    stats = loudnorm_stats(premix)
    if is_silent(stats):
        log.warning("mix is silent, writing it without loudness normalization")
        finalize(premix, target, samples, None)
        return
    source = premix
    if not peaks_fit(stats, spec):
        source, stats = condition_peaks(premix, stats, spec, scratch)
    measured_range = finite_or(stats["input_lra"], 0.0)
    allowed_range = spec.lra
    if measured_range > spec.lra:
        log.info("loudness range %.1f LU exceeds the %.1f LU target, kept to stay in linear mode", measured_range, spec.lra)
        allowed_range = measured_range + 0.5
    if not peaks_fit(stats, spec):
        log.info("true peak %.2f dBTP still too hot for linear mode, loudnorm will limit dynamically", stats["input_tp"])
    finalize(source, target, samples, loudnorm_filter(stats, spec, allowed_range))


def build_track(
    parts: list[AudioPart],
    total: float,
    out: Path,
    loudness: LoudnessSpec,
    bed: Path | None = None,
    bed_offset: float = 0.0,
) -> Path:
    if total <= 0:
        raise ValueError("track length must be positive")
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    samples = round(total * MIX_RATE)
    with tempfile.TemporaryDirectory(prefix="mitosis-mix-") as folder:
        scratch = Path(folder)
        premix = render_voice(parts, total, samples, scratch / "voice.wav")
        if bed is not None:
            music = render_bed(Path(bed), total, samples, bed_offset, scratch / "bed.wav")
            premix = duck_bed(premix, music, samples, loudness, scratch / "premix.wav")
        normalize(premix, target, samples, loudness, scratch)
    log.debug("built %.2fs track %s", total, target.name)
    return target
