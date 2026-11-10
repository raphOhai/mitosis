from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from mitosis.media import ff, probe

log = logging.getLogger(__name__)

FRAME_PREFIX = "frame_"
EDGE_MARGIN = 0.05


def image_codec_args(out: Path) -> list[str]:
    suffix = out.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return ["-q:v", "2"]
    if suffix == ".png":
        return ["-compression_level", "6"]
    if suffix == ".webp":
        return ["-quality", "90"]
    raise ValueError(f"unsupported image format '{out.suffix}' for {out}")


def scale_filter(width: int | None) -> str:
    return "setsar=1" if width is None else f"scale={int(width)}:-2:flags=lanczos,setsar=1"


def grab(video: Path, t: float, out: Path, width: int | None) -> None:
    ff.ffmpeg(
        "-loglevel", "error",
        "-ss", ff.seconds(t),
        "-i", str(video),
        "-frames:v", "1",
        "-vf", scale_filter(width),
        "-update", "1",
        *image_codec_args(out),
        str(out),
    )


def written(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def extract_frame(video: Path, t: float, out: Path, width: int | None = None) -> Path:
    source, target = Path(video), Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    failure: ff.FFmpegError | None = None
    try:
        grab(source, t, target, width)
    except ff.FFmpegError as exc:
        failure = exc
    if written(target):
        return target
    info = probe.probe(source)
    last = max(0.0, info.duration - max(1.0 / info.fps if info.fps else 0.0, EDGE_MARGIN))
    if t <= last and failure is not None:
        raise failure
    log.debug("no frame at %.3fs in %s, retrying at %.3fs", t, source, last)
    grab(source, min(t, last), target, width)
    if not written(target):
        raise ff.FFmpegError(f"could not extract a frame at {t:.3f}s from {source}")
    return target


def sample_frames(
    video: Path,
    out_dir: Path,
    every: float = 0.5,
    width: int = 720,
    start: float = 0.0,
    end: float | None = None,
) -> list[tuple[float, Path]]:
    if every <= 0:
        raise ValueError("sampling interval must be positive")
    source, folder = Path(video), Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for stale in folder.glob(f"{FRAME_PREFIX}*.jpg"):
        stale.unlink()
    stop = probe.duration(source) if end is None else end
    if stop <= start:
        return []
    ff.ffmpeg(
        "-loglevel", "error",
        "-ss", ff.seconds(start),
        "-t", ff.seconds(stop - start),
        "-i", str(source),
        "-vf", f"fps=1/{every:.6f}:round=near:eof_action=pass,scale={int(width)}:-2:flags=lanczos,setsar=1",
        "-fps_mode", "passthrough",
        "-q:v", "3",
        str(folder / f"{FRAME_PREFIX}%05d.jpg"),
    )
    images = sorted(folder.glob(f"{FRAME_PREFIX}*.jpg"))
    samples = [(round(start + index * every, 3), path) for index, path in enumerate(images)]
    return [(t, path) for t, path in samples if t < stop]


def read_gray(image: Path) -> np.ndarray:
    try:
        buffer = np.fromfile(str(image), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"cannot read image {image}") from exc
    gray = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE) if buffer.size else None
    if gray is None:
        raise ValueError(f"cannot decode image {image}")
    return gray


def sharpness(image: Path) -> float:
    return float(cv2.Laplacian(read_gray(Path(image)), cv2.CV_64F).var())
