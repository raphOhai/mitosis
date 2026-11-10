from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def read_image(path: Path) -> np.ndarray | None:
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        log.warning("cannot read image %s: %s", path, exc)
        return None
    if buffer.size == 0:
        log.warning("image %s is empty", path)
        return None
    pixels = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if pixels is None:
        log.warning("cannot decode image %s", path)
    return pixels


def write_jpeg(pixels: np.ndarray, path: Path, quality: int = 95) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError(f"cannot encode JPEG for {path}")
    encoded.tofile(str(path))
    return path


def fit_within(pixels: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = pixels.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return pixels, 1.0
    scale = max_side / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(pixels, size, interpolation=cv2.INTER_AREA), scale


def video_duration(video: Path) -> float | None:
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            return None
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if frames > 0 and fps > 0:
            return float(frames / fps)
        return None
    finally:
        capture.release()


def grab_frames(video: Path, times: list[float]) -> list[tuple[float, np.ndarray | None]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        log.warning("cannot open video %s", video)
        capture.release()
        return [(moment, None) for moment in times]
    try:
        grabbed: list[tuple[float, np.ndarray | None]] = []
        for moment in times:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, moment) * 1000.0)
            ok, frame = capture.read()
            grabbed.append((moment, frame if ok else None))
        return grabbed
    finally:
        capture.release()


def spread(start: float, end: float, count: int) -> list[float]:
    if count <= 0:
        return []
    span = max(0.0, end - start)
    return [start + span * (index + 0.5) / count for index in range(count)]
