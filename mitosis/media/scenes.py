from __future__ import annotations

import logging
import re
from pathlib import Path

from mitosis.media import ff
from mitosis.models import MediaInfo

log = logging.getLogger(__name__)

ANALYSIS_WIDTH = 320
PTS_PATTERN = re.compile(r"pts_time:\s*(-?[0-9.]+)")
SCORE_PATTERN = re.compile(r"lavfi\.scene_score=\s*([0-9.]+)")


def scene_cuts(video: Path, threshold: float) -> list[float]:
    graph = f"scale={ANALYSIS_WIDTH}:-2,select='gt(scene,{threshold:.4f})',metadata=print"
    completed = ff.ffmpeg("-loglevel", "info", "-i", str(video), "-an", "-sn", "-dn", "-vf", graph, "-f", "null", "-")
    cuts: list[float] = []
    pending: float | None = None
    for line in completed.stderr.splitlines():
        stamp = PTS_PATTERN.search(line)
        if stamp:
            pending = float(stamp.group(1))
            continue
        if pending is not None and SCORE_PATTERN.search(line):
            cuts.append(pending)
            pending = None
    return sorted(set(cuts))


def merge_short(cuts: list[float], total: float, min_length: float) -> list[float]:
    kept: list[float] = []
    previous = 0.0
    for cut in cuts:
        if cut <= 0.0 or cut >= total:
            continue
        if cut - previous >= min_length:
            kept.append(cut)
            previous = cut
    while kept and total - kept[-1] < min_length:
        kept.pop()
    return kept


def detect_shots(video: Path, info: MediaInfo, threshold: float = 0.32, min_length: float = 0.6) -> list[tuple[float, float]]:
    total = max(0.0, info.duration)
    if total <= min_length or info.width == 0:
        return [(0.0, round(total, 3))]
    try:
        cuts = scene_cuts(Path(video), threshold)
    except ff.FFmpegError as exc:
        log.warning("scene detection failed for %s, using a single shot: %s", video, exc)
        cuts = []
    bounds = [0.0, *merge_short(cuts, total, min_length), total]
    spans = [(round(start, 3), round(end, 3)) for start, end in zip(bounds, bounds[1:])]
    log.debug("detected %d shots in %s", len(spans), video)
    return spans
