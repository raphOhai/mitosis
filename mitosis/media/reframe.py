from __future__ import annotations

import math
from collections.abc import Callable

from mitosis.models import Aspect, Box, MediaInfo, Shot

RATIO_TOLERANCE = 0.01
FACE_LINE = 1 / 3
FACE_HEADROOM = 0.25
COVER_FOCUS_Y = 0.4

Origin = tuple[int, int]


def even_floor(value: float) -> int:
    return max(2, int(math.floor(value / 2)) * 2)


def even_ceil(value: float) -> int:
    return max(2, int(math.ceil(value / 2)) * 2)


def even_round(value: float) -> int:
    return int(round(value / 2)) * 2


def scale_to(aspect: Aspect) -> str:
    width, height = aspect.size
    return f"scale={width}:{height}:flags=lanczos,setsar=1"


def crop_size(width: int, height: int, ratio: float) -> tuple[int, int]:
    if width / height > ratio:
        return min(even_floor(height * ratio), even_floor(width)), even_floor(height)
    return even_floor(width), min(even_floor(width / ratio), even_floor(height))


def window_origin(face: Box | None, width: int, height: int, crop_width: int, crop_height: int) -> Origin:
    if face is None:
        x = (width - crop_width) / 2
        y = (height - crop_height) / 2
    else:
        x = face.cx * width - crop_width / 2
        y = min(face.cy * height - crop_height * FACE_LINE, (face.y - face.h * FACE_HEADROOM) * height)
    x = min(max(x, 0.0), width - crop_width)
    y = min(max(y, 0.0), height - crop_height)
    return min(even_round(x), width - crop_width), min(even_round(y), height - crop_height)


def timeline(shots: list[Shot], source_start: float, origin_of: Callable[[Box | None], Origin]) -> list[tuple[float, Origin]]:
    entries: list[tuple[float, Origin]] = []
    for shot in sorted(shots, key=lambda item: item.start):
        begin = max(0.0, shot.start - source_start)
        origin = origin_of(shot.face)
        if entries and begin <= entries[-1][0] + 1e-6:
            entries[-1] = (entries[-1][0], origin)
        elif not entries or origin != entries[-1][1]:
            entries.append((begin, origin))
    if not entries:
        entries.append((0.0, origin_of(None)))
    entries[0] = (0.0, entries[0][1])
    return [entry for index, entry in enumerate(entries) if index == 0 or entry[1] != entries[index - 1][1]]


def piecewise(entries: list[tuple[float, Origin]], axis: int) -> str:
    expression = str(entries[-1][1][axis])
    for index in range(len(entries) - 2, -1, -1):
        current, following = entries[index][1][axis], entries[index + 1][0]
        expression = f"if(lt(t,{following:.4f}),{current},{expression})"
    return expression


def shots_filter(info: MediaInfo, shots: list[Shot], aspect: Aspect, source_start: float) -> str:
    if info.width <= 0 or info.height <= 0:
        raise ValueError(f"cannot reframe {info.path}: unknown frame size")
    if abs(info.ratio - aspect.ratio) < RATIO_TOLERANCE:
        return scale_to(aspect)
    crop_width, crop_height = crop_size(info.width, info.height, aspect.ratio)
    entries = timeline(
        shots,
        source_start,
        lambda face: window_origin(face, info.width, info.height, crop_width, crop_height),
    )
    x, y = piecewise(entries, 0), piecewise(entries, 1)
    return f"crop=w={crop_width}:h={crop_height}:x='{x}':y='{y}',{scale_to(aspect)}"


def cover_filter(width: int, height: int, aspect: Aspect) -> str:
    if width <= 0 or height <= 0:
        raise ValueError("cannot cover-fit a frame without a size")
    target_width, target_height = aspect.size
    factor = max(target_width / width, target_height / height)
    scaled_width = max(target_width, even_ceil(width * factor))
    scaled_height = max(target_height, even_ceil(height * factor))
    x = even_round((scaled_width - target_width) / 2)
    y = even_round((scaled_height - target_height) * COVER_FOCUS_Y)
    return (
        f"scale={scaled_width}:{scaled_height}:flags=lanczos,"
        f"crop={target_width}:{target_height}:{x}:{y},setsar=1"
    )
