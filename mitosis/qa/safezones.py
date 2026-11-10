from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from mitosis.config import Brand
from mitosis.models import Check, GateInput

log = logging.getLogger(__name__)

NARROW_CHAR = 0.62
MIXED_CHAR = 0.66
WIDE_CHAR = 0.70
SPACE_SHARE = 0.5
LINE_HEIGHT = 1.2
UPPERCASE_SHARE = 0.6
TOLERANCE_PX = 1.0
MAX_LISTED = 6
SNIPPET_CHARS = 28
HEAVY_MARKERS = ("black", "heavy", "bold", "impact", "extrabold", "ultra", "fat")

STYLE_FIELDS = (
    "name,fontname,fontsize,primarycolour,secondarycolour,outlinecolour,backcolour,bold,italic,underline,strikeout,"
    "scalex,scaley,spacing,angle,borderstyle,outline,shadow,alignment,marginl,marginr,marginv,encoding"
).split(",")
EVENT_FIELDS = "layer,start,end,style,name,marginl,marginr,marginv,effect,text".split(",")
LEGACY_ALIGNMENT = {1: 1, 2: 2, 3: 3, 5: 7, 6: 8, 7: 9, 9: 4, 10: 5, 11: 6}

OVERRIDE_BLOCK = re.compile(r"\{([^}]*)\}")
TAG_PATTERN = re.compile(
    r"\\(?:"
    r"(?P<tag>an|a|fscx|fscy|fs|bord|shad|b|p)(?P<number>-?\d+(?:\.\d+)?)"
    r"|(?P<call>pos|move)\((?P<args>[^)]*)\)"
    r"|fn(?P<font>[^\\]*)"
    r"|r(?P<reset>[^\\]*)"
    r")"
)
SIZING_TAGS = {"fs", "fscx", "fscy", "bord", "shad"}
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
TIMESTAMP = re.compile(r"(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)")


@dataclass(frozen=True)
class Look:
    font: str = "Arial"
    size: float = 20.0
    bold: bool = False
    scale_x: float = 100.0
    scale_y: float = 100.0
    spacing: float = 0.0
    outline: float = 2.0
    shadow: float = 2.0


@dataclass(frozen=True)
class Style:
    name: str = "Default"
    look: Look = field(default_factory=Look)
    alignment: int = 2
    margin_l: float = 10.0
    margin_r: float = 10.0
    margin_v: float = 10.0


@dataclass(frozen=True)
class Event:
    start: float
    end: float
    style: str
    margin_l: float
    margin_r: float
    margin_v: float
    text: str


@dataclass(frozen=True)
class Script:
    width: int
    height: int
    wrap_style: int
    styles: dict[str, Style]
    events: list[Event]


@dataclass(frozen=True)
class Piece:
    text: str
    look: Look


@dataclass(frozen=True)
class Layout:
    pieces: list[Piece]
    drawing: tuple[float, float] | None
    alignment: int
    anchors: list[tuple[float, float]]
    look: Look


@dataclass(frozen=True)
class Placement:
    event: Event
    box: tuple[float, float, float, float]
    text: str


def as_number(value: str, default: float = 0.0) -> float:
    try:
        return float(value.strip())
    except (AttributeError, ValueError):
        return default


def as_flag(value: str) -> bool:
    return as_number(value) != 0


def parse_time(value: str) -> float:
    match = TIMESTAMP.search(value)
    if not match:
        return 0.0
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_style(value: str, fields: list[str], legacy: bool) -> Style:
    record = dict(zip(fields, (part.strip() for part in value.split(","))))
    alignment = int(as_number(record.get("alignment", "2"), 2))
    if legacy:
        alignment = LEGACY_ALIGNMENT.get(alignment, 2)
    look = Look(
        font=record.get("fontname", "Arial"),
        size=as_number(record.get("fontsize", "20"), 20.0),
        bold=as_flag(record.get("bold", "0")),
        scale_x=as_number(record.get("scalex", "100"), 100.0),
        scale_y=as_number(record.get("scaley", "100"), 100.0),
        spacing=as_number(record.get("spacing", "0")),
        outline=as_number(record.get("outline", "2"), 2.0),
        shadow=as_number(record.get("shadow", "2"), 2.0),
    )
    return Style(
        name=record.get("name", "Default"),
        look=look,
        alignment=alignment if 1 <= alignment <= 9 else 2,
        margin_l=as_number(record.get("marginl", "10"), 10.0),
        margin_r=as_number(record.get("marginr", "10"), 10.0),
        margin_v=as_number(record.get("marginv", "10"), 10.0),
    )


def parse_event(value: str, fields: list[str]) -> Event:
    parts = value.split(",", len(fields) - 1)
    record = dict(zip(fields, parts))
    return Event(
        start=parse_time(record.get("start", "")),
        end=parse_time(record.get("end", "")),
        style=record.get("style", "Default").strip().lstrip("*"),
        margin_l=as_number(record.get("marginl", "0")),
        margin_r=as_number(record.get("marginr", "0")),
        margin_v=as_number(record.get("marginv", "0")),
        text=record.get("text", ""),
    )


def parse_ass(text: str, fallback_size: tuple[int, int]) -> Script:
    section = ""
    info: dict[str, str] = {}
    style_fields = list(STYLE_FIELDS)
    event_fields = list(EVENT_FIELDS)
    styles: dict[str, Style] = {}
    events: list[Event] = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("﻿")
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            continue
        key, separator, value = line.partition(":")
        if not separator:
            continue
        key = key.strip().lower()
        if section == "script info":
            info[key] = value.strip()
        elif section in {"v4+ styles", "v4 styles", "v4 styles+"}:
            if key == "format":
                style_fields = [name.strip().lower() for name in value.split(",")]
            elif key == "style":
                style = parse_style(value, style_fields, legacy=section == "v4 styles")
                styles[style.name] = style
        elif section == "events":
            if key == "format":
                event_fields = [name.strip().lower() for name in value.split(",")]
            elif key == "dialogue":
                events.append(parse_event(value.lstrip(), event_fields))
    width = int(as_number(info.get("playresx", "0")))
    height = int(as_number(info.get("playresy", "0")))
    if width <= 0 or height <= 0:
        width, height = fallback_size
    return Script(
        width=width,
        height=height,
        wrap_style=int(as_number(info.get("wrapstyle", "0"))),
        styles=styles,
        events=events,
    )


def split_transforms(block: str) -> tuple[str, list[str]]:
    kept: list[str] = []
    transforms: list[str] = []
    index = 0
    while index < len(block):
        start = block.find("\\t(", index)
        if start < 0:
            kept.append(block[index:])
            break
        kept.append(block[index:start])
        depth = 0
        cursor = start + 2
        while cursor < len(block):
            if block[cursor] == "(":
                depth += 1
            elif block[cursor] == ")":
                depth -= 1
                if depth == 0:
                    break
            cursor += 1
        transforms.append(block[start + 3 : cursor])
        index = cursor + 1
    return "".join(kept), transforms


def numbers(value: str) -> list[float]:
    return [float(item) for item in NUMBER.findall(value)]


class LayoutBuilder:
    def __init__(self, event: Event, style: Style, styles: dict[str, Style], wrap_style: int) -> None:
        self.style = style
        self.styles = styles
        self.wrap_style = wrap_style
        self.look = style.look
        self.peak = style.look
        self.alignment: int | None = None
        self.anchors: list[tuple[float, float]] | None = None
        self.drawing_level = 0
        self.drawing_points: list[float] = []
        self.pieces: list[Piece] = []
        self.event = event

    def build(self) -> Layout:
        cursor = 0
        for match in OVERRIDE_BLOCK.finditer(self.event.text):
            self.add_text(self.event.text[cursor : match.start()])
            self.apply_block(match.group(1))
            cursor = match.end()
        self.add_text(self.event.text[cursor:])
        drawing = None
        if len(self.drawing_points) >= 4:
            xs, ys = self.drawing_points[0::2], self.drawing_points[1::2]
            drawing = (max(xs) - min(xs), max(ys) - min(ys))
        return Layout(
            pieces=self.pieces,
            drawing=drawing,
            alignment=self.alignment or self.style.alignment,
            anchors=self.anchors or [],
            look=self.peak,
        )

    def add_text(self, chunk: str) -> None:
        if not chunk:
            return
        if self.drawing_level > 0:
            scale = 2 ** (self.drawing_level - 1)
            values = numbers(chunk)
            for index in range(0, len(values) - 1, 2):
                self.drawing_points.append(values[index] / scale * self.look.scale_x / 100.0)
                self.drawing_points.append(values[index + 1] / scale * self.look.scale_y / 100.0)
            return
        soft_break = "\n" if self.wrap_style == 2 else " "
        plain = chunk.replace("\\N", "\n").replace("\\n", soft_break).replace("\\h", " ")
        self.pieces.append(Piece(text=plain, look=self.look))

    def apply_block(self, block: str) -> None:
        immediate, transforms = split_transforms(block)
        for match in TAG_PATTERN.finditer(immediate):
            self.apply_tag(match)
        for transform in transforms:
            for match in TAG_PATTERN.finditer(transform):
                if match["tag"] in SIZING_TAGS:
                    self.grow(match["tag"], float(match["number"]))
        self.peak = self.widest(self.peak, self.look)

    def apply_tag(self, match: re.Match[str]) -> None:
        if match["call"]:
            self.position(match["call"], numbers(match["args"]))
        elif match["font"] is not None:
            self.look = replace(self.look, font=match["font"].strip() or self.style.look.font)
        elif match["reset"] is not None:
            target = match["reset"].strip()
            self.look = self.styles.get(target, self.style).look if target else self.style.look
        else:
            self.apply_number(match["tag"], float(match["number"]))

    def position(self, call: str, values: list[float]) -> None:
        if self.anchors is not None:
            return
        if call == "pos" and len(values) >= 2:
            self.anchors = [(values[0], values[1])]
        elif call == "move" and len(values) >= 4:
            self.anchors = [(values[0], values[1]), (values[2], values[3])]

    def apply_number(self, tag: str, amount: float) -> None:
        if tag == "an":
            if self.alignment is None and 1 <= int(amount) <= 9:
                self.alignment = int(amount)
        elif tag == "a":
            if self.alignment is None:
                self.alignment = LEGACY_ALIGNMENT.get(int(amount))
        elif tag == "p":
            self.drawing_level = max(0, int(amount))
        elif tag == "b":
            self.look = replace(self.look, bold=amount == 1 or amount >= 600)
        else:
            self.look = self.adjusted(self.look, tag, amount)

    def grow(self, tag: str, amount: float) -> None:
        adjusted = self.adjusted(self.look, tag, amount)
        self.peak = self.widest(self.peak, adjusted)
        self.look = self.widest(self.look, adjusted)

    @staticmethod
    def adjusted(look: Look, tag: str, amount: float) -> Look:
        if amount < 0:
            return look
        if tag == "fs" and amount > 0:
            return replace(look, size=amount)
        if tag == "fscx":
            return replace(look, scale_x=amount)
        if tag == "fscy":
            return replace(look, scale_y=amount)
        if tag == "bord":
            return replace(look, outline=amount)
        if tag == "shad":
            return replace(look, shadow=amount)
        return look

    @staticmethod
    def widest(first: Look, second: Look) -> Look:
        return replace(
            second,
            size=max(first.size, second.size),
            scale_x=max(first.scale_x, second.scale_x),
            scale_y=max(first.scale_y, second.scale_y),
            outline=max(first.outline, second.outline),
            shadow=max(first.shadow, second.shadow),
        )


def is_heavy(look: Look) -> bool:
    name = look.font.lower()
    return look.bold or any(marker in name for marker in HEAVY_MARKERS)


def is_uppercase(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    return bool(letters) and sum(char.isupper() for char in letters) / len(letters) >= UPPERCASE_SHARE


def char_factor(look: Look, uppercase: bool) -> float:
    heavy = is_heavy(look)
    if heavy and uppercase:
        return WIDE_CHAR
    if heavy or uppercase:
        return MIXED_CHAR
    return NARROW_CHAR


def text_width(text: str, look: Look, uppercase: bool) -> float:
    advance = char_factor(look, uppercase) * look.size * look.scale_x / 100.0
    total = 0.0
    for char in text:
        total += advance * (SPACE_SHARE if char.isspace() else 1.0) + look.spacing
    return total


def split_lines(pieces: list[Piece]) -> list[list[Piece]]:
    lines: list[list[Piece]] = [[]]
    for piece in pieces:
        parts = piece.text.split("\n")
        for index, part in enumerate(parts):
            if index:
                lines.append([])
            if part:
                lines[-1].append(Piece(text=part, look=piece.look))
    return lines


def wrap_line(line: list[Piece], limit: float, uppercase: bool) -> list[list[Piece]]:
    words: list[Piece] = []
    for piece in line:
        for token in re.split(r"(\s+)", piece.text):
            if token:
                words.append(Piece(text=token, look=piece.look))
    rows: list[list[Piece]] = [[]]
    width = 0.0
    for word in words:
        span = text_width(word.text, word.look, uppercase)
        if word.text.isspace():
            if rows[-1]:
                rows[-1].append(word)
                width += span
            continue
        if rows[-1] and width + span > limit:
            while rows[-1] and rows[-1][-1].text.isspace():
                rows[-1].pop()
            rows.append([])
            width = 0.0
        rows[-1].append(word)
        width += span
    return [row for row in rows if row] or [[]]


def measure_text(layout: Layout, available_width: float, wrap: bool) -> tuple[float, float]:
    uppercase = is_uppercase("".join(piece.text for piece in layout.pieces))
    rows: list[list[Piece]] = []
    for line in split_lines(layout.pieces):
        too_wide = sum(text_width(p.text, p.look, uppercase) for p in line) > available_width
        rows.extend(wrap_line(line, available_width, uppercase) if wrap and too_wide else [line])
    default_height = layout.look.size * layout.look.scale_y / 100.0
    width = 0.0
    height = 0.0
    for row in rows:
        width = max(width, sum(text_width(p.text, p.look, uppercase) for p in row))
        height += max((p.look.size * p.look.scale_y / 100.0 for p in row), default=default_height) * LINE_HEIGHT
    return width, height


def measure(layout: Layout, available_width: float, wrap: bool) -> tuple[float, float]:
    has_text = bool("".join(piece.text for piece in layout.pieces).strip())
    if layout.drawing is None:
        return measure_text(layout, available_width, wrap)
    if not has_text:
        return layout.drawing
    width, height = measure_text(layout, available_width, wrap)
    return max(width, layout.drawing[0]), height + layout.drawing[1]


def anchor_points(layout: Layout, style: Style, event: Event, script: Script) -> list[tuple[float, float]]:
    if layout.anchors:
        return layout.anchors
    margin_l = event.margin_l or style.margin_l
    margin_r = event.margin_r or style.margin_r
    margin_v = event.margin_v or style.margin_v
    column = (layout.alignment - 1) % 3
    row = (layout.alignment - 1) // 3
    x = (margin_l, margin_l + (script.width - margin_l - margin_r) / 2, script.width - margin_r)[column]
    y = (script.height - margin_v, script.height / 2, margin_v)[row]
    return [(x, y)]


def box_at(anchor: tuple[float, float], size: tuple[float, float], alignment: int, look: Look) -> tuple[float, float, float, float]:
    width, height = size
    column = (alignment - 1) % 3
    row = (alignment - 1) // 3
    left = anchor[0] - (0.0, width / 2, width)[column]
    top = anchor[1] - (height, height / 2, 0.0)[row]
    pad = look.outline
    return (left - pad, top - pad, left + width + pad + look.shadow, top + height + pad + look.shadow)


def place_event(event: Event, script: Script) -> list[Placement]:
    style = script.styles.get(event.style) or script.styles.get("Default") or next(iter(script.styles.values()), Style())
    layout = LayoutBuilder(event, style, script.styles, script.wrap_style).build()
    plain = "".join(piece.text for piece in layout.pieces).strip()
    if not plain and layout.drawing is None:
        return []
    margin_l = event.margin_l or style.margin_l
    margin_r = event.margin_r or style.margin_r
    size = measure(layout, max(1.0, script.width - margin_l - margin_r), wrap=script.wrap_style != 2)
    label = " ".join(plain.split())[:SNIPPET_CHARS] or "[drawing]"
    return [
        Placement(event=event, box=box_at(anchor, size, layout.alignment, layout.look), text=label)
        for anchor in anchor_points(layout, style, event, script)
    ]


def to_video(box: tuple[float, float, float, float], script: Script, video_size: tuple[int, int]) -> tuple[float, float, float, float]:
    scale_x = video_size[0] / script.width
    scale_y = video_size[1] / script.height
    return (box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y)


def overflow(box: tuple[float, float, float, float], rect: tuple[int, int, int, int]) -> list[str]:
    left, top, right, bottom = rect
    crossings = {
        "left": left - box[0],
        "top": top - box[1],
        "right": box[2] - right,
        "bottom": box[3] - bottom,
    }
    return [f"{side} by {amount:.0f}px" for side, amount in crossings.items() if amount > TOLERANCE_PX]


def describe(placement: Placement, box: tuple[float, float, float, float], crossings: list[str]) -> str:
    event = placement.event
    region = f"[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]"
    return f"{event.start:.2f}-{event.end:.2f}s '{placement.text}' {region} crosses {', '.join(crossings)}"


def check_text_placement(inp: GateInput, brand: Brand) -> Check:
    name = "safe_zones"
    if inp.subtitles is None:
        return Check(name=name, passed=False, skipped=True, detail="no subtitle file to check")
    path = Path(inp.subtitles)
    if not path.is_file():
        return Check(name=name, passed=False, skipped=True, detail=f"subtitle file {path.name} not found")
    try:
        source = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        log.warning("cannot read subtitles %s: %s", path, exc)
        return Check(name=name, passed=False, skipped=True, detail=f"cannot read {path.name}: {exc}")
    video_size = inp.aspect.size
    script = parse_ass(source, video_size)
    rect = brand.safe_zone(inp.aspect).rect(inp.aspect)
    placements = [placement for event in script.events if event.end > event.start for placement in place_event(event, script)]
    violations: list[str] = []
    for placement in placements:
        box = to_video(placement.box, script, video_size)
        crossings = overflow(box, rect)
        if crossings:
            violations.append(describe(placement, box, crossings))
    zone = f"x {rect[0]}-{rect[2]}, y {rect[1]}-{rect[3]} ({inp.aspect.value})"
    total = len(placements)
    if not violations:
        return Check(name=name, passed=True, score=1.0, threshold=1.0, detail=f"{total} text placements inside safe zone {zone}")
    listed = "; ".join(violations[:MAX_LISTED])
    extra = f"; +{len(violations) - MAX_LISTED} more" if len(violations) > MAX_LISTED else ""
    return Check(
        name=name,
        passed=False,
        score=round(1 - len(violations) / max(1, total), 3),
        threshold=1.0,
        detail=f"{len(violations)}/{total} text placements outside safe zone {zone}: {listed}{extra}",
    )
