from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

from mitosis.config import Brand
from mitosis.models import Aspect, Headline, Word

log = logging.getLogger(__name__)

PHRASE_BREAKS = ".!?…,;:—–"
TRAILING_STRIP = ",.;:…—–-"
CLOSING_QUOTES = "\"'”’)"
WORD_JOINER = "\u2060"
FONT_DIRECTORIES = (
    Path("/System/Library/Fonts/Supplemental"),
    Path("/System/Library/Fonts"),
    Path("/Library/Fonts"),
    Path.home() / "Library" / "Fonts",
    Path("/usr/share/fonts/truetype/msttcorefonts"),
    Path("/usr/local/share/fonts"),
)
FONT_SUFFIXES = (".ttf", ".otf", ".ttc")
REFERENCE_SIZE = 200
FALLBACK_ADVANCE = 0.62
MEASURE_MARGIN = 1.06
POP_PEAK = 1.16
LINE_SPACING = 1.2
POP_SETTLE = 1.08
POP_RISE_MS = 70
POP_SETTLE_MS = 170
LINGER_SECONDS = 0.35
HEADLINE_FADE_MS = 180
HEADLINE_PADDING_RATIO = 0.3
MAX_LINES = 2
MIN_FONT_SIZE = 12


@dataclass(frozen=True)
class Layout:
    lines: list[list[str]]
    size: int
    x: int
    y: int


def ass_color(hex_color: str, opacity: float = 1.0) -> str:
    value = hex_color.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(char * 2 for char in value)
    if len(value) not in {6, 8} or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise ValueError(f"invalid color '{hex_color}'")
    red, green, blue = value[0:2], value[2:4], value[4:6]
    if len(value) == 8:
        opacity *= int(value[6:8], 16) / 255
    alpha = round((1.0 - min(max(opacity, 0.0), 1.0)) * 255)
    return f"&H{alpha:02X}{blue}{green}{red}".upper()


def inline_color(hex_color: str) -> str:
    return f"&H{ass_color(hex_color)[4:]}&"


def ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, rest = divmod(centiseconds, 360000)
    minutes, rest = divmod(rest, 6000)
    secs, cs = divmod(rest, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def escape_text(text: str) -> str:
    return " ".join(text.split()).replace("\\", "\\" + WORD_JOINER).replace("{", "(").replace("}", ")")


def display(text: str, uppercase: bool) -> str:
    label = text.strip().rstrip(TRAILING_STRIP) or text.strip()
    return label.upper() if uppercase else label


def ends_phrase(text: str) -> bool:
    core = text.rstrip(CLOSING_QUOTES)
    return bool(core) and core[-1] in PHRASE_BREAKS


@lru_cache(maxsize=16)
def font_file(name: str) -> Path | None:
    variants = dict.fromkeys([name, name.replace(" ", ""), name.replace(" ", "-")])
    for directory in FONT_DIRECTORIES:
        for variant in variants:
            for suffix in FONT_SUFFIXES:
                candidate = directory / f"{variant}{suffix}"
                if candidate.is_file():
                    return candidate
    return None


@lru_cache(maxsize=16)
def reference_font(name: str) -> ImageFont.FreeTypeFont | None:
    path = font_file(name)
    if path is None:
        log.debug("font '%s' not found, using width estimates", name)
        return None
    try:
        return ImageFont.truetype(str(path), REFERENCE_SIZE)
    except OSError:
        log.debug("font file %s unreadable, using width estimates", path)
        return None


def text_width(text: str, font: str, size: float) -> float:
    reference = reference_font(font)
    if reference is None:
        return len(text) * size * FALLBACK_ADVANCE
    ascent, descent = reference.getmetrics()
    return reference.getlength(text) * size / (ascent + descent) * MEASURE_MARGIN


def chunk(words: list[Word], max_words: int, max_chars: int, max_gap: float) -> list[list[Word]]:
    groups: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        label = display(word.text, uppercase=False)
        if not label:
            continue
        if current:
            length = sum(len(display(w.text, uppercase=False)) + 1 for w in current) + len(label)
            if len(current) >= max_words or length > max_chars or word.start - current[-1].end > max_gap:
                groups.append(current)
                current = []
        current.append(word)
        if ends_phrase(word.text):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def split_lines(labels: list[str], font: str, size: float) -> list[list[str]]:
    best = [labels]
    best_width = text_width(" ".join(labels), font, size)
    for index in range(1, len(labels)):
        candidate = [labels[:index], labels[index:]]
        width = max(text_width(" ".join(line), font, size) for line in candidate)
        if width < best_width:
            best, best_width = candidate, width
    return best


def block_size(lines: list[list[str]], font: str, size: float, outline: float, grow: float) -> tuple[float, float]:
    widths = [text_width(" ".join(line), font, size) for line in lines]
    widest_word = max(text_width(label, font, size) for line in lines for label in line)
    width = max(widths) + widest_word * (grow - 1.0) + 2 * outline
    height = len(lines) * size * LINE_SPACING + size * (grow - 1.0) + 2 * outline
    return width, height


def centered(anchor: float, half: float, low: float, high: float) -> int:
    lowest, highest = math.ceil(low + half), math.floor(high - half)
    if lowest > highest:
        return round((low + high) / 2)
    return min(max(round(anchor), lowest), highest)


def fit_block(
    lines: list[list[str]],
    font: str,
    base_size: int,
    border_ratio: float,
    grow: float,
    anchor: tuple[float, float],
    zone: tuple[int, int, int, int],
) -> Layout:
    left, top, right, bottom = zone
    size = base_size
    while True:
        width, height = block_size(lines, font, size, size * border_ratio, grow)
        if (width <= right - left and height <= bottom - top) or size <= MIN_FONT_SIZE:
            break
        scale = min((right - left) / width, (bottom - top) / height)
        size = max(MIN_FONT_SIZE, min(size - 1, math.floor(size * scale)))
    x = centered(anchor[0], width / 2, left, right)
    y = centered(anchor[1], height / 2, top, bottom)
    return Layout(lines=lines, size=size, x=x, y=y)


def caption_layout(labels: list[str], brand: Brand, aspect: Aspect) -> Layout:
    style = brand.subtitle
    width, height = aspect.size
    zone = brand.safe_zone(aspect).rect(aspect)
    base = round(style.size_ratio * width)
    grow = POP_PEAK if style.pop else 1.0
    single_width, _ = block_size([labels], style.font, base, base * style.outline_ratio, grow)
    wrap = len(labels) > 1 and (single_width > zone[2] - zone[0] or len(" ".join(labels)) > style.max_chars)
    lines = split_lines(labels, style.font, base) if wrap else [labels]
    anchor = (width / 2, style.anchor.get(aspect.value, 0.6) * height)
    return fit_block(lines, style.font, base, style.outline_ratio, grow, anchor, zone)


def wrap_words(text: str, limit: int) -> list[list[str]]:
    lines: list[list[str]] = []
    current: list[str] = []
    for token in text.split():
        if current and len(" ".join([*current, token])) > limit:
            lines.append(current)
            current = []
        current.append(token)
    if current:
        lines.append(current)
    return lines


def headline_layout(text: str, brand: Brand, aspect: Aspect) -> Layout:
    style = brand.headline
    width, height = aspect.size
    zone = brand.safe_zone(aspect).rect(aspect)
    lines = wrap_words(text, style.max_chars_per_line)
    anchor = (width / 2, style.anchor.get(aspect.value, 0.15) * height)
    return fit_block(lines, style.font, round(style.size_ratio * width), HEADLINE_PADDING_RATIO, 1.0, anchor, zone)


def placement(layout: Layout, base_size: int) -> str:
    size = f"\\fs{layout.size}" if layout.size != base_size else ""
    return f"\\an5\\pos({layout.x},{layout.y}){size}"


def highlighted_text(lines: list[list[str]], active: int, brand: Brand) -> str:
    style = brand.subtitle
    reset = f"{{\\1c{inline_color(style.primary)}\\fscx100\\fscy100}}"
    if style.pop:
        pop = (
            f"\\t(0,{POP_RISE_MS},\\fscx{POP_PEAK * 100:.0f}\\fscy{POP_PEAK * 100:.0f})"
            f"\\t({POP_RISE_MS},{POP_SETTLE_MS},\\fscx{POP_SETTLE * 100:.0f}\\fscy{POP_SETTLE * 100:.0f})"
        )
    else:
        pop = ""
    rendered: list[str] = []
    position = 0
    for line in lines:
        parts: list[str] = []
        for label in line:
            text = escape_text(label)
            if position == active:
                text = f"{{\\1c{inline_color(style.highlight)}{pop}}}{text}{reset}"
            parts.append(text)
            position += 1
        rendered.append(" ".join(parts))
    return "\\N".join(rendered)


def dialogue(layer: int, start: float, end: float, style: str, text: str) -> str:
    return f"Dialogue: {layer},{ass_time(start)},{ass_time(end)},{style},,0,0,0,,{text}"


def caption_events(words: list[Word], aspect: Aspect, brand: Brand) -> list[str]:
    style = brand.subtitle
    base = round(style.size_ratio * aspect.size[0])
    groups = chunk(words, style.max_words, style.max_chars, style.max_gap)
    events: list[str] = []
    for index, group in enumerate(groups):
        following = groups[index + 1][0].start if index + 1 < len(groups) else math.inf
        shown_until = max(min(group[-1].end + LINGER_SECONDS, following), group[-1].end)
        labels = [display(word.text, style.uppercase) for word in group]
        layout = caption_layout(labels, brand, aspect)
        prefix = f"{{{placement(layout, base)}}}"
        for position, word in enumerate(group):
            start = group[0].start if position == 0 else word.start
            end = group[position + 1].start if position + 1 < len(group) else shown_until
            if round(end * 100) <= round(start * 100):
                continue
            events.append(dialogue(1, start, end, "Caption", prefix + highlighted_text(layout.lines, position, brand)))
    return events


def headline_event(headline: Headline, aspect: Aspect, brand: Brand) -> str | None:
    text = " ".join(headline.text.split())
    if not text or headline.end <= headline.start:
        return None
    layout = headline_layout(text, brand, aspect)
    base = round(brand.headline.size_ratio * aspect.size[0])
    body = "\\N".join(escape_text(" ".join(line)) for line in layout.lines)
    fade = f"\\fad({HEADLINE_FADE_MS},{HEADLINE_FADE_MS})"
    return dialogue(2, headline.start, headline.end, "Headline", f"{{{placement(layout, base)}{fade}}}{body}")


def style_lines(aspect: Aspect, brand: Brand) -> list[str]:
    width = aspect.size[0]
    caption, banner = brand.subtitle, brand.headline
    caption_size = round(caption.size_ratio * width)
    banner_size = round(banner.size_ratio * width)
    outline = max(1.0, caption_size * caption.outline_ratio)
    padding = banner_size * HEADLINE_PADDING_RATIO
    box = ass_color(banner.box, banner.box_opacity)
    return [
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
        (
            f"Style: Caption,{caption.font},{caption_size},{ass_color(caption.primary)},{ass_color(caption.highlight)},"
            f"{ass_color(caption.outline)},{ass_color(caption.outline, 0.6)},-1,0,0,0,100,100,0,0,1,"
            f"{outline:.1f},{caption.shadow:.1f},5,0,0,0,1"
        ),
        (
            f"Style: Headline,{banner.font},{banner_size},{ass_color(banner.color)},{ass_color(banner.color)},"
            f"{box},{box},-1,0,0,0,100,100,0,0,3,{padding:.1f},0,5,0,0,0,1"
        ),
    ]


def write_ass(words: list[Word], aspect: Aspect, brand: Brand, out: Path, headline: Headline | None = None) -> Path:
    width, height = aspect.size
    events = caption_events(words, aspect, brand)
    banner = headline_event(headline, aspect, brand) if headline else None
    if banner:
        events.append(banner)
    document = [
        "[Script Info]",
        f"Title: {brand.name} {aspect.value}",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        *style_lines(aspect, brand),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        *events,
        "",
    ]
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(document), encoding="utf-8")
    log.debug("wrote %d subtitle events to %s", len(events), target.name)
    return target
