from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from pathlib import Path

from mitosis.media import ff, probe
from mitosis.media.audio import speech_intervals
from mitosis.models import Word

log = logging.getLogger(__name__)

SENTENCE_MARKS = ".!?…"
CLAUSE_MARKS = ",;:—–-"
ATTACHABLE = set(SENTENCE_MARKS + CLAUSE_MARKS)
SPOKEN_SYMBOLS = {"&": "and", "+": "plus", "%": "percent"}
NON_WORD = re.compile(r"[\W_]+", re.UNICODE)
SECONDS_PER_WEIGHT = 0.06
MIN_WORD_SECONDS = 0.08
SNAP_MIN_SECONDS = 0.6
SNAP_RATIO = 0.18
SENTENCE_FACTOR = 1.0
CLAUSE_FACTOR = 1.6

Span = tuple[float, float]


def tokenize(script: str) -> list[str]:
    tokens: list[str] = []
    for raw in script.split():
        if any(char.isalnum() for char in raw) or raw in SPOKEN_SYMBOLS:
            tokens.append(raw)
        elif tokens and set(raw) <= ATTACHABLE:
            tokens[-1] += raw
    return tokens


def normalize(token: str) -> str:
    return SPOKEN_SYMBOLS.get(token, NON_WORD.sub("", token.lower()))


def weight(token: str) -> int:
    return max(len(normalize(token)), 1) + 1


def make_word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=round(start, 3), end=round(max(end, start), 3))


def spread(tokens: list[str], start: float, end: float) -> list[Word]:
    weights = [weight(token) for token in tokens]
    total = sum(weights)
    span = max(end - start, 0.0)
    words: list[Word] = []
    consumed = 0
    for token, share in zip(tokens, weights):
        first = start + span * consumed / total
        consumed += share
        words.append(make_word(token, first, start + span * consumed / total))
    return words


def proportional_words(text: str, start: float, end: float) -> list[Word]:
    tokens = tokenize(text)
    return spread(tokens, start, end) if tokens else []


def window(words: list[Word], start: float, end: float, offset: float = 0.0, scale: float = 1.0) -> list[Word]:
    return [
        make_word(word.text, (max(word.start, start) + offset) * scale, (min(word.end, end) + offset) * scale)
        for word in words
        if start <= word.start < end
    ]


def align_script(script: str, audio: Path, transcript: list[Word] | None = None) -> list[Word]:
    tokens = tokenize(script)
    if not tokens:
        return []
    heard = [word for word in transcript or [] if normalize(word.text)]
    if heard:
        aligned = align_to_transcript(tokens, heard, Path(audio))
        if aligned:
            return aligned
        log.info("transcript shares no words with the script, aligning to speech activity instead")
    return align_to_speech(tokens, Path(audio))


def align_to_transcript(tokens: list[str], heard: list[Word], audio: Path) -> list[Word]:
    timings: list[Span | None] = [None] * len(tokens)
    matcher = SequenceMatcher(None, [normalize(t) for t in tokens], [normalize(w.text) for w in heard], autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for step in range(i2 - i1):
                timings[i1 + step] = (heard[j1 + step].start, heard[j1 + step].end)
        elif tag == "replace":
            for step, word in enumerate(spread(tokens[i1:i2], heard[j1].start, heard[j2 - 1].end)):
                timings[i1 + step] = (word.start, word.end)
    if all(timing is None for timing in timings):
        return []
    fill_gaps(tokens, timings, audio)
    return [make_word(token, *timing) for token, timing in zip(tokens, timings) if timing is not None]


def audio_length(audio: Path) -> float | None:
    try:
        return probe.duration(audio)
    except ff.FFmpegError:
        return None


def fill_gaps(tokens: list[str], timings: list[Span | None], audio: Path) -> None:
    index = 0
    while index < len(tokens):
        if timings[index] is not None:
            index += 1
            continue
        stop = index
        while stop < len(tokens) and timings[stop] is None:
            stop += 1
        interpolate_run(tokens, timings, index, stop, audio)
        index = stop


def interpolate_run(tokens: list[str], timings: list[Span | None], first: int, stop: int, audio: Path) -> None:
    estimate = sum(weight(token) for token in tokens[first:stop]) * SECONDS_PER_WEIGHT
    left = timings[first - 1][1] if first > 0 else None
    right = timings[stop][0] if stop < len(tokens) else None
    if left is None:
        left = max(0.0, right - estimate)
    if right is None:
        length = audio_length(audio)
        right = left + estimate if length is None else max(left, min(length, left + estimate))
    needed = MIN_WORD_SECONDS * (stop - first)
    if right - left >= needed:
        assign(timings, first, spread(tokens[first:stop], left, right))
    elif first > 0:
        anchor = timings[first - 1][0]
        assign(timings, first - 1, spread(tokens[first - 1 : stop], anchor, max(right, anchor + needed)))
    else:
        anchor = timings[stop][1] if stop < len(tokens) else right + needed
        assign(timings, first, spread(tokens[first : stop + 1], left, max(anchor, left + needed)))


def assign(timings: list[Span | None], first: int, words: list[Word]) -> None:
    for step, word in enumerate(words):
        if first + step < len(timings):
            timings[first + step] = (word.start, word.end)


def align_to_speech(tokens: list[str], audio: Path) -> list[Word]:
    intervals = speech_intervals(audio)
    if not intervals:
        length = audio_length(audio) or sum(weight(t) for t in tokens) * SECONDS_PER_WEIGHT
        log.info("no speech detected in %s, spreading words over %.2fs", audio.name, length)
        return spread(tokens, 0.0, length)
    words: list[Word] = []
    for token_range, interval_range in snap_groups(tokens, intervals):
        words.extend(place(tokens[slice(*token_range)], intervals[slice(*interval_range)]))
    return words


def boundary_factor(token: str) -> float | None:
    if token[-1] in SENTENCE_MARKS:
        return SENTENCE_FACTOR
    if token[-1] in CLAUSE_MARKS:
        return CLAUSE_FACTOR
    return None


def snap_groups(tokens: list[str], intervals: list[Span]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    lengths = [end - start for start, end in intervals]
    speech = sum(lengths)
    weights = [weight(token) for token in tokens]
    total_weight = sum(weights)
    pauses = [sum(lengths[: k + 1]) for k in range(len(intervals) - 1)]
    boundaries: list[tuple[int, float, float]] = []
    consumed = 0
    for index, token in enumerate(tokens[:-1]):
        consumed += weights[index]
        factor = boundary_factor(token)
        if factor is not None:
            boundaries.append((index, speech * consumed / total_weight, factor))
    tolerance = max(SNAP_MIN_SECONDS, SNAP_RATIO * speech)
    anchors = match_pauses(pauses, boundaries, tolerance)
    groups: list[tuple[tuple[int, int], tuple[int, int]]] = []
    token_cursor, interval_cursor = 0, 0
    for token_index, pause_index in anchors:
        groups.append(((token_cursor, token_index + 1), (interval_cursor, pause_index + 1)))
        token_cursor, interval_cursor = token_index + 1, pause_index + 1
    groups.append(((token_cursor, len(tokens)), (interval_cursor, len(intervals))))
    return groups


def match_pauses(pauses: list[float], boundaries: list[tuple[int, float, float]], tolerance: float) -> list[tuple[int, int]]:
    rows, cols = len(pauses), len(boundaries)
    cost = [[0.0] * (cols + 1) for _ in range(rows + 1)]
    move = [[""] * (cols + 1) for _ in range(rows + 1)]
    for k in range(rows + 1):
        for b in range(cols + 1):
            if k == 0 and b == 0:
                continue
            options: list[tuple[float, str]] = []
            if k > 0:
                options.append((cost[k - 1][b] + tolerance, "pause"))
            if b > 0:
                options.append((cost[k][b - 1], "boundary"))
            if k > 0 and b > 0:
                _, position, factor = boundaries[b - 1]
                distance = abs(pauses[k - 1] - position) * factor
                if distance <= tolerance:
                    options.append((cost[k - 1][b - 1] + distance, "match"))
            cost[k][b], move[k][b] = min(options)
    anchors: list[tuple[int, int]] = []
    k, b = rows, cols
    while k > 0 or b > 0:
        step = move[k][b]
        if step == "match":
            anchors.append((boundaries[b - 1][0], k - 1))
            k, b = k - 1, b - 1
        elif step == "pause":
            k -= 1
        else:
            b -= 1
    return sorted(anchors)


def place(tokens: list[str], intervals: list[Span]) -> list[Word]:
    offsets: list[tuple[float, float, float]] = []
    elapsed = 0.0
    for start, end in intervals:
        offsets.append((elapsed, elapsed + end - start, start))
        elapsed += end - start
    words: list[Word] = []
    for word in spread(tokens, 0.0, elapsed):
        low, high, origin = max(offsets, key=lambda item: min(word.end, item[1]) - max(word.start, item[0]))
        words.append(make_word(word.text, origin + max(word.start, low) - low, origin + min(word.end, high) - low))
    return words
