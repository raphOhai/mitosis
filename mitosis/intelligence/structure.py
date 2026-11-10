from __future__ import annotations

from mitosis.intelligence.text import ends_clause, ends_sentence
from mitosis.models import Word

MIN_HOOK_END = 1.2
MAX_HOOK_END = 4.5
FALLBACK_HOOK_END = 3.0
SNAP_TOLERANCE = 0.35
CLAUSE_BONUS = 0.25
CTA_REGION = 0.6
CTA_FALLBACK_SECONDS = 4.0
WORD_EPSILON = 0.02


def sentences(words: list[Word]) -> list[list[Word]]:
    groups: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        current.append(word)
        if ends_sentence(word.text):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def cut_points(shots: list[tuple[float, float]], duration: float) -> list[float]:
    points = {round(edge, 3) for start, end in shots for edge in (start, end)}
    return sorted(p for p in points if 0.05 < p < duration - 0.05)


def inside_word(time: float, words: list[Word]) -> bool:
    return any(w.start + WORD_EPSILON < time < w.end - WORD_EPSILON for w in words)


def snap(time: float, words: list[Word], cuts: list[float]) -> float:
    nearby = [c for c in cuts if abs(c - time) <= SNAP_TOLERANCE and not inside_word(c, words)]
    return min(nearby, key=lambda c: abs(c - time)) if nearby else time


def boundary_near(target: float, candidates: list[Word]) -> float:
    def distance(word: Word) -> float:
        return abs(word.end - target) - (CLAUSE_BONUS if ends_clause(word.text) else 0.0)

    return min(candidates, key=distance).end


def hook_boundary(groups: list[list[Word]]) -> float:
    for group in groups:
        end = group[-1].end
        if end < MIN_HOOK_END:
            continue
        if end > MAX_HOOK_END:
            inside = [w for w in group if w.end <= MAX_HOOK_END] or group[:1]
            return boundary_near(FALLBACK_HOOK_END, inside)
        return end
    return groups[-1][-1].end


def segment(words: list[Word], shots: list[tuple[float, float]], duration: float) -> tuple[float, float]:
    ordered = sorted(words, key=lambda w: w.start)
    duration = max(duration, ordered[-1].end if ordered else 0.0)
    if duration <= 0:
        return 0.0, 0.0
    cuts = cut_points(shots, duration)
    groups = sentences(ordered)
    raw_end = hook_boundary(groups) if groups else min(FALLBACK_HOOK_END, duration)
    hook_end = round(min(snap(raw_end, ordered, cuts), duration), 3)
    fallback = max(hook_end, duration - CTA_FALLBACK_SECONDS)
    final_start = groups[-1][0].start if len(groups) > 1 else 0.0
    cta_start = max(hook_end, final_start) if final_start >= CTA_REGION * duration else fallback
    return hook_end, round(min(cta_start, duration), 3)
