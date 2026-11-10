from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from mitosis.models import Prelaunch, Priors, VariantSpec, Word

log = logging.getLogger(__name__)

OPENING_SECONDS = 1.5
ANALYSIS_WIDTH = 160
BIAS = -0.8
SCALE = 0.5
REASON_THRESHOLD = 0.04
WEIGHTS: dict[str, float] = {
    "time_to_first_word": 0.9,
    "hook_words": 0.5,
    "hook_wps": 0.5,
    "on_screen_words": 0.35,
    "hook_seconds": 0.4,
    "motion_energy": 0.6,
    "brightness": 0.35,
    "contrast": 0.35,
    "hook_score": 0.5,
    "prior_archetype": 0.6,
    "prior_voice": 0.4,
    "prior_location": 0.4,
}


@dataclass(frozen=True)
class Opening:
    readable: bool
    motion: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0


@dataclass(frozen=True)
class Term:
    name: str
    value: float
    contribution: float
    reason: str


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def band(value: float, low: float, high: float, falloff: float) -> float:
    if low <= value <= high:
        return 1.0
    distance = low - value if value < low else value - high
    return clamp(1.0 - 2.0 * distance / falloff)


def read_opening(video: Path) -> Opening:
    capture = cv2.VideoCapture(str(video))
    frames: list[np.ndarray] = []
    try:
        if not capture.isOpened():
            return Opening(readable=False)
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and math.isfinite(fps) and fps > 0 else 30.0
        limit = max(2, round(fps * OPENING_SECONDS))
        while len(frames) < limit:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height = max(1, round(gray.shape[0] * ANALYSIS_WIDTH / max(gray.shape[1], 1)))
            frames.append(cv2.resize(gray, (ANALYSIS_WIDTH, height), interpolation=cv2.INTER_AREA).astype(np.float32))
    except cv2.error as exc:
        log.warning("could not decode %s: %s", video, exc)
    finally:
        capture.release()
    if not frames:
        return Opening(readable=False)
    diffs = [float(np.mean(np.abs(b - a))) / 255.0 for a, b in pairwise(frames)]
    first = frames[0]
    return Opening(
        readable=True,
        motion=float(np.mean(diffs)) if diffs else 0.0,
        brightness=float(first.mean()) / 255.0,
        contrast=float(first.std()) / 255.0,
    )


def term(name: str, value: float, goodness: float, good: str, bad: str) -> Term:
    contribution = SCALE * WEIGHTS[name] * clamp(goodness)
    return Term(name=name, value=value, contribution=contribution, reason=good if contribution >= 0 else bad)


def speech_terms(words: list[Word], hook_seconds: float, on_screen: str) -> list[Term]:
    ordered = sorted(words, key=lambda w: w.start)
    first = ordered[0].start if ordered else hook_seconds
    hook_words = [w for w in ordered if w.start < hook_seconds]
    count = len(hook_words)
    speaking = max(hook_seconds - first, 0.3)
    wps = count / speaking
    screen_words = len(on_screen.split())
    return [
        term(
            "time_to_first_word",
            first,
            (0.6 - first) / 0.6,
            f"speech starts at {first:.2f}s, so the hook is heard immediately",
            f"the first word lands at {first:.2f}s; dead air at the start costs thumb-stops",
        ),
        term(
            "hook_words",
            float(count),
            1.0 - abs(count - 8) / 5,
            f"{count} words in the hook is a tight, scannable line",
            f"{count} words in the hook window is {'too few to land an idea' if count < 8 else 'a lot to absorb in under 3 seconds'}",
        ),
        term(
            "hook_wps",
            wps,
            band(wps, 2.4, 3.8, 1.2),
            f"delivery at {wps:.1f} words/sec is brisk but clear",
            f"delivery at {wps:.1f} words/sec feels {'slow' if wps < 2.4 else 'rushed'}",
        ),
        term(
            "on_screen_words",
            float(screen_words),
            1.0 if 2 <= screen_words <= 6 else -1.0 if screen_words == 0 else -0.5,
            f"the on-screen headline is {screen_words} words, readable at a glance",
            "there is no on-screen headline for sound-off viewers" if screen_words == 0 else f"the on-screen headline is {screen_words} words, hard to read in time",
        ),
        term(
            "hook_seconds",
            hook_seconds,
            band(hook_seconds, 1.8, 3.2, 1.0),
            f"the hook lasts {hook_seconds:.1f}s, inside the 1.8-3.2s sweet spot",
            f"the hook runs {hook_seconds:.1f}s, {'too short to register' if hook_seconds < 1.8 else 'long enough that viewers may drop before the body'}",
        ),
    ]


def visual_terms(opening: Opening) -> list[Term]:
    if not opening.readable:
        return []
    motion = opening.motion
    return [
        term(
            "motion_energy",
            motion,
            1.0 - abs(math.log(max(motion, 1e-4) / 0.06)) / 1.6,
            f"the opening 1.5s has visible, natural movement (energy {motion:.3f})",
            f"the opening 1.5s is {'almost static' if motion < 0.06 else 'very jumpy'} (energy {motion:.3f})",
        ),
        term(
            "brightness",
            opening.brightness,
            band(opening.brightness, 0.35, 0.75, 0.4),
            "the first frame is well exposed",
            f"the first frame is {'dark' if opening.brightness < 0.35 else 'washed out'} ({opening.brightness:.0%} brightness)",
        ),
        term(
            "contrast",
            opening.contrast,
            (opening.contrast - 0.12) / 0.08,
            "the first frame has strong contrast",
            "the first frame looks flat (low contrast)",
        ),
    ]


def prior_terms(spec: VariantSpec, priors: Priors) -> list[Term]:
    levels = (("prior_archetype", "archetype", spec.hook.archetype), ("prior_voice", "voice", spec.voice.id), ("prior_location", "location", spec.location.id))
    terms: list[Term] = []
    for name, factor, level in levels:
        arm = priors.arm(factor, level)
        terms.append(
            term(
                name,
                arm.mean,
                (arm.mean - 0.5) * 4 if arm.trials else 0.0,
                f"'{level}' {factor} has won {arm.wins} of {arm.trials} past tests" if arm.trials else f"no live results yet for '{level}' {factor}",
                f"'{level}' {factor} has underperformed ({arm.wins} wins in {arm.trials} tests)",
            )
        )
    return terms


def copy_term(spec: VariantSpec) -> Term:
    score = spec.hook.score
    return term(
        "hook_score",
        score,
        (score - 0.5) * 2,
        f"the hook copy scores {score:.2f} on the writing rubric",
        f"the hook copy scores only {score:.2f} on the writing rubric",
    )


def explain(terms: list[Term], readable: bool) -> list[str]:
    positives = sorted((t for t in terms if t.contribution > REASON_THRESHOLD), key=lambda t: -t.contribution)[:3]
    negatives = sorted((t for t in terms if t.contribution < -REASON_THRESHOLD), key=lambda t: t.contribution)[:2]
    reasons = [f"+ {t.reason[:1].upper()}{t.reason[1:]}" for t in positives]
    reasons += [f"- {t.reason[:1].upper()}{t.reason[1:]}" for t in negatives]
    if not readable:
        reasons.append("Video frames could not be read; the score uses speech, copy and prior features only")
    return reasons


def prelaunch_score(video: Path, spec: VariantSpec, words: list[Word], hook_seconds: float, priors: Priors) -> Prelaunch:
    opening = read_opening(Path(video))
    terms = speech_terms(words, hook_seconds, spec.hook.on_screen_text) + visual_terms(opening) + [copy_term(spec)] + prior_terms(spec, priors)
    logit = BIAS + sum(t.contribution for t in terms)
    score = round(100.0 / (1.0 + math.exp(-logit)), 1)
    features = {t.name: round(t.value, 4) for t in terms}
    features.update({f"{t.name}_contribution": round(t.contribution, 4) for t in terms})
    features["video_readable"] = 1.0 if opening.readable else 0.0
    return Prelaunch(score=score, features=features, reasons=explain(terms, opening.readable))
