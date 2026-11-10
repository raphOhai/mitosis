from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mitosis.config import Brand
from mitosis.intelligence.copybank import ANGLES, Angle, classify_angle, partner_word
from mitosis.intelligence.hooks import ARCHETYPES
from mitosis.intelligence.llmio import array, ask_json, strict_object, string
from mitosis.intelligence.structure import sentences
from mitosis.intelligence.text import capitalize_first, normalize, split_sentences, tokens
from mitosis.models import CreativeDNA, Winner, Word
from mitosis.providers.base import LanguageModel

log = logging.getLogger(__name__)

DEFAULT_PACING = 2.6
FORMATS = ["talking_head", "voiceover", "mixed"]
FIRST_PERSON = {"i", "i'm", "i've", "i'd", "i'll", "me", "my", "we", "we'd", "we're", "our", "us"}
TONE_CUES = {
    "confessional": {"honestly", "admit", "truth", "confession", "never", "realized"},
    "playful": {"lol", "funny", "laugh", "joke", "literally", "okay"},
    "urgent": {"now", "today", "stop", "before", "tonight"},
    "reflective": {"realized", "noticed", "wondered", "thought", "remember"},
}

SYSTEM = (
    "You are a senior performance-creative strategist who reverse-engineers winning UGC video ads for Meta. "
    "You read the timestamped transcript, the hook/body/CTA split and the keyframes, and describe the creative DNA precisely: "
    "the angle, the specific audience insight, the promise, the hook archetype, the beats, and a reusable description of the on-screen "
    "person so image models can recreate exactly the same person. Never invent facts that are not in the ad."
)


def spoken(words: list[Word], start: float, end: float) -> str:
    return " ".join(w.text for w in words if start - 0.02 <= w.start < end - 0.02).strip()


def split_text(winner: Winner) -> tuple[str, str, str]:
    words = winner.words
    if words:
        end = words[-1].end + 1.0
        hook_end = winner.hook_end or words[0].end
        cta_start = max(winner.cta_start, hook_end) if winner.cta_start else end
        return spoken(words, 0.0, hook_end), spoken(words, hook_end, cta_start), spoken(words, cta_start, end)
    parts = split_sentences(winner.script)
    if not parts:
        return "", "", ""
    if len(parts) == 1:
        return parts[0], "", ""
    if len(parts) == 2:
        return parts[0], "", parts[1]
    return parts[0], " ".join(parts[1:-1]), parts[-1]


def pacing(words: list[Word]) -> float:
    if len(words) < 3:
        return DEFAULT_PACING
    span = words[-1].end - words[0].start
    return round(len(words) / span, 2) if span > 0.5 else DEFAULT_PACING


def classify_archetype(hook_text: str) -> str:
    text = normalize(hook_text).lower()
    scores = {name: sum(1 for cue in archetype.cues if re.search(cue, text)) for name, archetype in ARCHETYPES.items()}
    best, hits = max(scores.items(), key=lambda item: item[1])
    if hits:
        return best
    words = tokens(text)
    return "confession" if sum(1 for w in words if w in FIRST_PERSON) >= 2 else "curiosity_gap"


def detect_format(winner: Winner) -> str:
    if winner.talking_head:
        return "talking_head"
    if winner.shots and not any(shot.face for shot in winner.shots):
        return "voiceover"
    return "mixed"


def age_phrase(audience: str) -> str:
    numbers = [int(n) for n in re.findall(r"\b(1[89]|[2-6]\d)\b", audience)]
    if len(numbers) >= 2:
        midpoint = (numbers[0] + numbers[1]) / 2
    elif numbers:
        midpoint = numbers[0]
    else:
        midpoint = 35
    decade = int(midpoint // 10 * 10)
    position = "early" if midpoint - decade < 4 else "mid" if midpoint - decade < 7 else "late"
    return f"{position}-{decade}s"


def frame_look(frames: list[Path]) -> tuple[str, str]:
    stats: list[tuple[float, float, float]] = []
    for path in frames:
        image = cv2.imread(str(path))
        if image is None:
            continue
        small = cv2.resize(image, (64, max(1, int(64 * image.shape[0] / max(image.shape[1], 1)))))
        blue, _, red = (float(channel.mean()) for channel in cv2.split(small))
        stats.append((float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean()), red, blue))
    if not stats:
        return "an everyday home interior", "natural indoor light"
    brightness, red, blue = (float(v) for v in np.mean(np.array(stats), axis=0))
    warmth = "warm" if red > blue * 1.08 else "cool" if blue > red * 1.08 else "neutral"
    level = "bright" if brightness > 150 else "dim" if brightness < 80 else "soft"
    return "an everyday home interior", f"{level}, {warmth} natural light"


def hero_description(winner: Winner, brand: Brand, frames: list[Path]) -> str:
    audience = brand.audience.lower()
    person = "woman" if re.search(r"\bwomen\b|\bfemale\b", audience) and "men" not in audience.split() else "man"
    pronoun = "her" if person == "woman" else "his"
    _, light = frame_look(frames)
    framing = (
        "speaking straight into a handheld phone camera at eye level, medium close-up"
        if winner.talking_head
        else "filmed in casual handheld phone footage"
    )
    return (
        f"the same {person} in {pronoun} {age_phrase(brand.audience)} seen in the winning ad, {framing}, in {light}; "
        f"keep {pronoun} exact face, hairline, facial hair, skin tone, build, wardrobe and grooming exactly as in the reference frames"
    )


def audience_for(text: str, brand: Brand) -> str:
    lowered = text.lower()
    details: list[str] = []
    partner = partner_word(text)
    if partner in {"wife", "husband"}:
        details.append("married")
    if re.search(r"\b(?:kids?|baby|toddler|bedtime|school)\b", lowered):
        details.append("with young kids")
    if re.search(r"\b(?:years?|decade)\b", lowered):
        details.append("several years in")
    insight = ", ".join(details)
    return f"{brand.audience}{' (' + insight + ')' if insight else ''} who feel the relationship slipped into autopilot and want closeness back without therapy-speak"


def promise_for(angle: Angle, text: str, brand: Brand) -> str:
    lowered = text.lower()
    vehicle = "a 28-day challenge" if re.search(r"\b28\b|\bchallenge\b", lowered) else "short daily exercises"
    return f"{capitalize_first(angle.promise)} ({vehicle} in the {brand.product} app)"


def tone_for(words: list[str], brand: Brand) -> str:
    counts = {tone: sum(1 for w in words if w in cues) for tone, cues in TONE_CUES.items()}
    detected = [tone for tone, count in sorted(counts.items(), key=lambda item: -item[1]) if count][:2]
    first_person = sum(1 for w in words if w in FIRST_PERSON) / max(len(words), 1)
    if first_person > 0.08 and "confessional" not in detected:
        detected.insert(0, "first-person")
    base = [part.strip() for part in brand.tone.split(",") if part.strip()][:3]
    return ", ".join(dict.fromkeys(detected + base))


def beats_for(winner: Winner, hook: str, body: str, cta: str, angle: Angle) -> list[str]:
    groups = [" ".join(w.text for w in group) for group in sentences(winner.words)] or split_sentences(winner.script)
    beats = [f"hook: {hook}"] if hook else []
    body_sentences = [s for s in groups if s and s not in hook and s not in cta] or split_sentences(body)
    for index, sentence in enumerate(body_sentences[:4]):
        role = angle.beats[min(index + 1, len(angle.beats) - 1)] if angle.beats else "story"
        beats.append(f"{role}: {sentence}")
    if cta:
        beats.append(f"cta: {cta}")
    return beats


def offline_dna(winner: Winner, brand: Brand, frames: list[Path]) -> CreativeDNA:
    hook, body, cta = split_text(winner)
    full = " ".join(part for part in (hook, body, cta) if part) or winner.script
    angle = ANGLES[classify_angle(full)]
    setting, light = frame_look(frames)
    return CreativeDNA(
        angle=angle.label,
        audience=audience_for(full, brand),
        promise=promise_for(angle, full, brand),
        hook_archetype=classify_archetype(hook or full),
        hook_text=hook,
        body_text=body,
        cta_text=cta or brand.cta,
        beats=beats_for(winner, hook, body, cta, angle),
        hero_description=hero_description(winner, brand, frames),
        setting=f"{setting} in {light}",
        tone=tone_for(tokens(full), brand),
        pacing_wps=pacing(winner.words),
        format=detect_format(winner),
    )


def transcript_lines(words: list[Word]) -> str:
    return "\n".join(
        f"[{group[0].start:05.2f}-{group[-1].end:05.2f}] {' '.join(w.text for w in group)}" for group in sentences(words)
    )


def schema() -> dict[str, Any]:
    return strict_object(
        {
            "angle": string(),
            "audience": string(),
            "promise": string(),
            "hook_archetype": string(list(ARCHETYPES)),
            "hook_text": string(),
            "body_text": string(),
            "cta_text": string(),
            "beats": array(string()),
            "hero_description": string(),
            "setting": string(),
            "tone": string(),
            "format": string(FORMATS),
        }
    )


def prompt_for(winner: Winner, brand: Brand, baseline: CreativeDNA) -> str:
    archetypes = "\n".join(f"- {name}: {a.description}" for name, a in ARCHETYPES.items())
    return (
        f"Brand: {brand.name} - {brand.category}. Target: {brand.audience}. Tone: {brand.tone}. CTA: {brand.cta}.\n"
        f"Video: {winner.info.duration:.1f}s, {winner.info.width}x{winner.info.height}, {len(winner.shots)} shots, "
        f"talking head: {winner.talking_head}.\n"
        f"Hook ends at {winner.hook_end:.2f}s, CTA starts at {winner.cta_start:.2f}s.\n"
        f"Timestamped transcript:\n{transcript_lines(winner.words) or winner.script}\n\n"
        f"Split from timing - hook: {baseline.hook_text}\nbody: {baseline.body_text}\ncta: {baseline.cta_text}\n\n"
        f"Hook archetypes:\n{archetypes}\n\n"
        "Return the creative DNA. Keep hook_text, body_text and cta_text verbatim from the split. Angle starts with a short "
        "label followed by a colon. hero_description must describe the person in the keyframes (apparent age, hair, facial hair, "
        "build, skin tone, wardrobe, framing) so a model can recreate exactly the same person, and must end with "
        "'exactly as in the reference frames'. Beats are 3-6 short 'role: content' lines."
    )


def merge(baseline: CreativeDNA, data: dict[str, Any]) -> CreativeDNA:
    values = baseline.model_dump()
    for key in ("angle", "audience", "promise", "hero_description", "setting", "tone"):
        if isinstance(data.get(key), str) and data[key].strip():
            values[key] = data[key].strip()
    for key in ("hook_text", "body_text", "cta_text"):
        if not values[key] and isinstance(data.get(key), str):
            values[key] = data[key].strip()
    if data.get("hook_archetype") in ARCHETYPES:
        values["hook_archetype"] = data["hook_archetype"]
    if data.get("format") in FORMATS and values["format"] != "talking_head":
        values["format"] = data["format"]
    beats = [str(b).strip() for b in data.get("beats", []) if str(b).strip()] if isinstance(data.get("beats"), list) else []
    if beats:
        values["beats"] = beats
    return CreativeDNA(**values)


def deconstruct(winner: Winner, brand: Brand, llm: LanguageModel | None = None, frames: list[Path] | None = None) -> CreativeDNA:
    keyframes = [Path(f) for f in frames or [] if Path(f).exists()]
    baseline = offline_dna(winner, brand, keyframes)
    if llm is None:
        return baseline
    data = ask_json(llm, SYSTEM, prompt_for(winner, brand, baseline), schema(), images=keyframes[:6], max_tokens=3000)
    if data is None:
        return baseline
    return merge(baseline, data)
