from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import field_validator

from mitosis.models import Check, GateInput, HeroPack, Model, VariantSpec
from mitosis.qa.footage import fit_within, grab_frames, video_duration, write_jpeg

log = logging.getLogger(__name__)

HOOK_POSITIONS = (0.15, 0.5, 0.85)
MIN_BODY_SECONDS = 0.3
FRAME_MAX_SIDE = 1024
FRAME_QUALITY = 88
HERO_MATCH_THRESHOLD = 0.6

SYSTEM_PROMPT = (
    "You are the final visual QA reviewer for paid social video ads that run on Meta (Reels, Stories and Feed). "
    "You inspect still frames from a rendered ad and report only what is visible in them. "
    "A defect you miss costs ad spend or gets the ad rejected, so be strict about AI-generation artifacts and identity drift, "
    "but do not invent problems that are not visible. Answer with the requested JSON only."
)

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "artifacts",
        "artifact_detail",
        "same_person",
        "text_legible",
        "brand_safe",
        "policy_risk",
        "overall",
        "notes",
    ],
    "properties": {
        "artifacts": {
            "type": "boolean",
            "description": "True if any ad frame shows an AI-generation or rendering defect.",
        },
        "artifact_detail": {
            "type": "string",
            "description": "Which frame and what defect; empty string when there is none.",
        },
        "same_person": {
            "type": "number",
            "description": "Probability from 0 to 1 that the person in the ad frames is the same individual as the reference photo.",
        },
        "text_legible": {
            "type": "boolean",
            "description": "True if every on-screen text element is fully inside the frame and easy to read on a phone.",
        },
        "brand_safe": {
            "type": "boolean",
            "description": "True if nothing visible would embarrass a mainstream relationships brand.",
        },
        "policy_risk": {
            "type": "string",
            "enum": ["none", "low", "high"],
            "description": "Likelihood that Meta ad review rejects these visuals or on-screen text.",
        },
        "overall": {
            "type": "number",
            "description": "Overall production quality from 0 to 1 as a native-looking, scroll-stopping ad.",
        },
        "notes": {
            "type": "string",
            "description": "One or two sentences that justify the verdict.",
        },
    },
}


class Verdict(Model):
    artifacts: bool
    artifact_detail: str = ""
    same_person: float
    text_legible: bool
    brand_safe: bool
    policy_risk: Literal["none", "low", "high"]
    overall: float
    notes: str = ""

    @field_validator("same_person", "overall")
    @classmethod
    def within_unit_range(cls, value: float) -> float:
        return min(1.0, max(0.0, value))

    @field_validator("policy_risk", mode="before")
    @classmethod
    def lowercase_risk(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value


def sample_times(inp: GateInput) -> list[tuple[str, float]]:
    duration = video_duration(Path(inp.video)) or inp.expected_seconds
    hook_end = min(inp.hook_seconds, duration) if inp.hook_seconds > 0 else min(2.0, duration)
    moments = [("hook", hook_end * position) for position in HOOK_POSITIONS]
    if duration - hook_end >= MIN_BODY_SECONDS:
        moments.append(("body", hook_end + (duration - hook_end) / 2))
    return moments


def export_frames(inp: GateInput, folder: Path) -> list[tuple[str, float, Path]]:
    moments = sample_times(inp)
    exported: list[tuple[str, float, Path]] = []
    for (phase, moment), (_, pixels) in zip(moments, grab_frames(Path(inp.video), [m for _, m in moments])):
        if pixels is None:
            continue
        resized, _ = fit_within(pixels, FRAME_MAX_SIDE)
        target = folder / f"frame_{len(exported) + 1:02d}_{phase}.jpg"
        exported.append((phase, moment, write_jpeg(resized, target, FRAME_QUALITY)))
    return exported


def build_prompt(inp: GateInput, spec: VariantSpec, frames: list[tuple[str, float, Path]], has_reference: bool) -> str:
    offset = 2 if has_reference else 1
    listing = "\n".join(
        f"- Image {index + offset}: {phase} frame at {moment:.2f}s" for index, (phase, moment, _) in enumerate(frames)
    )
    reference = (
        "Image 1 is the reference photo of the hero: the exact person who must appear in the ad."
        if has_reference
        else "There is no reference photo; set same_person to 1."
    )
    return "\n".join(
        [
            reference,
            f"The following images are frames from the rendered {inp.aspect.value} ad (hook ends at {inp.hook_seconds:.2f}s):",
            listing,
            "",
            f'Spoken hook: "{inp.hook_text}"',
            f'On-screen headline: "{inp.on_screen_text}"',
            f"Intended hook visual: {spec.hook.visual}",
            f"Intended location: {spec.location.label}",
            "",
            "Report:",
            "- artifacts: warped or melting faces, extra, missing or fused fingers, broken anatomy, duplicated people, "
            "objects morphing, smeared textures, garbled or misspelled text, generator watermarks or logos.",
            "- same_person: compare face shape, hairline, facial hair, skin tone, apparent age and build with the reference.",
            "- text_legible: subtitles and headline readable at phone size, enough contrast, not cut off, not overlapping each other.",
            "- brand_safe: false for nudity or implied nudity, sexual acts or sexualized framing, focus on body parts, "
            "anyone who could be a minor, violence, drugs, profanity or shocking imagery.",
            "- policy_risk: Meta ad review risk, including adult content and text that asserts or implies the viewer's personal "
            "attributes (for example calling out their relationship or sexual problems directly).",
            "- overall: production quality and native UGC feel.",
        ]
    )


def parse_verdict(data: Any) -> Verdict:
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        raise ValueError(f"judge returned {type(data).__name__}, expected an object")
    return Verdict.model_validate(data)


def skipped(detail: str) -> list[Check]:
    return [Check(name="vision_judge", passed=False, skipped=True, severity="warn", detail=detail)]


def verdict_checks(verdict: Verdict, spec: VariantSpec, has_reference: bool) -> list[Check]:
    unsafe = not verdict.brand_safe or verdict.policy_risk == "high"
    hero_match = Check(
        name="hero_match_judge",
        passed=verdict.same_person >= HERO_MATCH_THRESHOLD,
        severity="warn",
        score=round(verdict.same_person, 3),
        threshold=HERO_MATCH_THRESHOLD,
        retry="shot",
        detail=f"judge rates same person at {verdict.same_person:.2f}",
    )
    if not has_reference:
        hero_match = Check(name="hero_match_judge", passed=False, skipped=True, severity="warn", detail="no hero reference image")
    return [
        Check(
            name="visual_artifacts",
            passed=not verdict.artifacts,
            score=round(verdict.overall, 3),
            retry="shot",
            detail=verdict.artifact_detail.strip() or f"no generation artifacts seen (overall {verdict.overall:.2f})",
        ),
        hero_match,
        Check(
            name="brand_safety",
            passed=not unsafe,
            retry="shot" if spec.new_shot else "none",
            detail=f"brand safe: {'yes' if verdict.brand_safe else 'no'}, policy risk {verdict.policy_risk}. {verdict.notes}".strip(),
        ),
        Check(
            name="text_legibility",
            passed=verdict.text_legible,
            severity="warn",
            detail="on-screen text legible" if verdict.text_legible else f"text legibility issue. {verdict.notes}".strip(),
        ),
    ]


def judge(inp: GateInput, spec: VariantSpec, hero: HeroPack, llm=None) -> list[Check]:
    if llm is None:
        return skipped("no vision model configured; visual review skipped")
    reference = next((Path(path) for path in hero.references[:1] if Path(path).is_file()), None)
    with tempfile.TemporaryDirectory(prefix="mitosis-judge-") as scratch:
        frames = export_frames(inp, Path(scratch))
        if not frames:
            return skipped(f"no frames could be read from {Path(inp.video).name}")
        images = ([reference] if reference else []) + [path for _, _, path in frames]
        prompt = build_prompt(inp, spec, frames, reference is not None)
        try:
            result = llm.json(system=SYSTEM_PROMPT, prompt=prompt, schema=VERDICT_SCHEMA, images=images)
            verdict = parse_verdict(result.data)
        except ValueError as exc:
            log.warning("vision judge returned an unusable verdict for %s: %s", inp.variant_id, exc)
            return skipped(f"judge verdict unusable: {exc}")
        except Exception as exc:
            log.warning("vision judge failed for %s: %s", inp.variant_id, exc, exc_info=True)
            return skipped(f"judge unavailable: {type(exc).__name__}: {exc}")
    return verdict_checks(verdict, spec, reference is not None)
