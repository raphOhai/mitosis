from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


class Aspect(str, Enum):
    VERTICAL = "9:16"
    PORTRAIT = "4:5"
    SQUARE = "1:1"

    @property
    def size(self) -> tuple[int, int]:
        return {"9:16": (1080, 1920), "4:5": (1080, 1350), "1:1": (1080, 1080)}[self.value]

    @property
    def slug(self) -> str:
        return self.value.replace(":", "x")

    @property
    def ratio(self) -> float:
        width, height = self.size
        return width / height


class Model(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, use_enum_values=False)


class Box(Model):
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


class Word(Model):
    text: str
    start: float
    end: float

    def shifted(self, offset: float) -> Word:
        return Word(text=self.text, start=self.start + offset, end=self.end + offset)


class MediaInfo(Model):
    path: Path
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool
    sample_rate: int | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    pix_fmt: str | None = None
    bitrate: int | None = None

    @property
    def ratio(self) -> float:
        return self.width / self.height


class Shot(Model):
    index: int
    start: float
    end: float
    keyframe: Path | None = None
    face: Box | None = None
    sharpness: float = 0.0

    @property
    def length(self) -> float:
        return self.end - self.start


class Stems(Model):
    vocals: Path
    accompaniment: Path


class Winner(Model):
    id: str
    video: Path
    script: str
    info: MediaInfo
    audio: Path
    stems: Stems | None = None
    words: list[Word] = Field(default_factory=list)
    shots: list[Shot] = Field(default_factory=list)
    hook_end: float = 0.0
    cta_start: float = 0.0
    talking_head: bool = False


class CreativeDNA(Model):
    angle: str
    audience: str
    promise: str
    hook_archetype: str
    hook_text: str
    body_text: str
    cta_text: str
    beats: list[str] = Field(default_factory=list)
    hero_description: str
    setting: str
    tone: str
    pacing_wps: float = 2.6
    format: Literal["talking_head", "voiceover", "mixed"] = "mixed"


class HeroPack(Model):
    id: str
    description: str
    references: list[Path] = Field(default_factory=list)
    embedding: list[float] | None = None
    embedding_model: str | None = None
    seed: int = 0
    lora: str | None = None
    lora_scale: float = 0.85
    negative: str = ""


class PolicyFlag(Model):
    rule: str
    severity: Literal["block", "warn"]
    span: str = ""
    message: str
    suggestion: str = ""


class Hook(Model):
    id: str
    archetype: str
    text: str
    on_screen_text: str
    visual: str
    motion: str
    rationale: str = ""
    score: float = 0.0
    flags: list[PolicyFlag] = Field(default_factory=list)


class Voice(Model):
    id: str
    label: str
    gender: Literal["male", "female", "neutral"] = "male"
    age: str = ""
    tone: str = ""
    elevenlabs_id: str | None = None
    say_voice: str | None = None
    pitch_semitones: float = 0.0
    stability: float = 0.45
    similarity: float = 0.8
    style: float = 0.25
    speed: float = 1.0


class Location(Model):
    id: str
    label: str
    prompt: str
    grade: str = "neutral"
    original: bool = False


class VariantSpec(Model):
    id: str
    index: int
    hook: Hook
    voice: Voice
    location: Location
    aspects: list[Aspect]
    seed: int
    new_shot: bool = True
    changed: list[str] = Field(default_factory=list)
    body_mode: Literal["keep", "sts", "tts"] = "sts"


class CostEstimate(Model):
    usd: float = 0.0
    seconds: float = 0.0
    per_variant_usd: float = 0.0
    breakdown: dict[str, float] = Field(default_factory=dict)


class VariantPlan(Model):
    winner_id: str
    design: str
    variants: list[VariantSpec]
    levels: dict[str, list[str]] = Field(default_factory=dict)
    coverage: dict[str, float] = Field(default_factory=dict)
    estimate: CostEstimate = Field(default_factory=CostEstimate)
    notes: list[str] = Field(default_factory=list)


class Check(Model):
    name: str
    passed: bool
    severity: Literal["block", "warn"] = "block"
    score: float | None = None
    threshold: float | None = None
    detail: str = ""
    skipped: bool = False
    retry: Literal["none", "shot", "render"] = "none"


class QAReport(Model):
    variant_id: str
    aspect: Aspect
    checks: list[Check] = Field(default_factory=list)

    @computed_field
    @property
    def passed(self) -> bool:
        return all(c.passed or c.skipped or c.severity == "warn" for c in self.checks)

    @property
    def blocking(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and not c.skipped and c.severity == "block"]


class Prelaunch(Model):
    score: float
    features: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


class Render(Model):
    aspect: Aspect
    path: Path
    thumbnail: Path | None = None
    subtitles: Path | None = None
    duration: float = 0.0
    qa: QAReport | None = None
    name: str = ""


class VariantResult(Model):
    spec: VariantSpec
    renders: list[Render] = Field(default_factory=list)
    status: Literal["passed", "flagged", "failed"] = "failed"
    attempts: int = 0
    usd: float = 0.0
    shadow_usd: float = 0.0
    seconds: float = 0.0
    prelaunch: Prelaunch | None = None
    review: Literal["pending", "approved", "rejected"] = "pending"
    error: str | None = None
    hook_seconds: float = 0.0
    words: list[Word] = Field(default_factory=list)


class LedgerEntry(Model):
    ts: float
    variant: str | None = None
    stage: str | None = None
    kind: str
    provider: str
    model: str
    usage: dict[str, float] = Field(default_factory=dict)
    usd: float = 0.0
    shadow_usd: float = 0.0
    saved_usd: float = 0.0
    seconds: float = 0.0
    cached: bool = False
    attempt: int = 1
    ok: bool = True
    error: str | None = None


class Totals(Model):
    variants: int = 0
    passed: int = 0
    flagged: int = 0
    failed: int = 0
    renders: int = 0
    usd: float = 0.0
    shadow_usd: float = 0.0
    seconds: float = 0.0
    usd_per_variant: float = 0.0
    shadow_per_variant: float = 0.0
    seconds_per_variant: float = 0.0
    cache_hits: int = 0
    saved_usd: float = 0.0
    by_stage: dict[str, float] = Field(default_factory=dict)
    by_provider: dict[str, float] = Field(default_factory=dict)


class BetaArm(Model):
    alpha: float = 1.0
    beta: float = 1.0
    trials: int = 0
    wins: int = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class Priors(Model):
    factors: dict[str, dict[str, BetaArm]] = Field(default_factory=dict)
    updated_at: str | None = None

    def arm(self, factor: str, level: str) -> BetaArm:
        return self.factors.get(factor, {}).get(level, BetaArm())


class Brief(Model):
    winner_id: str
    video: Path
    script: str
    variants: int = 10
    aspects: list[Aspect] = Field(default_factory=lambda: [Aspect.VERTICAL, Aspect.PORTRAIT, Aspect.SQUARE])
    voices: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    hook_count: int = 6
    archetypes: list[str] | None = None
    budget_usd: float = 40.0
    max_attempts: int = 3
    offline: bool | None = None
    seed: int = 7
    music: Path | None = None
    body_mode: Literal["auto", "keep", "sts", "tts"] = "auto"
    out: Path | None = None
    notes: str = ""


class Event(Model):
    type: Literal["stage", "variant", "log", "done", "error"]
    stage: str | None = None
    variant: str | None = None
    status: str | None = None
    message: str = ""
    progress: float | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    ts: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())


class RunManifest(Model):
    run_id: str
    created_at: str
    out_dir: Path
    brief: Brief
    winner: Winner
    dna: CreativeDNA
    hero: HeroPack
    hooks: list[Hook] = Field(default_factory=list)
    rejected_hooks: list[Hook] = Field(default_factory=list)
    plan: VariantPlan
    results: list[VariantResult] = Field(default_factory=list)
    totals: Totals = Field(default_factory=Totals)
    mode: Literal["offline", "live", "hybrid"] = "offline"
    models: dict[str, list[str]] = Field(default_factory=dict)
    ledger: Path | None = None
    report: Path | None = None
    exports: dict[str, Path] = Field(default_factory=dict)


class Loudness(Model):
    integrated: float
    true_peak: float
    lra: float


class AudioPart(Model):
    path: Path
    offset: float
    gain_db: float = 0.0


class Headline(Model):
    text: str
    start: float
    end: float


class Segment(Model):
    path: Path
    seconds: float
    start: float = 0.0
    end: float | None = None
    reframe: Literal["shots", "cover"] = "cover"
    shots: list[Shot] = Field(default_factory=list)
    info: MediaInfo | None = None


class RenderJob(Model):
    segments: list[Segment]
    audio: Path
    aspect: Aspect
    out: Path
    subtitles: Path | None = None


class GateInput(Model):
    variant_id: str
    aspect: Aspect
    video: Path
    subtitles: Path | None = None
    expected_seconds: float
    hook_seconds: float
    hook_text: str
    on_screen_text: str
    spoken_text: str
