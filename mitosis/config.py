from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from mitosis.models import Aspect, Brief, Location, Model, Voice

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


class SafeZone(Model):
    top: float = 0.0
    bottom: float = 0.0
    left: float = 0.0
    right: float = 0.0

    def rect(self, aspect: Aspect) -> tuple[int, int, int, int]:
        width, height = aspect.size
        return (
            round(width * self.left),
            round(height * self.top),
            round(width * (1 - self.right)),
            round(height * (1 - self.bottom)),
        )


class SubtitleStyle(Model):
    font: str = "Arial Black"
    size_ratio: float = 0.062
    primary: str = "#FFFFFF"
    highlight: str = "#FFD84D"
    outline: str = "#000000"
    outline_ratio: float = 0.09
    shadow: float = 0.0
    uppercase: bool = True
    max_words: int = 3
    max_chars: int = 18
    max_gap: float = 0.45
    pop: bool = True
    anchor: dict[str, float] = Field(default_factory=lambda: {"9:16": 0.60, "4:5": 0.74, "1:1": 0.76})


class HeadlineStyle(Model):
    font: str = "Arial Black"
    size_ratio: float = 0.058
    color: str = "#111111"
    box: str = "#FFFFFF"
    box_opacity: float = 0.94
    max_chars_per_line: int = 22
    anchor: dict[str, float] = Field(default_factory=lambda: {"9:16": 0.20, "4:5": 0.12, "1:1": 0.12})


class LoudnessSpec(Model):
    target_lufs: float = -14.0
    true_peak: float = -1.0
    lra: float = 11.0
    tolerance: float = 1.5
    bed_gain_db: float = -20.0


class VideoSpec(Model):
    fps: int = 30
    crf: int = 19
    preset: str = "medium"
    codec: str = "libx264"
    pix_fmt: str = "yuv420p"
    audio_bitrate: str = "192k"
    sample_rate: int = 48000
    max_bytes: int = 4_000_000_000


class Brand(Model):
    name: str
    product: str
    category: str
    audience: str
    cta: str
    tone: str
    subtitle: SubtitleStyle = Field(default_factory=SubtitleStyle)
    headline: HeadlineStyle = Field(default_factory=HeadlineStyle)
    safe_zones: dict[str, SafeZone] = Field(default_factory=dict)
    loudness: LoudnessSpec = Field(default_factory=LoudnessSpec)
    video: VideoSpec = Field(default_factory=VideoSpec)
    naming: str = "{brand}_{winner}_{variant}_{hook}_{voice}_{location}_{aspect}"
    min_hook_seconds: float = 1.4
    max_hook_seconds: float = 4.5
    identity_threshold: float = 0.42
    image_identity_threshold: float = 0.45
    shot_style: str = ""
    policy: dict[str, Any] = Field(default_factory=dict)

    def safe_zone(self, aspect: Aspect) -> SafeZone:
        return self.safe_zones.get(aspect.value, SafeZone())


class ModelSpec(Model):
    id: str
    provider: str
    kind: str
    model: str = ""
    price: dict[str, float] = Field(default_factory=dict)
    requires: list[str] = Field(default_factory=list)
    concurrency: int = 4
    timeout: float = 900.0
    durations: list[float] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    offline: bool = False
    notes: str = ""

    def snap(self, seconds: float) -> float:
        if not self.durations:
            return seconds
        fitting = [d for d in sorted(self.durations) if d >= seconds - 1e-6]
        return fitting[0] if fitting else max(self.durations)

    def cost(self, usage: dict[str, float]) -> float:
        return round(sum(self.price.get(unit, 0.0) * amount for unit, amount in usage.items()), 6)


class Catalog(Model):
    chains: dict[str, list[str]]
    models: dict[str, ModelSpec]

    def chain(self, kind: str) -> list[ModelSpec]:
        return [self.models[name] for name in self.chains.get(kind, []) if name in self.models]


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_yaml(path: Path) -> Any:
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


class Settings(Model):
    root: Path
    config_dir: Path
    runs_dir: Path
    cache_dir: Path
    models_dir: Path
    priors_path: Path
    workers: int = 4
    offline: bool | None = None

    @classmethod
    def load(cls, root: Path | None = None, **overrides: Any) -> Settings:
        base = Path(root or os.environ.get("MITOSIS_HOME") or PACKAGE_ROOT).resolve()
        load_env(base / ".env")
        offline_env = os.environ.get("MITOSIS_OFFLINE")
        state = base / ".mitosis"
        values: dict[str, Any] = {
            "root": base,
            "config_dir": base / "config",
            "runs_dir": base / "runs",
            "cache_dir": state / "cache",
            "models_dir": Path(os.environ.get("MITOSIS_MODELS_DIR", Path.home() / ".cache" / "mitosis" / "models")),
            "priors_path": state / "priors.json",
            "workers": int(os.environ.get("MITOSIS_WORKERS", "4")),
            "offline": None if offline_env is None else offline_env.lower() in {"1", "true", "yes"},
        }
        values.update({k: v for k, v in overrides.items() if v is not None})
        settings = cls(**values)
        for directory in (settings.runs_dir, settings.cache_dir, settings.models_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return settings

    @cached_property
    def brand(self) -> Brand:
        data = read_yaml(self.config_dir / "brand.yaml")
        policy_path = self.config_dir / "policy.yaml"
        if policy_path.exists():
            data["policy"] = read_yaml(policy_path)
        return Brand(**data)

    @cached_property
    def catalog(self) -> Catalog:
        data = read_yaml(self.config_dir / "models.yaml")
        models = {key: ModelSpec(id=key, **value) for key, value in data.get("models", {}).items()}
        return Catalog(chains=data.get("chains", {}), models=models)

    @cached_property
    def voices(self) -> dict[str, Voice]:
        data = read_yaml(self.config_dir / "voices.yaml")
        return {key: Voice(id=key, **value) for key, value in data.get("voices", {}).items()}

    @cached_property
    def locations(self) -> dict[str, Location]:
        data = read_yaml(self.config_dir / "locations.yaml")
        return {key: Location(id=key, **value) for key, value in data.get("locations", {}).items()}


def load_brief(path: Path) -> Brief:
    path = Path(path).resolve()
    data = read_yaml(path)
    base = path.parent
    winner = data.pop("winner", {})
    script = winner.get("script", "")
    script_path = base / script if script and not script.strip().count(" ") and (base / script).exists() else None
    if script_path:
        script = script_path.read_text().strip()
    fields: dict[str, Any] = {
        "winner_id": winner.get("id", path.stem),
        "video": (base / winner["video"]).resolve(),
        "script": script.strip(),
    }
    if data.get("music"):
        data["music"] = (base / data["music"]).resolve()
    if data.get("out"):
        data["out"] = (base / data["out"]).resolve()
    if "aspects" in data:
        data["aspects"] = [Aspect(a) for a in data["aspects"]]
    fields.update(data)
    return Brief(**fields)
