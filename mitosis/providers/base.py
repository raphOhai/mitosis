from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import Field

from mitosis.config import ModelSpec, Settings
from mitosis.models import Aspect, Model, Voice, Word


class ProviderError(RuntimeError):
    pass


class TransientError(ProviderError):
    pass


class PermanentError(ProviderError):
    pass


class GenResult(Model):
    path: Path | None = None
    data: Any = None
    words: list[Word] = Field(default_factory=list)
    duration: float | None = None
    usage: dict[str, float] = Field(default_factory=dict)
    provider: str = ""
    model: str = ""
    seconds: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)
    cached: bool = False
    usd: float = 0.0
    shadow_usd: float = 0.0


class Provider:
    def __init__(self, spec: ModelSpec, settings: Settings) -> None:
        self.spec = spec
        self.settings = settings

    @property
    def name(self) -> str:
        return self.spec.id


@runtime_checkable
class LanguageModel(Protocol):
    def json(
        self,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        images: list[Path] | None = None,
        max_tokens: int = 8000,
    ) -> GenResult: ...


@runtime_checkable
class ImageModel(Protocol):
    def generate(
        self,
        prompt: str,
        references: list[Path],
        aspect: Aspect,
        seed: int,
        out: Path,
        negative: str = "",
    ) -> GenResult: ...


@runtime_checkable
class VideoModel(Protocol):
    def animate(
        self,
        image: Path,
        prompt: str,
        seconds: float,
        aspect: Aspect,
        seed: int,
        out: Path,
    ) -> GenResult: ...


@runtime_checkable
class SpeechModel(Protocol):
    def speak(self, text: str, voice: Voice, out: Path) -> GenResult: ...


@runtime_checkable
class VoiceModel(Protocol):
    def convert(self, audio: Path, voice: Voice, out: Path) -> GenResult: ...


@runtime_checkable
class TranscribeModel(Protocol):
    def transcribe(self, audio: Path, out: Path, hint: str | None = None) -> GenResult: ...


@runtime_checkable
class LipSyncModel(Protocol):
    def sync(self, video: Path, audio: Path, out: Path) -> GenResult: ...


@runtime_checkable
class UpscaleModel(Protocol):
    def upscale(self, video: Path, out: Path, factor: int = 2) -> GenResult: ...


FACTORIES: dict[tuple[str, str], type[Provider]] = {}


def register(provider: str, *kinds: str):
    def decorate(cls: type[Provider]) -> type[Provider]:
        for kind in kinds:
            FACTORIES[(provider, kind)] = cls
        return cls

    return decorate


KIND_METHODS: dict[str, str] = {
    "llm": "json",
    "image": "generate",
    "video": "animate",
    "speech": "speak",
    "voice": "convert",
    "transcribe": "transcribe",
    "lipsync": "sync",
    "upscale": "upscale",
}
