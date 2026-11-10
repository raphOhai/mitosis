from __future__ import annotations

import base64
import binascii
import json
import logging
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx

from mitosis.models import Voice, Word
from mitosis.providers.base import GenResult, PermanentError, TransientError, register
from mitosis.providers.remote import HttpProvider, env, media_duration, media_type, payload, to_wav

log = logging.getLogger(__name__)

API_URL = "https://api.elevenlabs.io"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
TARGET_SAMPLE_RATE = 48000
RAW_CODECS = {"pcm": "s16le", "ulaw": "mulaw", "alaw": "alaw"}
CONTAINER_SUFFIX = {"opus": "ogg"}
TERM_PATTERN = re.compile(r"[A-Za-z][A-Za-z'\-]+")


def voice_settings(voice: Voice, include_speed: bool) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "stability": voice.stability,
        "similarity_boost": voice.similarity,
        "style": voice.style,
        "use_speaker_boost": True,
    }
    if include_speed:
        settings["speed"] = voice.speed
    return settings


def decode_format(output_format: str) -> tuple[str, list[str]]:
    codec, _, rest = output_format.partition("_")
    rate = rest.split("_", 1)[0] or str(TARGET_SAMPLE_RATE)
    if codec in RAW_CODECS:
        return codec, ["-f", RAW_CODECS[codec], "-ar", rate, "-ac", "1"]
    return CONTAINER_SUFFIX.get(codec, codec), []


def write_wav(audio: bytes, output_format: str, out: Path) -> Path:
    if not audio:
        raise TransientError("elevenlabs returned empty audio")
    suffix, input_args = decode_format(output_format)
    out.parent.mkdir(parents=True, exist_ok=True)
    source = out.with_name(f".{out.stem}.source.{suffix}")
    try:
        source.write_bytes(audio)
        return to_wav(source, out, TARGET_SAMPLE_RATE, input_args)
    finally:
        source.unlink(missing_ok=True)


def character_tokens(characters: Sequence[str], starts: Sequence[float], ends: Sequence[float]) -> Iterator[tuple[str, float, float]]:
    buffer: list[str] = []
    first = last = 0.0
    for char, start, end in zip(characters, starts, ends):
        if char.isspace():
            if buffer:
                yield "".join(buffer), first, last
                buffer = []
            continue
        if not buffer:
            first = float(start)
        buffer.append(char)
        last = float(end)
    if buffer:
        yield "".join(buffer), first, last


def words_from_alignment(alignment: dict[str, Any] | None) -> list[Word]:
    if not alignment:
        return []
    tokens = character_tokens(
        alignment.get("characters") or [],
        alignment.get("character_start_times_seconds") or [],
        alignment.get("character_end_times_seconds") or [],
    )
    words: list[Word] = []
    for text, start, end in tokens:
        if words and not any(char.isalnum() for char in text):
            previous = words[-1]
            words[-1] = Word(text=previous.text + text, start=previous.start, end=max(previous.end, end))
            continue
        words.append(Word(text=text, start=start, end=max(start, end)))
    return words


def words_from_transcript(transcript: dict[str, Any]) -> list[Word]:
    words: list[Word] = []
    for item in transcript.get("words") or []:
        if item.get("type", "word") != "word" or item.get("start") is None:
            continue
        start = float(item["start"])
        end = float(item["end"]) if item.get("end") is not None else start
        text = str(item.get("text", "")).strip()
        if text:
            words.append(Word(text=text, start=start, end=max(start, end)))
    return words


def keyterms(hint: str, limit: int) -> list[str]:
    seen: dict[str, str] = {}
    for match in TERM_PATTERN.finditer(hint):
        term = match.group(0).strip("'-")
        if len(term) >= 5 and term.lower() not in seen:
            seen[term.lower()] = term
    return list(seen.values())[:limit]


def read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise PermanentError(f"cannot read audio {path}: {exc}") from exc


class ElevenLabsProvider(HttpProvider):
    service = "elevenlabs"

    def base_url(self) -> str:
        return str(self.params.get("api_url", API_URL))

    def auth_headers(self) -> dict[str, str]:
        return {"xi-api-key": env("ELEVENLABS_API_KEY")}

    @property
    def output_format(self) -> str:
        return str(self.params.get("output_format", DEFAULT_OUTPUT_FORMAT))

    def voice_id(self, voice: Voice) -> str:
        voice_id = voice.elevenlabs_id or self.params.get("default_voice_id")
        if not voice_id:
            raise PermanentError(f"voice '{voice.id}' has no elevenlabs_id")
        return str(voice_id)

    def request_meta(self, response: httpx.Response, voice_id: str | None = None) -> dict[str, Any]:
        meta: dict[str, Any] = {"model_id": self.spec.model, "request_id": response.headers.get("request-id")}
        if voice_id:
            meta["voice_id"] = voice_id
        return meta


@register("elevenlabs", "speech")
class ElevenLabsSpeech(ElevenLabsProvider):
    def speak(self, text: str, voice: Voice, out: Path) -> GenResult:
        if not text.strip():
            raise PermanentError("cannot synthesize empty text")
        voice_id = self.voice_id(voice)
        body: dict[str, Any] = {
            "text": text,
            "model_id": self.spec.model,
            "voice_settings": voice_settings(voice, include_speed=bool(self.params.get("speed", True))),
            **self.params.get("extra", {}),
        }
        response = self.call(
            "POST",
            f"/v1/text-to-speech/{voice_id}/with-timestamps",
            params={"output_format": self.output_format},
            json=body,
        )
        data = payload(response, self.service)
        try:
            audio = base64.b64decode(data.get("audio_base64") or "", validate=True)
        except (binascii.Error, ValueError) as exc:
            raise TransientError(f"elevenlabs returned malformed audio: {exc}") from exc
        write_wav(audio, self.output_format, out)
        words = words_from_alignment(data.get("alignment") or data.get("normalized_alignment"))
        if not words:
            log.warning("elevenlabs %s returned no alignment for voice %s", self.spec.model, voice_id)
        duration = media_duration(out) or (words[-1].end if words else None)
        return GenResult(
            path=out,
            duration=duration,
            words=words,
            usage={"kchar": len(text) / 1000},
            meta=self.request_meta(response, voice_id),
        )


@register("elevenlabs", "voice")
class ElevenLabsVoiceChanger(ElevenLabsProvider):
    def convert(self, audio: Path, voice: Voice, out: Path) -> GenResult:
        voice_id = self.voice_id(voice)
        seconds = media_duration(audio)
        if seconds is None:
            raise PermanentError(f"cannot read duration of {audio}")
        form: dict[str, Any] = {
            "model_id": self.spec.model,
            "voice_settings": json.dumps(voice_settings(voice, include_speed=False)),
            "remove_background_noise": str(bool(self.params.get("remove_background_noise", True))).lower(),
        }
        if self.params.get("seed") is not None:
            form["seed"] = str(int(self.params["seed"]))
        path = str(self.params.get("path", "/v1/voice-changer/{voice_id}")).format(voice_id=voice_id)
        response = self.call(
            "POST",
            path,
            params={"output_format": self.output_format},
            data=form,
            files={"audio": (audio.name, read_bytes(audio), media_type(audio))},
        )
        write_wav(response.content, self.output_format, out)
        return GenResult(
            path=out,
            duration=media_duration(out) or seconds,
            usage={"minute": seconds / 60},
            meta=self.request_meta(response, voice_id),
        )


@register("elevenlabs", "transcribe")
class ElevenLabsScribe(ElevenLabsProvider):
    def transcribe(self, audio: Path, out: Path, hint: str | None = None) -> GenResult:
        form: dict[str, Any] = {
            "model_id": self.spec.model,
            "timestamps_granularity": "word",
            "tag_audio_events": "false",
            "diarize": "false",
        }
        language = self.params.get("language_code")
        if language:
            form["language_code"] = str(language)
        if hint and self.params.get("keyterms", True):
            terms = keyterms(hint, int(self.params.get("max_keyterms", 50)))
            if terms:
                form["keyterms"] = terms
        response = self.call(
            "POST",
            "/v1/speech-to-text",
            data=form,
            files={"file": (audio.name, read_bytes(audio), media_type(audio))},
        )
        transcript = payload(response, self.service)
        if not isinstance(transcript, dict):
            raise TransientError("elevenlabs returned an unexpected transcript payload")
        words = words_from_transcript(transcript)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(transcript, indent=2, ensure_ascii=False))
        seconds = transcript.get("audio_duration_secs") or media_duration(audio) or (words[-1].end if words else 0.0)
        return GenResult(
            path=out,
            data={"text": transcript.get("text", ""), "language_code": transcript.get("language_code")},
            duration=float(seconds),
            words=words,
            usage={"minute": float(seconds) / 60},
            meta=self.request_meta(response),
        )
