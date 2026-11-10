from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import httpx
import httpx2
import pytest
from PIL import Image

from mitosis.config import PACKAGE_ROOT, ModelSpec, Settings
from mitosis.models import Aspect, Voice
from mitosis.providers import anthropic_llm, comfyui, elevenlabs, fal, higgsfield, runway
from mitosis.providers.base import FACTORIES, KIND_METHODS, GenResult, PermanentError, TransientError

Handler = Callable[[httpx.Request], httpx.Response]

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


def reply(status: int = 200, **kwargs: Any) -> Handler:
    return lambda request: httpx.Response(status, **kwargs)


class Router:
    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], list[Handler]] = {}
        self.requests: list[httpx.Request] = []

    def add(self, method: str, url: str, *handlers: Handler) -> None:
        self.routes[(method, url)] = list(handlers)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        key = (request.method, str(request.url.copy_with(query=None)))
        handlers = self.routes.get(key)
        if not handlers:
            return httpx.Response(404, json={"detail": f"no route for {key}"})
        handler = handlers.pop(0) if len(handlers) > 1 else handlers[0]
        return handler(request)

    def sent(self, method: str, url: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == method and str(r.url.copy_with(query=None)) == url]


def build(cls: type, settings: Settings, router: Router, **spec: Any) -> Any:
    params = {"poll_interval": 0.0, **spec.pop("params", {})}
    provider = cls(ModelSpec(id="test", params=params, **spec), settings)
    provider.transport = httpx.MockTransport(router)
    return provider


def png_bytes(size: tuple[int, int] = (64, 64), mode: str = "RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, "white").save(buffer, format="PNG")
    return buffer.getvalue()


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        root=PACKAGE_ROOT,
        config_dir=PACKAGE_ROOT / "config",
        runs_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        models_dir=tmp_path / "models",
        priors_path=tmp_path / "priors.json",
    )


@pytest.fixture
def reference(tmp_path: Path) -> Path:
    path = tmp_path / "hero_0.png"
    path.write_bytes(png_bytes())
    return path


@pytest.fixture(scope="module")
def media(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    folder = tmp_path_factory.mktemp("media")
    clip = folder / "clip.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=black:s=64x112:d=1", "-c:v", "mpeg4", str(clip))
    tone = folder / "tone.mp3"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=0.8", "-c:a", "libmp3lame", str(tone))
    speech = folder / "speech.wav"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=220:duration=1.5", "-ar", "44100", str(speech))
    return {"clip": clip, "tone": tone, "speech": speech}


@pytest.fixture
def voice() -> Voice:
    return Voice(id="warm", label="Warm", elevenlabs_id="voice123", stability=0.4, similarity=0.8, style=0.3, speed=1.05)


def test_every_live_model_has_a_registered_adapter(settings: Settings) -> None:
    expected = {
        ("anthropic", "llm"),
        ("fal", "image"),
        ("fal", "video"),
        ("fal", "lipsync"),
        ("fal", "upscale"),
        ("elevenlabs", "speech"),
        ("elevenlabs", "voice"),
        ("elevenlabs", "transcribe"),
        ("runway", "video"),
        ("higgsfield", "video"),
        ("comfyui", "image"),
    }
    assert expected <= set(FACTORIES)
    for spec in settings.catalog.models.values():
        if spec.offline:
            continue
        factory = FACTORIES[(spec.provider, spec.kind)]
        assert callable(getattr(factory, KIND_METHODS[spec.kind]))


def test_offline_models_stay_last_in_every_chain(settings: Settings) -> None:
    for names in settings.catalog.chains.values():
        flags = [settings.catalog.models[name].offline for name in names]
        assert flags == sorted(flags)


def test_live_prices_use_the_units_adapters_report(settings: Settings) -> None:
    units = {
        "llm": {"input_mtok", "output_mtok", "cache_write_mtok", "cache_read_mtok"},
        "image": {"image"},
        "video": {"second"},
        "speech": {"kchar"},
        "voice": {"minute"},
        "transcribe": {"minute"},
        "lipsync": {"second"},
        "upscale": {"second"},
    }
    for spec in settings.catalog.models.values():
        if not spec.offline:
            assert set(spec.price) <= units[spec.kind], spec.id
            assert spec.notes


class FakeStream:
    def __init__(self, message: Any) -> None:
        self.message = message

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_final_message(self) -> Any:
        return self.message


class FakeMessages:
    def __init__(self, message: Any = None, error: Exception | None = None) -> None:
        self.message = message
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(("create", kwargs))
        if self.error:
            raise self.error
        return self.message

    def stream(self, **kwargs: Any) -> FakeStream:
        self.calls.append(("stream", kwargs))
        if self.error:
            raise self.error
        return FakeStream(self.message)


class FakeAnthropic:
    def __init__(self, message: Any = None, error: Exception | None = None) -> None:
        self.messages = FakeMessages(message, error)
        self.beta = type("Beta", (), {"messages": FakeMessages(message, error)})()


def claude_message(text: str = '{"angle": "trust"}', stop_reason: str = "end_turn", stop_details: Any = None) -> Any:
    return type(
        "Message",
        (),
        {
            "content": [type("Thinking", (), {"type": "thinking"})(), type("Text", (), {"type": "text", "text": text})()],
            "stop_reason": stop_reason,
            "stop_details": stop_details,
            "model": "claude-opus-5-5",
            "usage": type("Usage", (), {"input_tokens": 1200, "output_tokens": 300, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0})(),
        },
    )()


def claude(settings: Settings, client: FakeAnthropic, **params: Any) -> anthropic_llm.AnthropicLLM:
    provider = anthropic_llm.AnthropicLLM(
        ModelSpec(id="claude", provider="anthropic", kind="llm", model="claude-opus-5-5", params={"effort": "medium", **params}),
        settings,
    )
    provider.client = client
    return provider


def status_error(cls: type, status: int) -> Exception:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("failure", response=httpx2.Response(status, request=request), body=None)


def test_claude_request_shape_and_usage(settings: Settings, tmp_path: Path) -> None:
    wide = tmp_path / "wide.png"
    wide.write_bytes(png_bytes((3000, 1000)))
    client = FakeAnthropic(claude_message())
    schema = {"type": "object", "properties": {"angle": {"type": "string"}, "beats": {"type": "array", "items": {"type": "object", "properties": {"t": {"type": "number"}}}}}}
    result = claude(settings, client).json("You are a strategist.", "Deconstruct this ad.", schema, images=[wide], max_tokens=4000)
    kind, request = client.beta.messages.calls[0]
    assert kind == "create" and not client.messages.calls
    assert request["model"] == "claude-opus-5-5"
    assert request["system"] == "You are a strategist."
    assert request["betas"] == ["server-side-fallback-2026-07-01"]
    assert request["fallbacks"] == "default"
    assert request["output_config"]["effort"] == "medium"
    sent_schema = request["output_config"]["format"]["schema"]
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert sent_schema["additionalProperties"] is False
    assert sent_schema["properties"]["beats"]["items"]["additionalProperties"] is False
    assert "additionalProperties" not in schema
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(request)
    image, text = request["messages"][0]["content"]
    assert text == {"type": "text", "text": "Deconstruct this ad."}
    assert image["source"]["media_type"] == "image/jpeg"
    decoded = Image.open(io.BytesIO(base64.b64decode(image["source"]["data"])))
    assert max(decoded.size) == 1568
    assert result.data == {"angle": "trust"}
    assert result.usage == {"input_mtok": 0.0012, "output_mtok": 0.0003}


def test_claude_keeps_small_images_and_skips_fallbacks_when_disabled(settings: Settings, reference: Path) -> None:
    client = FakeAnthropic(claude_message())
    claude(settings, client, fallbacks=False).json("s", "p", {"type": "object"}, images=[reference])
    kind, request = client.messages.calls[0]
    assert kind == "create" and not client.beta.messages.calls
    assert "betas" not in request and "fallbacks" not in request
    assert request["messages"][0]["content"][0]["source"]["media_type"] == "image/png"


def test_claude_streams_large_outputs(settings: Settings) -> None:
    client = FakeAnthropic(claude_message())
    claude(settings, client).json("s", "p", {"type": "object"}, max_tokens=32000)
    assert client.beta.messages.calls[0][0] == "stream"


@pytest.mark.parametrize(
    ("message", "error"),
    [
        (claude_message(stop_reason="refusal", stop_details=type("Details", (), {"category": "cyber", "explanation": "no"})()), PermanentError),
        (claude_message(stop_reason="max_tokens"), TransientError),
        (claude_message(text="not json"), TransientError),
    ],
)
def test_claude_stop_reasons(settings: Settings, message: Any, error: type[Exception]) -> None:
    with pytest.raises(error):
        claude(settings, FakeAnthropic(message)).json("s", "p", {"type": "object"})


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        (status_error(anthropic.RateLimitError, 429), TransientError),
        (status_error(anthropic.InternalServerError, 500), TransientError),
        (status_error(anthropic.OverloadedError, 529), TransientError),
        (anthropic.APITimeoutError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")), TransientError),
        (anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")), TransientError),
        (status_error(anthropic.BadRequestError, 400), PermanentError),
        (status_error(anthropic.AuthenticationError, 401), PermanentError),
        (status_error(anthropic.PermissionDeniedError, 403), PermanentError),
        (status_error(anthropic.NotFoundError, 404), PermanentError),
    ],
)
def test_claude_error_mapping(settings: Settings, failure: Exception, error: type[Exception]) -> None:
    with pytest.raises(error):
        claude(settings, FakeAnthropic(error=failure)).json("s", "p", {"type": "object"})


QUEUE = "https://queue.fal.run"
REQUESTS = f"{QUEUE}/fal-ai/app/requests/r1"


def fal_queue(router: Router, endpoint: str, result: dict[str, Any], *states: str) -> None:
    ticket = {"request_id": "r1", "status_url": f"{REQUESTS}/status", "response_url": REQUESTS, "cancel_url": f"{REQUESTS}/cancel"}
    router.add("POST", f"{QUEUE}/{endpoint}", reply(json=ticket))
    router.add("GET", f"{REQUESTS}/status", *[reply(json={"status": state}) for state in states])
    router.add("GET", REQUESTS, reply(json=result))


@pytest.fixture
def uploads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    uploaded: list[Path] = []

    def upload_file(path: Path) -> str:
        uploaded.append(Path(path))
        return f"https://v3.fal.media/files/{Path(path).name}"

    monkeypatch.setenv("FAL_KEY", "fal-test-key")
    monkeypatch.setattr(fal.fal_client, "upload_file", upload_file)
    return uploaded


def test_fal_image_polls_queue_and_downloads(settings: Settings, reference: Path, uploads: list[Path], tmp_path: Path) -> None:
    router = Router()
    endpoint = "fal-ai/nano-banana-pro/edit"
    fal_queue(router, endpoint, {"images": [{"url": "https://v3.fal.media/files/out.png"}], "seed": 42}, "IN_QUEUE", "IN_PROGRESS", "COMPLETED")
    router.add("GET", "https://v3.fal.media/files/out.png", reply(content=png_bytes()))
    second = tmp_path / "hero_1.png"
    second.write_bytes(png_bytes((32, 32)))
    provider = build(
        fal.FalImage,
        settings,
        router,
        provider="fal",
        kind="image",
        model=endpoint,
        params={"extra": {"num_images": 1, "output_format": "png"}},
    )
    out = tmp_path / "shots" / "key.png"
    result = provider.generate("same man, kitchen", [reference, second], Aspect.VERTICAL, 4242, out, negative="")
    provider.generate("same man, kitchen", [reference, second], Aspect.VERTICAL, 4243, tmp_path / "again.png")
    body = json.loads(router.sent("POST", f"{QUEUE}/{endpoint}")[0].content)
    assert body == {
        "num_images": 1,
        "output_format": "png",
        "prompt": "same man, kitchen",
        "image_urls": ["https://v3.fal.media/files/hero_0.png", "https://v3.fal.media/files/hero_1.png"],
        "aspect_ratio": "9:16",
        "seed": 4242,
    }
    assert len(router.sent("GET", f"{REQUESTS}/status")) == 3 + 1
    assert router.requests[0].headers["Authorization"] == "Key fal-test-key"
    assert "Authorization" not in router.sent("GET", "https://v3.fal.media/files/out.png")[0].headers
    assert uploads == [reference, second]
    assert out.read_bytes() == png_bytes()
    assert result.usage == {"image": 1}
    assert result.meta["seed"] == 42


def test_fal_image_fields_come_from_params(settings: Settings, reference: Path, uploads: list[Path], tmp_path: Path) -> None:
    router = Router()
    endpoint = "fal-ai/bytedance/seedream/v4.5/edit"
    fal_queue(router, endpoint, {"images": [{"url": "https://v3.fal.media/files/out.png"}]}, "COMPLETED")
    router.add("GET", "https://v3.fal.media/files/out.png", reply(content=png_bytes()))
    params = {
        "fields": {"aspect": "image_size"},
        "aspects": {"4:5": {"width": 1728, "height": 2160}},
        "max_references": 1,
    }
    provider = build(fal.FalImage, settings, router, provider="fal", kind="image", model=endpoint, params=params)
    provider.generate("p", [reference, reference], Aspect.PORTRAIT, 1, tmp_path / "k.png", negative="ugly")
    body = json.loads(router.sent("POST", f"{QUEUE}/{endpoint}")[0].content)
    assert body["image_size"] == {"width": 1728, "height": 2160}
    assert "aspect_ratio" not in body and "negative_prompt" not in body
    assert body["image_urls"] == ["https://v3.fal.media/files/hero_0.png"]


@needs_ffmpeg
@pytest.mark.parametrize(
    ("params", "duration_field", "duration_value"),
    [
        ({"fields": {"image": "start_image_url", "seed": None}, "extra": {"generate_audio": False}}, "start_image_url", "5"),
        ({"duration_template": "{seconds}s", "fields": {"aspect": "aspect_ratio"}, "extra": {"generate_audio": False}}, "image_url", "5s"),
    ],
)
def test_fal_video_duration_and_audio_toggle(
    settings: Settings,
    reference: Path,
    uploads: list[Path],
    media: dict[str, Path],
    tmp_path: Path,
    params: dict[str, Any],
    duration_field: str,
    duration_value: str,
) -> None:
    router = Router()
    endpoint = "fal-ai/video/image-to-video"
    fal_queue(router, endpoint, {"video": {"url": "https://v3.fal.media/files/clip.mp4"}}, "IN_PROGRESS", "COMPLETED")
    router.add("GET", "https://v3.fal.media/files/clip.mp4", reply(content=media["clip"].read_bytes()))
    provider = build(fal.FalVideo, settings, router, provider="fal", kind="video", model=endpoint, durations=[5, 10], params=params)
    result = provider.animate(reference, "slow push in", 5.0, Aspect.VERTICAL, 9, tmp_path / "clip.mp4")
    body = json.loads(router.sent("POST", f"{QUEUE}/{endpoint}")[0].content)
    assert body["duration"] == duration_value
    assert body[duration_field] == "https://v3.fal.media/files/hero_0.png"
    assert body["generate_audio"] is False
    assert result.usage == {"second": 5.0}
    assert result.duration == pytest.approx(1.0, abs=0.1)


def test_fal_content_policy_failure_is_permanent(settings: Settings, reference: Path, uploads: list[Path], tmp_path: Path) -> None:
    router = Router()
    endpoint = "fal-ai/nano-banana-pro/edit"
    fal_queue(router, endpoint, {}, "COMPLETED")
    router.add("GET", f"{REQUESTS}/status", reply(json={"status": "COMPLETED", "error": "flagged", "error_type": "content_policy_violation"}))
    router.add("GET", REQUESTS, reply(422, json={"detail": [{"loc": ["body", "prompt"], "msg": "flagged", "type": "content_policy_violation"}]}))
    provider = build(fal.FalImage, settings, router, provider="fal", kind="image", model=endpoint)
    with pytest.raises(PermanentError, match="content_policy_violation"):
        provider.generate("p", [reference], Aspect.VERTICAL, 1, tmp_path / "k.png")


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (reply(503, json={"detail": "busy"}), TransientError),
        (reply(429, json={"detail": "slow down"}), TransientError),
        (reply(400, json={"detail": "runner died"}, headers={"X-Fal-Needs-Retry": "1"}), TransientError),
        (reply(422, json={"detail": [{"msg": "bad", "type": "value_error"}]}), PermanentError),
        (reply(401, json={"detail": "bad key"}), PermanentError),
    ],
)
def test_fal_submit_errors(settings: Settings, reference: Path, uploads: list[Path], tmp_path: Path, response: Handler, error: type[Exception]) -> None:
    router = Router()
    router.add("POST", f"{QUEUE}/fal-ai/nano-banana-pro/edit", response)
    provider = build(fal.FalImage, settings, router, provider="fal", kind="image", model="fal-ai/nano-banana-pro/edit")
    with pytest.raises(error):
        provider.generate("p", [reference], Aspect.VERTICAL, 1, tmp_path / "k.png")


def test_fal_queue_timeout_cancels_request(settings: Settings, reference: Path, uploads: list[Path], tmp_path: Path) -> None:
    router = Router()
    endpoint = "fal-ai/nano-banana-pro/edit"
    fal_queue(router, endpoint, {}, "IN_QUEUE")
    router.add("PUT", f"{REQUESTS}/cancel", reply(202, json={"status": "CANCELLATION_REQUESTED"}))
    provider = build(fal.FalImage, settings, router, provider="fal", kind="image", model=endpoint, timeout=0.05, params={"poll_interval": 0.01})
    with pytest.raises(TransientError, match="unfinished"):
        provider.generate("p", [reference], Aspect.VERTICAL, 1, tmp_path / "k.png")
    assert router.sent("PUT", f"{REQUESTS}/cancel")


@needs_ffmpeg
def test_fal_lipsync_and_upscale_bill_seconds(settings: Settings, uploads: list[Path], media: dict[str, Path], tmp_path: Path) -> None:
    router = Router()
    fal_queue(router, "fal-ai/sync-lipsync/v2", {"video": {"url": "https://v3.fal.media/files/synced.mp4"}}, "COMPLETED")
    router.add("GET", "https://v3.fal.media/files/synced.mp4", reply(content=media["clip"].read_bytes()))
    lipsync = build(fal.FalLipSync, settings, router, provider="fal", kind="lipsync", model="fal-ai/sync-lipsync/v2", params={"extra": {"sync_mode": "cut_off"}})
    synced = lipsync.sync(media["clip"], media["tone"], tmp_path / "synced.mp4")
    body = json.loads(router.sent("POST", f"{QUEUE}/fal-ai/sync-lipsync/v2")[0].content)
    assert body == {"sync_mode": "cut_off", "video_url": "https://v3.fal.media/files/clip.mp4", "audio_url": "https://v3.fal.media/files/tone.mp3"}
    assert synced.usage["second"] == pytest.approx(1.0, abs=0.1)
    router = Router()
    fal_queue(router, "fal-ai/topaz/upscale/video", {"video": {"url": "https://v3.fal.media/files/up.mp4"}}, "COMPLETED")
    router.add("GET", "https://v3.fal.media/files/up.mp4", reply(content=media["clip"].read_bytes()))
    upscaler = build(fal.FalUpscale, settings, router, provider="fal", kind="upscale", model="fal-ai/topaz/upscale/video")
    upscaled = upscaler.upscale(media["clip"], tmp_path / "up.mp4", factor=2)
    assert json.loads(router.sent("POST", f"{QUEUE}/fal-ai/topaz/upscale/video")[0].content)["upscale_factor"] == 2
    assert upscaled.usage["second"] == pytest.approx(1.0, abs=0.1)


def alignment(text: str, step: float = 0.1) -> dict[str, list[Any]]:
    return {
        "characters": list(text),
        "character_start_times_seconds": [round(i * step, 3) for i in range(len(text))],
        "character_end_times_seconds": [round((i + 1) * step, 3) for i in range(len(text))],
    }


def test_elevenlabs_alignment_becomes_words_with_punctuation() -> None:
    words = elevenlabs.words_from_alignment(alignment("Hey, you - listen  up."))
    assert [w.text for w in words] == ["Hey,", "you-", "listen", "up."]
    assert (words[0].start, words[0].end) == (0.0, 0.4)
    assert (words[1].start, words[1].end) == (0.5, 1.0)
    assert (words[3].start, words[3].end) == (1.9, 2.2)
    assert elevenlabs.words_from_alignment(None) == []


@pytest.fixture
def eleven_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "xi-test")


@needs_ffmpeg
def test_elevenlabs_tts_writes_48k_wav_with_words(settings: Settings, eleven_key: None, voice: Voice, media: dict[str, Path], tmp_path: Path) -> None:
    router = Router()
    text = "Stop scrolling, honestly."
    url = "https://api.elevenlabs.io/v1/text-to-speech/voice123/with-timestamps"
    audio = base64.b64encode(media["tone"].read_bytes()).decode()
    router.add("POST", url, reply(json={"audio_base64": audio, "alignment": alignment(text, 0.03)}, headers={"request-id": "req-1"}))
    provider = build(elevenlabs.ElevenLabsSpeech, settings, router, provider="elevenlabs", kind="speech", model="eleven_multilingual_v2")
    out = tmp_path / "audio" / "hook.wav"
    result = provider.speak(text, voice, out)
    request = router.sent("POST", url)[0]
    body = json.loads(request.content)
    assert request.headers["xi-api-key"] == "xi-test"
    assert request.url.params["output_format"] == "mp3_44100_128"
    assert body["model_id"] == "eleven_multilingual_v2"
    assert body["voice_settings"] == {"stability": 0.4, "similarity_boost": 0.8, "style": 0.3, "use_speaker_boost": True, "speed": 1.05}
    with wave.open(str(out)) as handle:
        assert handle.getframerate() == 48000
    assert [w.text for w in result.words] == ["Stop", "scrolling,", "honestly."]
    assert result.usage == {"kchar": len(text) / 1000}
    assert result.duration == pytest.approx(0.8, abs=0.1)
    assert result.meta["request_id"] == "req-1"


@needs_ffmpeg
def test_elevenlabs_tts_accepts_raw_pcm(settings: Settings, eleven_key: None, voice: Voice, tmp_path: Path) -> None:
    router = Router()
    url = "https://api.elevenlabs.io/v1/text-to-speech/voice123/with-timestamps"
    pcm = b"\x00\x00" * 24000
    router.add("POST", url, reply(json={"audio_base64": base64.b64encode(pcm).decode(), "alignment": alignment("Hi")}))
    provider = build(elevenlabs.ElevenLabsSpeech, settings, router, provider="elevenlabs", kind="speech", model="m", params={"output_format": "pcm_24000"})
    result = provider.speak("Hi", voice, tmp_path / "hi.wav")
    assert result.duration == pytest.approx(1.0, abs=0.05)


@needs_ffmpeg
def test_elevenlabs_voice_changer_keeps_timing_inputs(settings: Settings, eleven_key: None, voice: Voice, media: dict[str, Path], tmp_path: Path) -> None:
    router = Router()
    url = "https://api.elevenlabs.io/v1/voice-changer/voice123"
    router.add("POST", url, reply(content=media["tone"].read_bytes()))
    provider = build(
        elevenlabs.ElevenLabsVoiceChanger,
        settings,
        router,
        provider="elevenlabs",
        kind="voice",
        model="eleven_multilingual_sts_v2",
        params={"remove_background_noise": True},
    )
    result = provider.convert(media["speech"], voice, tmp_path / "body.wav")
    content = router.sent("POST", url)[0].content
    assert b'name="model_id"\r\n\r\neleven_multilingual_sts_v2' in content
    assert b'name="remove_background_noise"\r\n\r\ntrue' in content
    assert b'"similarity_boost": 0.8' in content
    assert b'name="audio"; filename="speech.wav"' in content
    assert result.usage["minute"] == pytest.approx(1.5 / 60, abs=0.002)
    assert result.path == tmp_path / "body.wav"


def test_elevenlabs_scribe_returns_words_and_minutes(settings: Settings, eleven_key: None, reference: Path, tmp_path: Path) -> None:
    router = Router()
    transcript = {
        "language_code": "en",
        "text": "Most couples never talk.",
        "audio_duration_secs": 30.0,
        "words": [
            {"text": "Most", "start": 0.1, "end": 0.4, "type": "word"},
            {"text": " ", "start": 0.4, "end": 0.45, "type": "spacing"},
            {"text": "(laughs)", "start": 0.45, "end": 0.9, "type": "audio_event"},
            {"text": "couples", "start": 0.9, "end": 1.3, "type": "word"},
        ],
    }
    router.add("POST", "https://api.elevenlabs.io/v1/speech-to-text", reply(json=transcript))
    provider = build(elevenlabs.ElevenLabsScribe, settings, router, provider="elevenlabs", kind="transcribe", model="scribe_v2", params={"language_code": "en"})
    out = tmp_path / "transcript.json"
    result = provider.transcribe(reference, out, hint="Betterly helps couples reconnect tonight")
    content = router.requests[0].content
    assert b'name="timestamps_granularity"\r\n\r\nword' in content
    assert b'name="keyterms"\r\n\r\nBetterly' in content
    assert b'name="keyterms"\r\n\r\nreconnect' in content
    assert [(w.text, w.start, w.end) for w in result.words] == [("Most", 0.1, 0.4), ("couples", 0.9, 1.3)]
    assert result.usage == {"minute": 0.5}
    assert json.loads(out.read_text())["text"] == "Most couples never talk."


@pytest.mark.parametrize(("status", "error"), [(429, TransientError), (500, TransientError), (401, PermanentError), (402, PermanentError), (422, PermanentError)])
def test_elevenlabs_error_mapping(settings: Settings, eleven_key: None, voice: Voice, tmp_path: Path, status: int, error: type[Exception]) -> None:
    router = Router()
    router.add("POST", "https://api.elevenlabs.io/v1/text-to-speech/voice123/with-timestamps", reply(status, json={"detail": {"code": "x", "message": "y"}}))
    provider = build(elevenlabs.ElevenLabsSpeech, settings, router, provider="elevenlabs", kind="speech", model="m")
    with pytest.raises(error):
        provider.speak("Hello", voice, tmp_path / "a.wav")


def test_elevenlabs_requires_a_voice_id(settings: Settings, eleven_key: None, tmp_path: Path) -> None:
    provider = build(elevenlabs.ElevenLabsSpeech, settings, Router(), provider="elevenlabs", kind="speech", model="m")
    with pytest.raises(PermanentError, match="elevenlabs_id"):
        provider.speak("Hello", Voice(id="nobody", label="x"), tmp_path / "a.wav")


RUNWAY = "https://api.dev.runwayml.com"


@pytest.fixture
def runway_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNWAYML_API_SECRET", "rw-secret")


def test_runway_creates_task_and_polls(settings: Settings, runway_key: None, reference: Path, tmp_path: Path) -> None:
    router = Router()
    router.add("POST", f"{RUNWAY}/v1/image_to_video", reply(json={"id": "task1"}))
    router.add(
        "GET",
        f"{RUNWAY}/v1/tasks/task1",
        reply(json={"id": "task1", "status": "PENDING"}),
        reply(json={"id": "task1", "status": "THROTTLED"}),
        reply(json={"id": "task1", "status": "RUNNING"}),
        reply(json={"id": "task1", "status": "SUCCEEDED", "output": ["https://dnznrvs05pmza.cloudfront.net/clip.mp4?sig=1"]}),
    )
    router.add("GET", "https://dnznrvs05pmza.cloudfront.net/clip.mp4", reply(content=b"mp4-bytes"))
    provider = build(runway.RunwayVideo, settings, router, provider="runway", kind="video", model="gen4_turbo", durations=[5, 10])
    result = provider.animate(reference, "push in", 5.0, Aspect.VERTICAL, 5_000_000_000, tmp_path / "clip.mp4")
    request = router.sent("POST", f"{RUNWAY}/v1/image_to_video")[0]
    body = json.loads(request.content)
    assert request.headers["Authorization"] == "Bearer rw-secret"
    assert request.headers["X-Runway-Version"] == "2024-11-06"
    assert body["model"] == "gen4_turbo"
    assert body["promptImage"][0]["position"] == "first"
    assert body["promptImage"][0]["uri"].startswith("data:image/png;base64,")
    assert body["ratio"] == "720:1280"
    assert body["duration"] == 5
    assert body["seed"] == 5_000_000_000 % 4_294_967_295
    assert body["promptText"] == "push in"
    download = router.sent("GET", "https://dnznrvs05pmza.cloudfront.net/clip.mp4")[0]
    assert "Authorization" not in download.headers
    assert (tmp_path / "clip.mp4").read_bytes() == b"mp4-bytes"
    assert result.usage == {"second": 5.0}


@pytest.mark.parametrize(("code", "error"), [("SAFETY.INPUT.TEXT", PermanentError), ("ASSET.INVALID", PermanentError), ("INTERNAL.BAD_OUTPUT.CODE01", TransientError), (None, TransientError)])
def test_runway_failure_codes(settings: Settings, runway_key: None, reference: Path, tmp_path: Path, code: str | None, error: type[Exception]) -> None:
    router = Router()
    router.add("POST", f"{RUNWAY}/v1/image_to_video", reply(json={"id": "t"}))
    router.add("GET", f"{RUNWAY}/v1/tasks/t", reply(json={"id": "t", "status": "FAILED", "failure": "nope", "failureCode": code}))
    provider = build(runway.RunwayVideo, settings, router, provider="runway", kind="video", model="gen4_turbo")
    with pytest.raises(error):
        provider.animate(reference, "p", 5.0, Aspect.VERTICAL, 1, tmp_path / "clip.mp4")


HIGGS = "https://api.higgsfield.ai"


@pytest.fixture
def higgs_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIGGSFIELD_API_KEY", "kid")
    monkeypatch.setenv("HIGGSFIELD_API_SECRET", "ksecret")


def higgs_routes(router: Router, *statuses: dict[str, Any]) -> None:
    slot = {
        "upload_url": "https://storage.higgsfield.ai/put/abc?X-Amz-Signature=1",
        "public_url": "https://cdn.higgsfield.ai/abc.png",
        "upload_headers": {"Content-Type": "image/png", "x-amz-tagging": "retention=temporary"},
    }
    router.add("POST", f"{HIGGS}/files/generate-upload-url", reply(json=slot))
    router.add("PUT", "https://storage.higgsfield.ai/put/abc", reply(200))
    ticket = {"status": "queued", "request_id": "h1", "status_url": f"{HIGGS}/requests/h1/status", "cancel_url": f"{HIGGS}/requests/h1/cancel"}
    router.add("POST", f"{HIGGS}/kling-video/v2.6/pro/image-to-video", reply(json=ticket))
    router.add("GET", f"{HIGGS}/requests/h1/status", *[reply(json=status) for status in statuses])
    router.add("GET", "https://cdn.higgsfield.ai/h1.mp4", reply(content=b"video"))


def test_higgsfield_uploads_submits_and_polls(settings: Settings, higgs_keys: None, reference: Path, tmp_path: Path) -> None:
    router = Router()
    higgs_routes(router, {"status": "queued"}, {"status": "in_progress"}, {"status": "completed", "video": {"url": "https://cdn.higgsfield.ai/h1.mp4"}})
    params = {
        "duration_numeric": True,
        "fields": {"aspect": "aspect_ratio"},
        "aspects": {"9:16": "9:16"},
        "extra": {"sound": "off", "cfg_scale": 0.5},
    }
    provider = build(higgsfield.HiggsfieldVideo, settings, router, provider="higgsfield", kind="video", model="kling-video/v2.6/pro/image-to-video", params=params)
    result = provider.animate(reference, "push in", 5.0, Aspect.VERTICAL, 3, tmp_path / "clip.mp4")
    put = router.sent("PUT", "https://storage.higgsfield.ai/put/abc")[0]
    assert "Authorization" not in put.headers
    assert put.headers["x-amz-tagging"] == "retention=temporary"
    submit = router.sent("POST", f"{HIGGS}/kling-video/v2.6/pro/image-to-video")[0]
    assert submit.headers["Authorization"] == "Key kid:ksecret"
    assert submit.headers["Idempotency-Key"]
    assert json.loads(submit.content) == {
        "sound": "off",
        "cfg_scale": 0.5,
        "prompt": "push in",
        "image_url": "https://cdn.higgsfield.ai/abc.png",
        "duration": 5,
        "aspect_ratio": "9:16",
    }
    assert (tmp_path / "clip.mp4").read_bytes() == b"video"
    assert result.usage == {"second": 5.0}


@pytest.mark.parametrize(("status", "error"), [("nsfw", PermanentError), ("canceled", PermanentError), ("failed", TransientError)])
def test_higgsfield_terminal_failures(settings: Settings, higgs_keys: None, reference: Path, tmp_path: Path, status: str, error: type[Exception]) -> None:
    router = Router()
    higgs_routes(router, {"status": status, "error": "x"})
    provider = build(higgsfield.HiggsfieldVideo, settings, router, provider="higgsfield", kind="video", model="kling-video/v2.6/pro/image-to-video")
    with pytest.raises(error):
        provider.animate(reference, "p", 5.0, Aspect.VERTICAL, 1, tmp_path / "clip.mp4")


COMFY = "http://127.0.0.1:8188"


def comfy_routes(router: Router, history: dict[str, Any]) -> None:
    router.add("POST", f"{COMFY}/upload/image", reply(json={"name": "mitosis_ref.png", "subfolder": "", "type": "input"}))
    router.add("POST", f"{COMFY}/prompt", reply(json={"prompt_id": "p1", "number": 3, "node_errors": {}}))
    router.add("GET", f"{COMFY}/history/p1", reply(json={}), reply(json={"p1": history}))
    router.add("GET", f"{COMFY}/view", reply(content=png_bytes((16, 16))))


def comfy(settings: Settings, router: Router, **params: Any) -> comfyui.ComfyUIImage:
    spec = settings.catalog.models["comfyui_flux_pulid"]
    return build(
        comfyui.ComfyUIImage,
        settings,
        router,
        provider="comfyui",
        kind="image",
        model=spec.model,
        params={**spec.params, "poll_interval": 0.0, **params},
    )


def test_comfyui_workflow_matches_default_bindings(settings: Settings) -> None:
    spec = settings.catalog.models["comfyui_flux_pulid"]
    graph = comfyui.load_graph(settings.root / spec.model)
    for binding in spec.params["bindings"].values():
        for node_id, section, field in comfyui.binding_paths(binding):
            assert field in graph[node_id][section]
    assert graph[spec.params["output"]]["class_type"] == "SaveImage"
    classes = {node["class_type"] for node in graph.values()}
    assert {"UNETLoader", "ApplyPulidFlux", "LoraLoaderModelOnly", "KSampler", "VAEDecode", "SaveImage"} <= classes


def test_comfyui_patches_graph_and_bypasses_missing_lora(settings: Settings, monkeypatch: pytest.MonkeyPatch, reference: Path, tmp_path: Path) -> None:
    monkeypatch.setenv("COMFYUI_URL", f"{COMFY}/")
    router = Router()
    history = {"status": {"status_str": "success", "completed": True}, "outputs": {"9": {"images": [{"filename": "k_00001_.png", "subfolder": "mitosis", "type": "output"}]}}}
    comfy_routes(router, history)
    provider = comfy(settings, router)
    out = tmp_path / "key.png"
    result = provider.generate("the same man in a kitchen", [reference], Aspect.VERTICAL, 77, out, negative="blurry")
    graph = json.loads(router.sent("POST", f"{COMFY}/prompt")[0].content)["prompt"]
    assert graph["6"]["inputs"]["text"] == "the same man in a kitchen"
    assert graph["7"]["inputs"]["text"] == "blurry"
    assert graph["17"]["inputs"]["seed"] == 77
    assert graph["10"]["inputs"]["image"] == "mitosis_ref.png"
    assert graph["13"]["inputs"]["weight"] == 0.9
    assert "4" not in graph and graph["13"]["inputs"]["model"] == ["1", 0]
    width, height = comfyui.fit(Aspect.VERTICAL, 1.0, 16)
    assert (graph["16"]["inputs"]["width"], graph["16"]["inputs"]["height"]) == (width, height)
    assert (graph["14"]["inputs"]["width"], graph["14"]["inputs"]["height"]) == (width, height)
    assert width % 16 == 0 and height % 16 == 0 and height > width
    view = router.sent("GET", f"{COMFY}/view")[0]
    assert dict(view.url.params) == {"filename": "k_00001_.png", "subfolder": "mitosis", "type": "output"}
    assert out.read_bytes() == png_bytes((16, 16))
    assert result.usage == {"image": 1}


def test_comfyui_applies_lora_when_configured(settings: Settings, monkeypatch: pytest.MonkeyPatch, reference: Path, tmp_path: Path) -> None:
    monkeypatch.setenv("COMFYUI_URL", COMFY)
    router = Router()
    comfy_routes(router, {"status": {"status_str": "success", "completed": True}, "outputs": {"9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}}})
    provider = comfy(settings, router, values={"lora": "betterly_hero.safetensors", "lora_strength": 0.7, "identity_weight": 0.8})
    provider.generate("p", [reference], Aspect.SQUARE, 1, tmp_path / "k.png")
    graph = json.loads(router.sent("POST", f"{COMFY}/prompt")[0].content)["prompt"]
    assert graph["4"]["inputs"]["lora_name"] == "betterly_hero.safetensors"
    assert graph["4"]["inputs"]["strength_model"] == 0.7
    assert graph["13"]["inputs"]["model"] == ["4", 0]
    assert graph["16"]["inputs"]["width"] == graph["16"]["inputs"]["height"]


@pytest.mark.parametrize(("message", "error"), [("CUDA out of memory", TransientError), ("Missing face in reference", PermanentError)])
def test_comfyui_execution_errors(settings: Settings, monkeypatch: pytest.MonkeyPatch, reference: Path, tmp_path: Path, message: str, error: type[Exception]) -> None:
    monkeypatch.setenv("COMFYUI_URL", COMFY)
    router = Router()
    failure = {"node_id": "13", "node_type": "ApplyPulidFlux", "exception_message": message}
    comfy_routes(router, {"status": {"status_str": "error", "completed": False, "messages": [["execution_error", failure]]}, "outputs": {}})
    with pytest.raises(error, match="ApplyPulidFlux"):
        comfy(settings, router).generate("p", [reference], Aspect.VERTICAL, 1, tmp_path / "k.png")


def test_comfyui_rejected_workflow_is_permanent(settings: Settings, monkeypatch: pytest.MonkeyPatch, reference: Path, tmp_path: Path) -> None:
    monkeypatch.setenv("COMFYUI_URL", COMFY)
    router = Router()
    comfy_routes(router, {})
    router.add("POST", f"{COMFY}/prompt", reply(400, json={"error": {"type": "prompt_outputs_failed_validation"}, "node_errors": {"5": {}}}))
    with pytest.raises(PermanentError):
        comfy(settings, router).generate("p", [reference], Aspect.VERTICAL, 1, tmp_path / "k.png")


def test_missing_credentials_are_permanent(settings: Settings, monkeypatch: pytest.MonkeyPatch, reference: Path, tmp_path: Path) -> None:
    monkeypatch.delenv("RUNWAYML_API_SECRET", raising=False)
    provider = build(runway.RunwayVideo, settings, Router(), provider="runway", kind="video", model="gen4_turbo")
    with pytest.raises(PermanentError, match="RUNWAYML_API_SECRET"):
        provider.animate(reference, "p", 5.0, Aspect.VERTICAL, 1, tmp_path / "clip.mp4")


def test_results_are_gen_results(settings: Settings, reference: Path) -> None:
    client = FakeAnthropic(claude_message())
    assert isinstance(claude(settings, client).json("s", "p", {"type": "object"}, images=[reference]), GenResult)


def test_claude_request_through_real_sdk(settings: Settings) -> None:
    captured: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append(request)
        body = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": '{"hook": "Stop scrolling"}'}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 2000, "output_tokens": 100},
        }
        return httpx2.Response(200, json=body, headers={"request-id": "req_123"})

    client = anthropic.Anthropic(api_key="test-key", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    result = claude(settings, client).json("system", "prompt", {"type": "object", "properties": {"hook": {"type": "string"}}})
    request = captured[0]
    sent = json.loads(request.content)
    assert request.url.path == "/v1/messages"
    assert "server-side-fallback-2026-07-01" in request.headers["anthropic-beta"]
    assert sent["fallbacks"] == "default"
    assert sent["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert "thinking" not in sent and "temperature" not in sent
    assert result.data == {"hook": "Stop scrolling"}
    assert result.usage == {"input_mtok": 0.002, "output_mtok": 0.0001}
    assert result.meta["request_id"] == "req_123"
