from __future__ import annotations

import json
import logging
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from mitosis.media import ff, probe
from mitosis.media.align import align_script
from mitosis.models import Aspect, Voice
from mitosis.providers.base import GenResult, PermanentError, Provider, ProviderError, register

log = logging.getLogger(__name__)

VOICE_RATE = 48000
CLIP_FPS = 30
BASE_WORDS_PER_MINUTE = 180
ZOOMPAN_OVERSAMPLE = 2
SETTING_PATTERN = re.compile(r"setting:\s*(.+?)(?:\.\s|\.$|$)", re.IGNORECASE | re.DOTALL)
VOICE_LISTING = re.compile(r"^(.+?)\s+[a-z]{2,3}_[A-Za-z0-9]{2,4}\s+#")


@dataclass(frozen=True)
class Grade:
    gains: tuple[float, float, float] = (1.0, 1.0, 1.0)
    gamma: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    shadows: tuple[float, float, float] = (0.0, 0.0, 0.0)
    highlights: tuple[float, float, float] = (0.0, 0.0, 0.0)
    vignette: float = 0.25
    gradient: tuple[tuple[int, int, int], tuple[int, int, int]] = ((70, 70, 74), (24, 24, 28))


GRADES: dict[str, Grade] = {
    "neutral": Grade(contrast=1.03, saturation=1.02, vignette=0.22),
    "warm": Grade(
        gains=(1.07, 1.01, 0.88),
        contrast=1.04,
        saturation=1.06,
        highlights=(0.03, 0.01, -0.02),
        vignette=0.28,
        gradient=((150, 98, 60), (48, 28, 20)),
    ),
    "cool": Grade(
        gains=(0.93, 1.0, 1.08),
        gamma=0.95,
        contrast=1.02,
        saturation=0.95,
        shadows=(-0.01, 0.0, 0.02),
        vignette=0.18,
        gradient=((188, 204, 220), (92, 108, 128)),
    ),
    "teal_orange": Grade(
        contrast=1.1,
        saturation=1.12,
        shadows=(-0.05, 0.02, 0.05),
        highlights=(0.06, 0.02, -0.05),
        vignette=0.3,
        gradient=((214, 148, 96), (20, 64, 72)),
    ),
    "golden": Grade(
        gains=(1.12, 1.03, 0.8),
        gamma=0.95,
        contrast=1.05,
        saturation=1.1,
        highlights=(0.05, 0.03, -0.03),
        vignette=0.3,
        gradient=((240, 176, 92), (96, 56, 36)),
    ),
    "moody": Grade(
        gains=(0.95, 0.97, 1.03),
        gamma=1.18,
        contrast=1.16,
        saturation=0.82,
        shadows=(0.0, 0.01, 0.03),
        vignette=0.42,
        gradient=((44, 52, 72), (8, 10, 16)),
    ),
}

GRADE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "golden": ("golden", "sunset", "sunrise", "rim light", "magic hour"),
    "teal_orange": ("car", "windshield", "dashboard", "street", "neon", "city lights", "cinematic"),
    "moody": ("night", "moody", "dim", "glow", "dark", "shadow", "candle"),
    "warm": ("warm", "tungsten", "lamp", "evening", "cozy", "fireplace", "couch", "bedroom"),
    "cool": ("cool", "morning", "daylight", "overcast", "bright", "kitchen", "office", "window"),
}

MOTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "push_in": ("push in", "push-in", "dolly in", "zoom in", "move in", "closer"),
    "pull_out": ("pull out", "pull back", "pull-back", "dolly out", "zoom out", "reveal"),
    "drift_left": ("pan left", "drift left", "slide left", "truck left"),
    "drift_right": ("pan right", "drift right", "slide right", "truck right"),
}
MOTIONS = tuple(MOTION_KEYWORDS)


def setting_text(prompt: str) -> str:
    match = SETTING_PATTERN.search(prompt)
    return (match.group(1) if match else prompt).lower()


def mentions(text: str, keyword: str) -> bool:
    return re.search(rf"\b{re.escape(keyword)}s?\b", text) is not None


def grade_for(prompt: str) -> str:
    text = setting_text(prompt)
    scores = {name: sum(mentions(text, word) for word in words) for name, words in GRADE_KEYWORDS.items()}
    best = max(scores, key=lambda name: scores[name])
    return best if scores[best] > 0 else "neutral"


def motion_for(prompt: str, seed: int) -> str:
    text = prompt.lower()
    for motion, phrases in MOTION_KEYWORDS.items():
        if any(phrase in text for phrase in phrases):
            return motion
    return MOTIONS[int(np.random.default_rng(seed).integers(len(MOTIONS)))]


def luminance(pixels: np.ndarray) -> np.ndarray:
    return pixels[..., 0] * 0.2126 + pixels[..., 1] * 0.7152 + pixels[..., 2] * 0.0722


def apply_grade(pixels: np.ndarray, grade: Grade, rng: np.random.Generator) -> np.ndarray:
    image = np.clip(pixels * np.asarray(grade.gains, dtype=np.float32), 0.0, None) ** grade.gamma
    light = luminance(image)[..., None]
    image = image + np.asarray(grade.shadows, dtype=np.float32) * (1 - light) ** 2
    image = image + np.asarray(grade.highlights, dtype=np.float32) * light**2
    image = (image - 0.5) * grade.contrast + 0.5
    light = luminance(image)[..., None]
    image = light + (image - light) * grade.saturation
    height, width = image.shape[:2]
    rows = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    cols = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    radius = np.sqrt(rows**2 + cols**2) / math.sqrt(2)
    falloff = np.clip((radius - 0.35) / 0.65, 0.0, 1.0)
    image = image * (1 - grade.vignette * falloff * falloff * (3 - 2 * falloff))[..., None]
    grain = rng.normal(0.0, 0.012, size=(height, width, 1)).astype(np.float32)
    return np.clip(image + grain, 0.0, 1.0)


def load_reference(references: list[Path]) -> Image.Image | None:
    for reference in references:
        try:
            with Image.open(reference) as handle:
                return ImageOps.exif_transpose(handle).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            log.warning("unusable reference image %s: %s", reference, exc)
    return None


def reframe_reference(reference: Image.Image, size: tuple[int, int], rng: np.random.Generator) -> Image.Image:
    target_width, target_height = size
    width, height = reference.size
    ratio = target_width / target_height
    crop_width, crop_height = (height * ratio, height) if width / height > ratio else (width, width / ratio)
    zoom = float(rng.uniform(1.04, 1.2))
    crop_width, crop_height = crop_width / zoom, crop_height / zoom
    left = (width - crop_width) * float(rng.uniform(0.3, 0.7))
    top = (height - crop_height) * float(rng.uniform(0.15, 0.55))
    framed = reference.resize(size, Image.Resampling.LANCZOS, box=(left, top, left + crop_width, top + crop_height))
    return ImageOps.mirror(framed) if rng.random() < 0.35 else framed


def gradient_canvas(size: tuple[int, int], grade: Grade) -> np.ndarray:
    width, height = size
    top, bottom = (np.asarray(color, dtype=np.float32) / 255 for color in grade.gradient)
    blend = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    return np.broadcast_to(top * (1 - blend) + bottom * blend, (height, width, 3)).copy()


def even(value: float) -> int:
    return max(2, int(round(value / 2)) * 2)


@register("offline", "image")
class OfflineStill(Provider):
    def generate(
        self,
        prompt: str,
        references: list[Path],
        aspect: Aspect,
        seed: int,
        out: Path,
        negative: str = "",
    ) -> GenResult:
        rng = np.random.default_rng(seed)
        grade_name = grade_for(prompt)
        grade = GRADES[grade_name]
        reference = load_reference(list(references))
        if reference is None:
            log.info("no usable reference image, painting a %s gradient", grade_name)
            pixels = gradient_canvas(aspect.size, grade)
        else:
            pixels = np.asarray(reframe_reference(reference, aspect.size, rng), dtype=np.float32) / 255
        graded = apply_grade(pixels, grade, rng)
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((graded * 255 + 0.5).astype(np.uint8), "RGB").save(target, format="PNG")
        return GenResult(
            path=target,
            usage={"image": 1},
            meta={"grade": grade_name, "seed": seed, "reference": reference is not None},
        )


def motion_expressions(motion: str, frames: int, rng: np.random.Generator) -> tuple[str, str, str]:
    progress = f"(on/{max(frames - 1, 1)})"
    eased = f"({progress}*{progress}*(3-2*{progress}))"
    wobble_x = (
        f"iw*0.0022*sin(2*PI*on/{CLIP_FPS}*{rng.uniform(0.55, 0.85):.3f}+{rng.uniform(0, 6.28):.3f})"
        f"+iw*0.0011*sin(2*PI*on/{CLIP_FPS}*{rng.uniform(1.1, 1.6):.3f}+{rng.uniform(0, 6.28):.3f})"
    )
    wobble_y = (
        f"ih*0.0016*sin(2*PI*on/{CLIP_FPS}*{rng.uniform(0.45, 0.75):.3f}+{rng.uniform(0, 6.28):.3f})"
        f"+ih*0.0008*sin(2*PI*on/{CLIP_FPS}*{rng.uniform(1.2, 1.7):.3f}+{rng.uniform(0, 6.28):.3f})"
    )
    focus_y = f"{rng.uniform(0.35, 0.45):.3f}"
    if motion == "push_in":
        zoom, across = f"1.04+0.12*{eased}", "0.5"
    elif motion == "pull_out":
        zoom, across = f"1.16-0.12*{eased}", "0.5"
    elif motion == "drift_left":
        zoom, across = "1.1", f"(0.8-0.6*{eased})"
    else:
        zoom, across = "1.1", f"(0.2+0.6*{eased})"
    x = f"(iw-iw/zoom)*{across}+{wobble_x}"
    y = f"(ih-ih/zoom)*{focus_y}+{wobble_y}"
    return zoom, x, y


@register("offline", "video")
class OfflineKenBurns(Provider):
    def animate(
        self,
        image: Path,
        prompt: str,
        seconds: float,
        aspect: Aspect,
        seed: int,
        out: Path,
    ) -> GenResult:
        if seconds <= 0:
            raise PermanentError("clip length must be positive")
        if not Path(image).exists():
            raise PermanentError(f"keyframe {image} does not exist")
        rng = np.random.default_rng(seed)
        motion = motion_for(prompt, seed)
        frames = max(1, round(seconds * CLIP_FPS))
        width, height = aspect.size
        zoom, x, y = motion_expressions(motion, frames, rng)
        big_width, big_height = width * ZOOMPAN_OVERSAMPLE, height * ZOOMPAN_OVERSAMPLE
        graph = (
            f"scale={big_width}:{big_height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={big_width}:{big_height},setsar=1,"
            f"zoompan=z='{zoom}':x='{x}':y='{y}':d={frames}:s={width}x{height}:fps={CLIP_FPS},"
            "format=yuv420p,setsar=1"
        )
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        ff.ffmpeg(
            "-loglevel", "error",
            "-i", str(image),
            "-vf", graph,
            "-frames:v", str(frames),
            "-r", str(CLIP_FPS),
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(target),
            timeout=max(120.0, seconds * 60),
        )
        return GenResult(
            path=target,
            duration=round(frames / CLIP_FPS, 6),
            usage={"second": seconds},
            meta={"motion": motion, "frames": frames},
        )


def say_binary() -> str | None:
    return shutil.which("say")


@lru_cache(maxsize=1)
def installed_voices() -> frozenset[str]:
    binary = say_binary()
    if binary is None:
        return frozenset()
    try:
        listing = ff.run([binary, "-v", "?"], timeout=30.0).stdout
    except ff.FFmpegError as exc:
        log.warning("could not list say voices: %s", exc)
        return frozenset()
    names = (VOICE_LISTING.match(line) for line in listing.splitlines())
    return frozenset(match.group(1).strip() for match in names if match)


def speaking_rate(voice: Voice) -> int:
    return int(min(max(round(BASE_WORDS_PER_MINUTE * voice.speed), 90), 360))


@register("offline", "speech")
class OfflineSay(Provider):
    def speak(self, text: str, voice: Voice, out: Path) -> GenResult:
        binary = say_binary()
        if binary is None:
            raise PermanentError("macOS 'say' is not available on this machine")
        spoken = " ".join(text.split())
        if not spoken:
            raise PermanentError("nothing to speak")
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        chosen = voice.say_voice if voice.say_voice in installed_voices() else None
        if voice.say_voice and chosen is None:
            log.warning("say voice '%s' is not installed, using the system voice", voice.say_voice)
        with tempfile.TemporaryDirectory(prefix="mitosis-say-") as folder:
            script, recording = Path(folder) / "script.txt", Path(folder) / "speech.aiff"
            script.write_text(spoken, encoding="utf-8")
            chosen = self.record(binary, chosen, speaking_rate(voice), script, recording)
            ff.ffmpeg(
                "-loglevel", "error",
                "-i", str(recording),
                "-ar", str(VOICE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
                str(target),
            )
        duration = probe.duration(target)
        return GenResult(
            path=target,
            duration=duration,
            words=align_script(spoken, target, None),
            usage={"kchar": len(text) / 1000},
            meta={"voice": chosen or "system", "rate": speaking_rate(voice)},
        )

    def record(self, binary: str, voice_name: str | None, rate: int, script: Path, recording: Path) -> str | None:
        options = ["-r", str(rate), "-o", str(recording), "-f", str(script)]
        if voice_name:
            try:
                ff.run([binary, "-v", voice_name, *options], timeout=180.0)
                return voice_name
            except ff.FFmpegError as exc:
                log.warning("say failed with voice '%s', retrying with the system voice: %s", voice_name, exc)
        try:
            ff.run([binary, *options], timeout=180.0)
        except ff.FFmpegError as exc:
            raise ProviderError(f"say could not synthesize speech: {exc}") from exc
        return None


def pitch_filter(semitones: float) -> str:
    ratio = 2 ** (semitones / 12)
    if ff.has_filter("rubberband"):
        return f"rubberband=pitch={ratio:.6f}:formant=preserved:pitchq=quality"
    return f"asetrate={VOICE_RATE * ratio:.3f},aresample={VOICE_RATE},atempo={1 / ratio:.6f}"


def tone_filter(semitones: float) -> str:
    if semitones < 0:
        return "lowshelf=f=160:g=2.5,equalizer=f=3200:t=q:w=1.2:g=-1.5"
    if semitones > 0:
        return "lowshelf=f=160:g=-2,equalizer=f=3500:t=q:w=1.2:g=2"
    return "equalizer=f=2500:t=q:w=1:g=1.5"


@register("offline", "voice")
class OfflinePitch(Provider):
    def convert(self, audio: Path, voice: Voice, out: Path) -> GenResult:
        source = Path(audio)
        if not source.exists():
            raise PermanentError(f"source audio {source} does not exist")
        duration = probe.duration(source)
        samples = round(duration * VOICE_RATE)
        chain = [
            f"aresample={VOICE_RATE}",
            "aformat=sample_fmts=fltp:channel_layouts=mono",
            pitch_filter(voice.pitch_semitones),
            tone_filter(voice.pitch_semitones),
            "highpass=f=70",
            "volume=-1.5dB",
            f"apad=whole_len={samples}",
            f"atrim=end_sample={samples}",
        ]
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        ff.ffmpeg(
            "-loglevel", "error",
            "-i", str(source),
            "-vn", "-af", ",".join(chain),
            "-ar", str(VOICE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
            str(target),
        )
        return GenResult(
            path=target,
            duration=round(samples / VOICE_RATE, 6),
            usage={"minute": duration / 60},
            meta={"semitones": voice.pitch_semitones},
        )


@register("offline", "transcribe")
class OfflineAlign(Provider):
    def transcribe(self, audio: Path, out: Path, hint: str | None = None) -> GenResult:
        source = Path(audio)
        duration = probe.duration(source)
        words = align_script(hint, source, None) if hint and hint.strip() else []
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps([word.model_dump() for word in words], indent=2), encoding="utf-8")
        return GenResult(
            path=target,
            words=words,
            duration=duration,
            data={"text": " ".join(word.text for word in words)},
            usage={"minute": duration / 60},
        )


@register("offline", "upscale")
class OfflineLanczos(Provider):
    def upscale(self, video: Path, out: Path, factor: int = 2) -> GenResult:
        if factor < 1:
            raise PermanentError("upscale factor must be at least 1")
        info = probe.probe(Path(video))
        if info.width <= 0:
            raise PermanentError(f"{video} has no video stream to upscale")
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        audio_args = ["-c:a", "copy"] if info.has_audio else ["-an"]
        ff.ffmpeg(
            "-loglevel", "error",
            "-i", str(video),
            "-map", "0:v:0", *(["-map", "0:a:0"] if info.has_audio else []),
            "-vf", f"scale={even(info.width * factor)}:{even(info.height * factor)}:flags=lanczos,setsar=1",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p",
            *audio_args,
            "-movflags", "+faststart",
            str(target),
            timeout=max(300.0, info.duration * 30),
        )
        return GenResult(path=target, duration=info.duration, usage={"second": info.duration}, meta={"factor": factor})
