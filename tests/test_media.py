from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from mitosis.config import PACKAGE_ROOT, Brand, LoudnessSpec, ModelSpec, Settings, read_yaml
from mitosis.media import align, assemble, audio, ff, frames, probe, reframe, scenes, subtitles
from mitosis.models import Aspect, AudioPart, Box, Headline, Location, MediaInfo, RenderJob, Segment, Shot, Voice, Word
from mitosis.providers import offline
from mitosis.providers.base import FACTORIES, PermanentError, Provider

needs_say = pytest.mark.skipif(offline.say_binary() is None, reason="macOS say is not available")

POSITION = re.compile(r"\\pos\((\d+),(\d+)\)")
FONT_OVERRIDE = re.compile(r"\\fs(\d+)")
OVERRIDE_BLOCK = re.compile(r"\{[^}]*\}")


def synthesize(out: Path, *args: str) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    ff.ffmpeg("-loglevel", "error", *args, str(out))
    return out


@pytest.fixture(scope="session")
def media(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("media with spaces")


@pytest.fixture(scope="session")
def landscape_clip(media: Path) -> Path:
    return synthesize(
        media / "landscape clip.mp4",
        "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=2",
        "-f", "lavfi", "-i", "smptebars=s=640x360:r=30:d=2",
        "-f", "lavfi", "-i", "sine=f=330:d=4:r=44100",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-map", "2:a",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
    )


@pytest.fixture(scope="session")
def portrait_clip(media: Path) -> Path:
    return synthesize(
        media / "portrait clip.mp4",
        "-f", "lavfi", "-i", "testsrc2=s=360x640:r=24:d=2.5",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
    )


@pytest.fixture(scope="session")
def bursts_wav(media: Path) -> Path:
    return synthesize(
        media / "bursts.wav",
        "-f", "lavfi",
        "-i", "aevalsrc='0.4*sin(2*PI*220*t)*(between(t,0.2,1.2)+between(t,1.8,2.8))':s=48000:d=3.2",
        "-c:a", "pcm_s16le",
    )


@pytest.fixture(scope="session")
def spiky_wav(media: Path) -> Path:
    return synthesize(
        media / "spiky.wav",
        "-f", "lavfi",
        "-i", "aevalsrc='0.03*sin(2*PI*180*t)+0.95*lt(mod(t,0.4),0.0015)':s=48000:d=3",
        "-c:a", "pcm_s16le",
    )


@pytest.fixture(scope="session")
def music_wav(media: Path) -> Path:
    return synthesize(
        media / "music bed.wav",
        "-f", "lavfi", "-i", "anoisesrc=d=2:c=pink:a=0.3:seed=3",
        "-f", "lavfi", "-i", "sine=f=110:d=2",
        "-filter_complex", "[0:a][1:a]amix=inputs=2,aformat=channel_layouts=stereo",
        "-ar", "44100", "-c:a", "pcm_s16le",
    )


@pytest.fixture(scope="session")
def brand() -> Brand:
    loaded = Brand(**read_yaml(PACKAGE_ROOT / "config" / "brand.yaml"))
    return loaded.model_copy(update={"video": loaded.video.model_copy(update={"preset": "ultrafast"})})


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        root=tmp_path,
        config_dir=PACKAGE_ROOT / "config",
        runs_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        models_dir=tmp_path / "models",
        priors_path=tmp_path / "priors.json",
    )


def provider(kind: str, settings: Settings) -> Provider:
    spec = ModelSpec(id=f"offline_{kind}", provider="offline", kind=kind, offline=True)
    return FACTORIES[("offline", kind)](spec, settings)


def channels(path: Path) -> int:
    return int(ff.ffprobe_json(path)["streams"][0]["channels"])


def test_probe_reads_video_and_audio(landscape_clip: Path, bursts_wav: Path) -> None:
    info = probe.probe(landscape_clip)
    assert (info.width, info.height, info.fps) == (640, 360, 30.0)
    assert info.duration == pytest.approx(4.0, abs=0.05)
    assert info.has_audio and info.video_codec == "h264" and info.pix_fmt == "yuv420p"
    sound = probe.probe(bursts_wav)
    assert (sound.width, sound.height, sound.fps) == (0, 0, 0.0)
    assert sound.has_audio and sound.sample_rate == 48000
    assert probe.duration(bursts_wav) == pytest.approx(3.2, abs=0.001)


def test_probe_and_ffmpeg_report_failures(tmp_path: Path) -> None:
    with pytest.raises(ff.FFmpegError, match="missing"):
        probe.probe(tmp_path / "missing.mp4")
    with pytest.raises(ff.FFmpegError, match="exited with code"):
        ff.ffmpeg("-loglevel", "error", "-i", str(tmp_path / "absent.mp4"), "-f", "null", "-")


def test_detect_shots_finds_hard_cut(landscape_clip: Path) -> None:
    info = probe.probe(landscape_clip)
    spans = scenes.detect_shots(landscape_clip, info)
    assert len(spans) == 2
    assert spans[0][0] == 0.0 and spans[-1][1] == pytest.approx(info.duration, abs=0.01)
    assert spans[0][1] == spans[1][0] == pytest.approx(2.0, abs=0.1)
    assert scenes.detect_shots(landscape_clip, info, min_length=2.5) == [(0.0, round(info.duration, 3))]


def test_merge_short_drops_cuts_that_create_tiny_shots() -> None:
    assert scenes.merge_short([0.3, 1.0, 1.2, 3.8], 4.0, 0.6) == [1.0]
    assert scenes.merge_short([], 4.0, 0.6) == []


def test_extract_frame_scales_and_survives_late_seek(landscape_clip: Path, tmp_path: Path) -> None:
    jpeg = frames.extract_frame(landscape_clip, 1.0, tmp_path / "nested dir" / "frame.jpg", width=320)
    png = frames.extract_frame(landscape_clip, 2.5, tmp_path / "frame.png")
    late = frames.extract_frame(landscape_clip, 30.0, tmp_path / "late.jpg", width=160)
    assert Image.open(jpeg).size == (320, 180)
    assert Image.open(png).size == (640, 360) and Image.open(png).format == "PNG"
    assert Image.open(late).size == (160, 90)


def test_sample_frames_are_evenly_spaced(landscape_clip: Path, tmp_path: Path) -> None:
    samples = frames.sample_frames(landscape_clip, tmp_path / "samples", every=0.5, width=200)
    assert [t for t, _ in samples] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    assert all(path.exists() and Image.open(path).width == 200 for _, path in samples)
    window = frames.sample_frames(landscape_clip, tmp_path / "samples", every=1.0, width=200, start=1.0, end=3.5)
    assert [t for t, _ in window] == [1.0, 2.0, 3.0]
    assert len(list((tmp_path / "samples").glob("*.jpg"))) == 3


def test_sharpness_drops_when_blurred(landscape_clip: Path, tmp_path: Path) -> None:
    crisp = frames.extract_frame(landscape_clip, 1.0, tmp_path / "crisp.png")
    blurred = tmp_path / "blurred.png"
    Image.open(crisp).filter(ImageFilter.GaussianBlur(4)).save(blurred)
    assert frames.sharpness(crisp) > frames.sharpness(blurred) * 3
    with pytest.raises(ValueError):
        frames.sharpness(tmp_path / "missing.png")


def test_extract_audio_window_is_exact_mono_48k(landscape_clip: Path, tmp_path: Path) -> None:
    clip = audio.extract_audio(landscape_clip, tmp_path / "audio out" / "window.wav", 1.0, 3.0)
    info = probe.probe(clip)
    assert info.sample_rate == 48000 and channels(clip) == 1
    assert info.duration == pytest.approx(2.0, abs=0.001)
    padded = audio.extract_audio(landscape_clip, tmp_path / "padded.wav", 3.0, 5.0)
    assert probe.duration(padded) == pytest.approx(2.0, abs=0.001)


def test_extract_audio_from_silent_video_writes_silence(portrait_clip: Path, tmp_path: Path) -> None:
    silence = audio.extract_audio(portrait_clip, tmp_path / "silence.wav")
    assert probe.duration(silence) == pytest.approx(2.5, abs=0.001)
    assert audio.measure_loudness(silence).integrated == -70.0


def test_speech_intervals_follow_tone_bursts(bursts_wav: Path) -> None:
    intervals = audio.speech_intervals(bursts_wav)
    assert len(intervals) == 2
    for (start, end), (expected_start, expected_end) in zip(intervals, [(0.2, 1.2), (1.8, 2.8)]):
        assert start == pytest.approx(expected_start, abs=0.05)
        assert end == pytest.approx(expected_end, abs=0.05)


def test_separate_stems_returns_none_without_demucs(bursts_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio, "demucs_installed", lambda: False)
    assert audio.separate_stems(bursts_wav, tmp_path / "stems") is None


@pytest.mark.parametrize("with_bed", [False, True])
def test_build_track_hits_loudness_and_exact_length(
    with_bed: bool, bursts_wav: Path, spiky_wav: Path, music_wav: Path, tmp_path: Path
) -> None:
    spec = LoudnessSpec()
    parts = [AudioPart(path=bursts_wav, offset=0.0), AudioPart(path=spiky_wav, offset=2.9, gain_db=-3.0)]
    total = 6.5
    out = audio.build_track(parts, total, tmp_path / "mix dir" / "mix.wav", spec, bed=music_wav if with_bed else None)
    info = probe.probe(out)
    assert info.sample_rate == 48000 and channels(out) == 2
    assert info.duration == pytest.approx(total, abs=0.001)
    loudness = audio.measure_loudness(out)
    assert abs(loudness.integrated - spec.target_lufs) <= 1.0
    assert loudness.true_peak <= spec.true_peak + 0.5


def test_build_track_without_parts_is_silent_but_exact(tmp_path: Path) -> None:
    out = audio.build_track([], 1.5, tmp_path / "empty.wav", LoudnessSpec())
    assert probe.duration(out) == pytest.approx(1.5, abs=0.001)


def test_tokenize_keeps_punctuation() -> None:
    script = "Honestly, I thought we were fine. We weren't \u2014 right ?  Me & you..."
    assert align.tokenize(script) == ["Honestly,", "I", "thought", "we", "were", "fine.", "We", "weren't\u2014", "right?", "Me", "&", "you..."]


def test_proportional_words_cover_span_by_weight() -> None:
    words = align.proportional_words("a extraordinary day.", 1.0, 3.0)
    assert [w.text for w in words] == ["a", "extraordinary", "day."]
    assert words[0].start == 1.0 and words[-1].end == 3.0
    assert all(left.end == right.start for left, right in zip(words, words[1:]))
    assert words[1].end - words[1].start > words[0].end - words[0].start
    assert align.proportional_words("   ", 0.0, 1.0) == []


def test_window_clips_and_maps_times() -> None:
    words = [Word(text="a", start=0.5, end=1.5), Word(text="b", start=1.5, end=2.9), Word(text="c", start=2.6, end=3.0)]
    assert align.window(words, 1.0, 2.6, offset=-1.0, scale=2.0) == [Word(text="b", start=1.0, end=3.2)]
    assert [w.text for w in align.window(words, 0.0, 10.0)] == ["a", "b", "c"]


def test_align_with_transcript_takes_matched_timings(bursts_wav: Path) -> None:
    transcript = [
        Word(text="so", start=0.2, end=0.4),
        Word(text="i", start=0.4, end=0.5),
        Word(text="downloaded", start=0.5, end=1.0),
        Word(text="it", start=1.0, end=1.2),
        Word(text="10", start=1.8, end=2.1),
        Word(text="minutes", start=2.1, end=2.5),
    ]
    words = align.align_script("So I downloaded it. Ten minutes a day.", bursts_wav, transcript)
    assert [w.text for w in words] == ["So", "I", "downloaded", "it.", "Ten", "minutes", "a", "day."]
    assert (words[2].start, words[2].end) == (0.5, 1.0)
    assert (words[4].start, words[4].end) == (1.8, 2.1)
    assert words[6].start >= 2.5 and words[7].end <= 3.2
    assert all(left.start <= right.start for left, right in zip(words, words[1:]))


def test_align_without_transcript_snaps_sentences_to_pauses(bursts_wav: Path) -> None:
    words = align.align_script("We were totally fine. Honestly we were not.", bursts_wav)
    by_text = {w.text: w for w in words}
    assert by_text["fine."].end <= 1.25
    assert by_text["Honestly"].start >= 1.75
    assert words[0].start >= 0.15 and words[-1].end <= 2.85
    assert align.align_script("", bursts_wav) == []


def test_chunk_breaks_on_punctuation_gaps_and_limits() -> None:
    timed = align.proportional_words("Wait. I thought we were totally fine, honestly", 0.0, 4.0)
    late = Word(text="later", start=6.0, end=6.4)
    groups = subtitles.chunk([*timed, late], max_words=3, max_chars=18, max_gap=0.45)
    assert [[w.text for w in group] for group in groups] == [
        ["Wait."],
        ["I", "thought", "we"],
        ["were", "totally", "fine,"],
        ["honestly"],
        ["later"],
    ]


def test_ass_colors_use_alpha_blue_green_red() -> None:
    assert subtitles.ass_color("#FFD84D") == "&H004DD8FF"
    assert subtitles.ass_color("#FFFFFF", 0.94) == "&H0FFFFFFF"
    with pytest.raises(ValueError):
        subtitles.ass_color("not a color")


def parse_ass(path: Path) -> tuple[dict[str, str], dict[str, list[str]], list[list[str]]]:
    info: dict[str, str] = {}
    styles: dict[str, list[str]] = {}
    events: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Style:"):
            fields = line[len("Style:") :].strip().split(",")
            styles[fields[0]] = fields
        elif line.startswith("Dialogue:"):
            events.append(line[len("Dialogue:") :].strip().split(",", 9))
        elif ":" in line and not line.startswith("Format"):
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    return info, styles, events


def assert_inside_zone(event: list[str], styles: dict[str, list[str]], brand: Brand, aspect: Aspect) -> None:
    style = styles[event[3]]
    text = event[9]
    size = int(FONT_OVERRIDE.search(text).group(1)) if FONT_OVERRIDE.search(text) else int(style[2])
    border = float(style[16]) * size / int(style[2])
    grow = 1.16 if event[3] == "Caption" else 1.0
    lines = text.count("\\N") + 1
    half = (lines * size + size * (grow - 1.0) + 2 * border) / 2
    x, y = map(int, POSITION.search(text).groups())
    left, top, right, bottom = brand.safe_zone(aspect).rect(aspect)
    assert top <= y - half and y + half <= bottom
    assert left <= x <= right


@pytest.mark.parametrize("aspect", list(Aspect))
def test_write_ass_places_captions_inside_safe_zone(aspect: Aspect, brand: Brand, tmp_path: Path) -> None:
    words = align.proportional_words("Honestly, I thought we were fine. Were we? Nobody tells you this!", 0.0, 5.0)
    headline = Headline(text="Married guys: nobody tells you this about year seven", start=0.0, end=2.5)
    path = subtitles.write_ass(words, aspect, brand, tmp_path / "subs dir" / f"{aspect.slug}.ass", headline)
    info, styles, events = parse_ass(path)
    assert (int(info["PlayResX"]), int(info["PlayResY"])) == aspect.size
    assert styles["Caption"][2] == str(round(brand.subtitle.size_ratio * aspect.size[0]))
    assert styles["Headline"][15] == "3" and styles["Headline"][5] == "&H0FFFFFFF"
    captions = [event for event in events if event[3] == "Caption"]
    banners = [event for event in events if event[3] == "Headline"]
    assert len(banners) == 1 and "\\fad(" in banners[0][9]
    assert (banners[0][1], banners[0][2]) == ("0:00:00.00", "0:00:02.50")
    for event in captions + banners:
        assert_inside_zone(event, styles, brand, aspect)
    plain = [OVERRIDE_BLOCK.sub("", event[9]) for event in captions]
    assert "HONESTLY" in plain[0] and all("," not in text and "." not in text for text in plain)
    assert any("WE?" in text for text in plain) and any("THIS!" in text for text in plain)
    assert all(subtitles.inline_color(brand.subtitle.highlight) in event[9] and "\\t(" in event[9] for event in captions)
    assert len(captions) == len(words)


def test_write_ass_clamps_extreme_anchors(brand: Brand, tmp_path: Path) -> None:
    extreme = brand.model_copy(
        update={
            "subtitle": brand.subtitle.model_copy(update={"anchor": {"9:16": 0.98, "4:5": 0.98, "1:1": 0.98}}),
            "headline": brand.headline.model_copy(update={"anchor": {"9:16": 0.0, "4:5": 0.0, "1:1": 0.0}}),
        }
    )
    words = align.proportional_words("Supercalifragilisticexpialidocious nonsense words here", 0.0, 3.0)
    for aspect in Aspect:
        path = subtitles.write_ass(words, aspect, extreme, tmp_path / f"{aspect.slug}.ass", Headline(text="Top", start=0, end=1))
        _, styles, events = parse_ass(path)
        for event in events:
            assert_inside_zone(event, styles, extreme, aspect)


def test_write_ass_escapes_override_characters(brand: Brand, tmp_path: Path) -> None:
    words = [Word(text="{hack}", start=0.0, end=0.5), Word(text="back\\slash", start=0.5, end=1.0)]
    path = subtitles.write_ass(words, Aspect.SQUARE, brand, tmp_path / "escape.ass")
    _, _, events = parse_ass(path)
    for event in events:
        plain = OVERRIDE_BLOCK.sub("", event[9])
        assert "{" not in plain and "}" not in plain and "\\N" not in plain.replace("\\\u2060", "")


def test_shots_filter_switches_crop_window_at_shot_boundaries() -> None:
    info = MediaInfo(path=Path("source.mp4"), width=1920, height=1080, fps=30.0, duration=6.0, has_audio=True)
    shots = [
        Shot(index=0, start=0.0, end=2.0, face=Box(x=0.1, y=0.2, w=0.1, h=0.2)),
        Shot(index=1, start=2.0, end=4.0, face=Box(x=0.75, y=0.2, w=0.1, h=0.2)),
        Shot(index=2, start=4.0, end=6.0),
    ]
    graph = reframe.shots_filter(info, shots, Aspect.VERTICAL, source_start=1.0)
    assert graph.startswith("crop=w=606:h=1080:")
    assert graph.endswith("scale=1080:1920:flags=lanczos,setsar=1")
    x_expression = re.search(r"x='([^']*)'", graph).group(1)
    assert x_expression == "if(lt(t,1.0000),0,if(lt(t,3.0000),1232,656))"


def test_shots_filter_scales_when_ratio_matches_and_centers_without_faces() -> None:
    vertical = MediaInfo(path=Path("v.mp4"), width=720, height=1280, fps=30.0, duration=3.0, has_audio=False)
    assert reframe.shots_filter(vertical, [], Aspect.VERTICAL, 0.0) == "scale=1080:1920:flags=lanczos,setsar=1"
    square = reframe.shots_filter(vertical, [], Aspect.SQUARE, 0.0)
    assert square.startswith("crop=w=720:h=720:x='0':y='280'")


def test_cover_filter_scales_to_cover_and_crops_above_center() -> None:
    graph = reframe.cover_filter(360, 640, Aspect.SQUARE)
    assert graph == "scale=1080:1920:flags=lanczos,crop=1080:1080:0:336,setsar=1"


def test_frame_budget_sums_to_rounded_total() -> None:
    budget = assemble.frame_budget([1.37, 2.21, 0.9], 30)
    assert sum(budget) == round((1.37 + 2.21 + 0.9) * 30)


def test_render_two_segments_to_every_aspect(
    portrait_clip: Path, landscape_clip: Path, bursts_wav: Path, brand: Brand, tmp_path: Path
) -> None:
    info = probe.probe(landscape_clip)
    shots = [
        Shot(index=0, start=0.0, end=2.0, face=Box(x=0.1, y=0.2, w=0.2, h=0.3)),
        Shot(index=1, start=2.0, end=4.0, face=Box(x=0.7, y=0.1, w=0.2, h=0.3)),
    ]
    hook_seconds, body_seconds = 2.0, 3.4
    track = audio.build_track(
        [AudioPart(path=bursts_wav, offset=0.0)], hook_seconds + body_seconds, tmp_path / "track.wav", brand.loudness
    )
    words = align.proportional_words("Nobody tells you this. So I downloaded Betterly, and it works.", 0.0, 5.4)
    weird = tmp_path / "subs, it's [weird]: dir"
    for aspect in Aspect:
        captions = subtitles.write_ass(words, aspect, brand, weird / f"captions {aspect.slug}.ass", Headline(text="Year seven", start=0, end=2))
        job = RenderJob(
            segments=[
                Segment(path=portrait_clip, seconds=hook_seconds, start=0.0, end=2.4, reframe="cover"),
                Segment(path=landscape_clip, seconds=body_seconds, start=1.0, end=None, reframe="shots", shots=shots, info=info),
            ],
            audio=track,
            aspect=aspect,
            out=tmp_path / "renders" / f"variant {aspect.slug}.mp4",
            subtitles=captions,
        )
        rendered = probe.probe(assemble.render(job, brand))
        assert (rendered.width, rendered.height) == aspect.size
        assert rendered.duration == pytest.approx(hook_seconds + body_seconds, abs=0.1)
        assert rendered.fps == 30.0 and rendered.has_audio and rendered.pix_fmt == "yuv420p"
        thumb = assemble.thumbnail(job.out, tmp_path / "thumbs" / f"{aspect.slug}.jpg")
        assert Image.open(thumb).width == 360
    assert not list((tmp_path / "renders").glob(".*partial*"))


def test_offline_providers_are_registered() -> None:
    for kind in ("image", "video", "speech", "voice", "transcribe", "upscale"):
        assert ("offline", kind) in FACTORIES


@pytest.mark.parametrize("location", list(read_yaml(PACKAGE_ROOT / "config" / "locations.yaml")["locations"].items()))
def test_offline_still_reads_the_location_grade(location: tuple[str, dict]) -> None:
    key, data = location
    place = Location(id=key, **data)
    assert offline.grade_for(f"A man in his 30s. Setting: {place.prompt}. Action: looks up. natural window light") == place.grade


def test_offline_still_is_deterministic_and_graded(landscape_clip: Path, settings: Settings, tmp_path: Path) -> None:
    reference = frames.extract_frame(landscape_clip, 1.0, tmp_path / "reference.jpg")
    still = provider("image", settings)
    warm_prompt = "Setting: a living room in the evening, warm lamps. Action: smiles"
    cool_prompt = "Setting: a bright kitchen in the morning, cool daylight. Action: smiles"
    first = still.generate(warm_prompt, [reference], Aspect.VERTICAL, 5, tmp_path / "out" / "warm a.png")
    second = still.generate(warm_prompt, [reference], Aspect.VERTICAL, 5, tmp_path / "out" / "warm b.png")
    cool = still.generate(cool_prompt, [reference], Aspect.VERTICAL, 5, tmp_path / "out" / "cool.png")
    warm_pixels = np.asarray(Image.open(first.path), dtype=np.float64)
    cool_pixels = np.asarray(Image.open(cool.path), dtype=np.float64)
    assert np.array_equal(warm_pixels, np.asarray(Image.open(second.path), dtype=np.float64))
    assert warm_pixels.shape == (1920, 1080, 3) and first.usage == {"image": 1}
    warm_balance = warm_pixels[..., 0].mean() / warm_pixels[..., 2].mean()
    cool_balance = cool_pixels[..., 0].mean() / cool_pixels[..., 2].mean()
    assert warm_balance > cool_balance
    fallback = still.generate("Setting: nowhere", [tmp_path / "missing.png"], Aspect.SQUARE, 1, tmp_path / "gradient.png")
    assert Image.open(fallback.path).size == (1080, 1080) and fallback.meta["reference"] is False


def test_offline_kenburns_renders_exact_clip(settings: Settings, tmp_path: Path) -> None:
    image = tmp_path / "key.png"
    ramp = np.linspace(0, 255, 1080, dtype=np.uint8)
    Image.fromarray(np.stack([np.tile(ramp, (1920, 1)), np.tile(ramp[::-1], (1920, 1)), np.full((1920, 1080), 90, np.uint8)], axis=-1)).save(image)
    result = provider("video", settings).animate(image, "slow push-in on his face", 2.2, Aspect.VERTICAL, 9, tmp_path / "clips" / "clip.mp4")
    info = probe.probe(result.path)
    assert (info.width, info.height, info.fps) == (1080, 1920, 30.0)
    assert info.duration == pytest.approx(2.2, abs=0.04) and not info.has_audio
    assert result.usage == {"second": 2.2} and result.meta["motion"] == "push_in"


@needs_say
def test_offline_say_speaks_with_word_timings(settings: Settings, tmp_path: Path) -> None:
    text = "Nobody tells you this. It gets quiet."
    voice = Voice(id="deep", label="Deep", say_voice="Rocko (English (US))", speed=1.1)
    result = provider("speech", settings).speak(text, voice, tmp_path / "voice out" / "hook.wav")
    info = probe.probe(result.path)
    assert info.sample_rate == 48000 and channels(result.path) == 1
    assert result.duration == pytest.approx(info.duration, abs=0.001) and result.duration > 0.8
    assert [w.text for w in result.words] == align.tokenize(text)
    assert result.usage == {"kchar": len(text) / 1000}
    fallback = provider("speech", settings).speak("Fallback voice.", Voice(id="x", label="x", say_voice="No Such Voice"), tmp_path / "fb.wav")
    assert fallback.meta["voice"] == "system" and fallback.duration > 0.3


def test_offline_say_without_binary_is_permanent(settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(offline, "say_binary", lambda: None)
    with pytest.raises(PermanentError):
        provider("speech", settings).speak("hello", Voice(id="v", label="v"), tmp_path / "x.wav")


def test_offline_pitch_keeps_exact_timing(bursts_wav: Path, settings: Settings, tmp_path: Path) -> None:
    voice = Voice(id="low", label="Low", pitch_semitones=-3.0)
    result = provider("voice", settings).convert(bursts_wav, voice, tmp_path / "converted.wav")
    assert probe.duration(result.path) == pytest.approx(probe.duration(bursts_wav), abs=0.001)
    assert result.usage == {"minute": pytest.approx(3.2 / 60)}
    assert len(audio.speech_intervals(result.path)) == 2


def test_offline_align_writes_transcript_json(bursts_wav: Path, settings: Settings, tmp_path: Path) -> None:
    transcriber = provider("transcribe", settings)
    result = transcriber.transcribe(bursts_wav, tmp_path / "json dir" / "transcript.json", hint="We were fine. We were not.")
    stored = json.loads(result.path.read_text())
    assert [item["text"] for item in stored] == [w.text for w in result.words] == align.tokenize("We were fine. We were not.")
    assert result.usage == {"minute": pytest.approx(3.2 / 60)}
    empty = transcriber.transcribe(bursts_wav, tmp_path / "empty.json")
    assert empty.words == [] and json.loads(empty.path.read_text()) == []


def test_offline_lanczos_doubles_resolution(landscape_clip: Path, settings: Settings, tmp_path: Path) -> None:
    result = provider("upscale", settings).upscale(landscape_clip, tmp_path / "up" / "big.mp4", factor=2)
    info = probe.probe(result.path)
    assert (info.width, info.height) == (1280, 720) and info.has_audio
    assert result.usage == {"second": pytest.approx(4.0, abs=0.05)}
