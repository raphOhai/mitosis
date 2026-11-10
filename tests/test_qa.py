from __future__ import annotations

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from mitosis.config import Brand, SafeZone, Settings
from mitosis.models import Aspect, Check, GateInput, HeroPack, Hook, Location, PolicyFlag, QAReport, VariantSpec, Voice
from mitosis.providers.base import GenResult, ProviderError
from mitosis.qa import gate, identity, judge, safezones, technical

SAMPLE_WINNER = Path(__file__).resolve().parent.parent / "samples" / "winner.mp4"
SHARED_MODELS = Path(os.environ.get("MITOSIS_MODELS_DIR", Path.home() / ".cache" / "mitosis" / "models"))

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: {wrap}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Arial Black,60,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,4,0,2,90,90,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ffmpeg() -> str:
    try:
        return technical.binary("ffmpeg")
    except technical.ToolMissing:
        pytest.skip("ffmpeg is not installed")


def synthesize(target: Path, size: str = "1080x1080", seconds: float = 3.0, video_filter: str = "", volume_db: float = 7.0) -> Path:
    command = [
        ffmpeg(), "-v", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=1000:sample_rate=48000:duration={seconds}",
    ]
    if video_filter:
        command += ["-filter_complex", f"[0:v]{video_filter}[v]", "-map", "[v]"]
    else:
        command += ["-map", "0:v"]
    command += [
        "-map", "1:a", "-af", f"volume={volume_db}dB",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", str(target),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=120)
    return target


@pytest.fixture(scope="module")
def clips(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    folder = tmp_path_factory.mktemp("work demo")
    return {
        "good": synthesize(folder / "good clip.mp4"),
        "black": synthesize(folder / "black.mp4", video_filter="drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='between(t,1.5,2.2)'"),
        "frozen": synthesize(folder / "frozen.mp4", video_filter="split[a][b];[a][b]freezeframes=first=15:last=60:replace=15"),
        "quiet": synthesize(folder / "quiet.mp4", volume_db=-30.0),
        "small": synthesize(folder / "small.mp4", size="720x720"),
    }


@pytest.fixture
def brand() -> Brand:
    return Brand(
        name="Betterly",
        product="Betterly",
        category="relationships",
        audience="couples",
        cta="Try Betterly free",
        tone="warm",
        safe_zones={
            "9:16": SafeZone(top=0.14, bottom=0.35, left=0.06, right=0.06),
            "1:1": SafeZone(top=0.06, bottom=0.10, left=0.05, right=0.05),
        },
    )


def make_settings(tmp_path: Path, models_dir: Path | None = None) -> Settings:
    return Settings(
        root=tmp_path,
        config_dir=tmp_path,
        runs_dir=tmp_path,
        cache_dir=tmp_path,
        models_dir=models_dir or tmp_path / "models",
        priors_path=tmp_path / "priors.json",
    )


def gate_input(video: Path, aspect: Aspect = Aspect.SQUARE, subtitles: Path | None = None, expected: float = 3.0, hook: float = 1.0) -> GateInput:
    return GateInput(
        variant_id="V01",
        aspect=aspect,
        video=video,
        subtitles=subtitles,
        expected_seconds=expected,
        hook_seconds=hook,
        hook_text="We stopped talking about anything real.",
        on_screen_text="WE STOPPED TALKING",
        spoken_text="We stopped talking about anything real. Try Betterly free.",
    )


def variant_spec(new_shot: bool = True) -> VariantSpec:
    return VariantSpec(
        id="V01",
        index=0,
        hook=Hook(id="h1", archetype="confession", text="We stopped talking.", on_screen_text="WE STOPPED TALKING", visual="man on couch", motion="slow push in"),
        voice=Voice(id="v1", label="Warm male"),
        location=Location(id="kitchen", label="Kitchen, morning", prompt="bright kitchen"),
        aspects=[Aspect.SQUARE],
        seed=7,
        new_shot=new_shot,
    )


def by_name(checks: list[Check]) -> dict[str, Check]:
    return {check.name: check for check in checks}


def write_ass(folder: Path, lines: list[str], margin_v: int = 700, size: tuple[int, int] = (1080, 1920), wrap: int = 0) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "captions.ass"
    header = ASS_HEADER.format(width=size[0], height=size[1], wrap=wrap, margin_v=margin_v)
    path.write_text(header + "\n".join(f"Dialogue: 0,{line}" for line in lines) + "\n")
    return path


def report(*checks: Check) -> QAReport:
    return QAReport(variant_id="V01", aspect=Aspect.SQUARE, checks=list(checks))


def test_good_render_passes_every_technical_check(clips, brand):
    checks = technical.check_render(gate_input(clips["good"]), brand)
    names = {check.name for check in checks}
    assert {"resolution", "video_codec", "audio_stream", "duration", "black_frames", "frozen_hook", "loudness", "true_peak", "speech_start"} <= names
    failing = [f"{c.name}: {c.detail}" for c in checks if not c.passed and not c.skipped]
    assert failing == []
    assert by_name(checks)["loudness"].score == pytest.approx(-14.0, abs=1.0)


def test_black_gap_blocks_with_render_retry_outside_hook(clips, brand):
    checks = by_name(technical.check_render(gate_input(clips["black"], hook=1.0), brand))
    black = checks["black_frames"]
    assert not black.passed and black.severity == "block" and black.retry == "render"
    assert "1.5" in black.detail


def test_black_gap_inside_hook_asks_for_new_shot(clips, brand):
    black = by_name(technical.check_render(gate_input(clips["black"], hook=2.0), brand))["black_frames"]
    assert not black.passed and black.retry == "shot"


def test_frozen_hook_blocks_and_requests_new_shot(clips, brand):
    checks = by_name(technical.check_render(gate_input(clips["frozen"], hook=2.5), brand))
    frozen = checks["frozen_hook"]
    assert not frozen.passed and frozen.severity == "block" and frozen.retry == "shot"
    assert checks["frozen_body"].passed
    assert gate.decide([report(frozen)], attempt=1, max_attempts=3) == "retry_shot"


def test_freeze_after_hook_is_only_a_warning(clips, brand):
    checks = by_name(technical.check_render(gate_input(clips["frozen"], hook=0.3), brand))
    assert checks["frozen_hook"].passed
    assert not checks["frozen_body"].passed and checks["frozen_body"].severity == "warn"


def test_quiet_track_fails_loudness_and_speech_start(clips, brand):
    checks = by_name(technical.check_render(gate_input(clips["quiet"]), brand))
    assert not checks["loudness"].passed and checks["loudness"].severity == "block"
    assert checks["loudness"].score < -40
    assert not checks["speech_start"].passed and checks["speech_start"].severity == "warn"


def test_wrong_resolution_and_duration_are_blocking(clips, brand):
    checks = by_name(technical.check_render(gate_input(clips["small"], expected=4.0), brand))
    assert not checks["resolution"].passed and "720x720" in checks["resolution"].detail
    assert not checks["duration"].passed and checks["duration"].retry == "render"


def test_missing_render_reports_probe_failure(tmp_path, brand):
    checks = technical.check_render(gate_input(tmp_path / "missing file.mp4"), brand)
    assert [c.name for c in checks] == ["render_probe"]
    assert not checks[0].passed and checks[0].retry == "render"


def test_captions_inside_safe_zone_pass(tmp_path, brand):
    ass = write_ass(tmp_path / "subs dir", ["0:00:00.00,0:00:01.00,Caption,,0,0,0,,TRY IT FREE", "0:00:01.00,0:00:02.00,Caption,,0,0,0,,{\\an5\\pos(540,900)}WE STOPPED\\NTALKING"])
    check = safezones.check_text_placement(gate_input(Path("x.mp4"), Aspect.VERTICAL, ass), brand)
    assert check.passed, check.detail
    assert "2 text placements" in check.detail


def test_captions_in_reels_ui_zone_fail_with_timestamps(tmp_path, brand):
    ass = write_ass(tmp_path, ["0:00:01.20,0:00:02.40,Caption,,0,0,0,,TRY IT FREE"], margin_v=200)
    check = safezones.check_text_placement(gate_input(Path("x.mp4"), Aspect.VERTICAL, ass), brand)
    assert not check.passed and check.severity == "block"
    assert "1.20-2.40s" in check.detail and "bottom" in check.detail


def test_override_position_and_pop_scale_are_respected(tmp_path, brand):
    ass = write_ass(
        tmp_path,
        [
            "0:00:00.00,0:00:01.00,Caption,,0,0,0,,{\\an8\\pos(540,150)}HEADLINE",
            "0:00:01.00,0:00:02.00,Caption,,0,0,0,,{\\an5\\pos(540,1205)}STILL",
            "0:00:02.00,0:00:03.00,Caption,,0,0,0,,{\\an5\\pos(540,1205)\\t(0,100,\\fscx130\\fscy130)}POP",
        ],
    )
    check = safezones.check_text_placement(gate_input(Path("x.mp4"), Aspect.VERTICAL, ass), brand)
    assert not check.passed
    assert "2/3" in check.detail and "'HEADLINE'" in check.detail and "top" in check.detail
    assert "'POP'" in check.detail and "'STILL'" not in check.detail


def test_unwrapped_long_line_overflows_sides(tmp_path, brand):
    ass = write_ass(tmp_path, ["0:00:00.00,0:00:01.00,Caption,,0,0,0,,COUPLES WHO FEEL LIKE ROOMMATES"], size=(1080, 1080), wrap=2, margin_v=200)
    check = safezones.check_text_placement(gate_input(Path("x.mp4"), Aspect.SQUARE, ass), brand)
    assert not check.passed and "left" in check.detail and "right" in check.detail


def test_safe_zone_check_skips_without_subtitles(brand):
    check = safezones.check_text_placement(gate_input(Path("x.mp4")), brand)
    assert check.skipped and check.name == "safe_zones"


def test_decide_accepts_when_only_warnings_fail():
    reports = [report(Check(name="a", passed=True), Check(name="b", passed=False, severity="warn"))]
    assert gate.decide(reports, attempt=1, max_attempts=3) == "accept"


def test_decide_prefers_shot_retry_then_render_then_flag():
    shot = Check(name="identity_hook", passed=False, retry="shot")
    render = Check(name="duration", passed=False, retry="render")
    fixed = Check(name="policy_final", passed=False, retry="none")
    assert gate.decide([report(render), report(shot)], attempt=1, max_attempts=3) == "retry_shot"
    assert gate.decide([report(render)], attempt=2, max_attempts=3) == "retry_render"
    assert gate.decide([report(render)], attempt=3, max_attempts=3) == "flag"
    assert gate.decide([report(fixed)], attempt=1, max_attempts=3) == "flag"
    assert gate.decide([], attempt=1, max_attempts=3) == "flag"


def test_evaluate_isolates_crashes_and_blocks_on_policy(clips, brand, tmp_path, monkeypatch):
    def explode(inp, brand):
        raise RuntimeError("probe exploded")

    def lint(text, brand):
        if "Betterly" in text:
            return [PolicyFlag(rule="personal_attributes", severity="block", span="your marriage", message="implies attributes")]
        return []

    monkeypatch.setattr(technical, "check_render", explode)
    monkeypatch.setattr("mitosis.intelligence.policy.lint", lint)
    hero = HeroPack(id="hero-x", description="man")
    result = gate.evaluate(gate_input(clips["good"]), variant_spec(), hero, brand, make_settings(tmp_path))
    checks = by_name(result.checks)
    assert checks["qa_error"].severity == "warn" and "probe exploded" in checks["qa_error"].detail
    assert not checks["policy_final"].passed and "personal_attributes" in checks["policy_final"].detail
    assert checks["vision_judge"].skipped
    assert [c.name for c in result.blocking] == ["policy_final"]
    assert gate.decide([result], attempt=1, max_attempts=3) == "flag"


def test_evaluate_passes_clean_render_and_skips_missing_policy(clips, brand, tmp_path, monkeypatch):
    def unavailable(text, brand):
        raise NotImplementedError

    monkeypatch.setattr("mitosis.intelligence.policy.lint", unavailable)
    ass = write_ass(tmp_path, ["0:00:00.00,0:00:01.00,Caption,,0,0,0,,TRY IT FREE"], size=(1080, 1080), margin_v=150)
    result = gate.evaluate(gate_input(clips["good"], subtitles=ass), variant_spec(), HeroPack(id="h", description="d"), brand, make_settings(tmp_path))
    assert result.passed, [c for c in result.checks if not c.passed and not c.skipped]
    assert by_name(result.checks)["policy_final"].skipped
    assert gate.decide([result], attempt=1, max_attempts=3) == "accept"


class FakeJudge:
    def __init__(self, data=None, error: Exception | None = None) -> None:
        self.data = data
        self.error = error
        self.calls: list[dict] = []

    def json(self, system, prompt, schema, images=None, max_tokens=8000):
        self.calls.append({"system": system, "prompt": prompt, "schema": schema, "images": list(images or []), "exists": [Path(p).is_file() for p in images or []]})
        if self.error:
            raise self.error
        return GenResult(data=self.data)


def verdict(**overrides) -> dict:
    data = {
        "artifacts": False,
        "artifact_detail": "",
        "same_person": 0.9,
        "text_legible": True,
        "brand_safe": True,
        "policy_risk": "none",
        "overall": 0.8,
        "notes": "Clean UGC frames.",
    }
    data.update(overrides)
    return data


def hero_with_reference(tmp_path: Path) -> HeroPack:
    reference = tmp_path / "ref_01.jpg"
    cv2.imwrite(str(reference), np.full((64, 64, 3), 128, dtype=np.uint8))
    return HeroPack(id="hero-x", description="man", references=[reference])


def test_judge_without_model_is_a_skipped_warning(clips):
    checks = judge.judge(gate_input(clips["good"]), variant_spec(), HeroPack(id="h", description="d"), llm=None)
    assert len(checks) == 1 and checks[0].name == "vision_judge" and checks[0].skipped and checks[0].severity == "warn"


def test_judge_sends_reference_first_and_maps_verdict(clips, tmp_path):
    fake = FakeJudge(verdict(artifacts=True, artifact_detail="extra finger in frame 2", same_person=0.4, policy_risk="Low"))
    hero = hero_with_reference(tmp_path)
    checks = by_name(judge.judge(gate_input(clips["good"]), variant_spec(), hero, llm=fake))
    call = fake.calls[0]
    assert call["images"][0] == hero.references[0] and len(call["images"]) == 5 and all(call["exists"])
    assert call["schema"]["additionalProperties"] is False
    assert set(call["schema"]["required"]) == set(call["schema"]["properties"])
    assert not checks["visual_artifacts"].passed and checks["visual_artifacts"].retry == "shot"
    assert not checks["hero_match_judge"].passed and checks["hero_match_judge"].severity == "warn"
    assert checks["brand_safety"].passed and checks["text_legibility"].passed


def test_judge_blocks_unsafe_frames(clips, tmp_path):
    fake = FakeJudge(verdict(brand_safe=False, policy_risk="high", notes="implied nudity"))
    checks = by_name(judge.judge(gate_input(clips["good"]), variant_spec(new_shot=False), hero_with_reference(tmp_path), llm=fake))
    assert not checks["brand_safety"].passed and checks["brand_safety"].severity == "block" and checks["brand_safety"].retry == "none"


def test_judge_failures_degrade_to_skipped(clips, tmp_path):
    for fake in (FakeJudge(error=ProviderError("all models failed")), FakeJudge(data={"artifacts": "maybe"})):
        checks = judge.judge(gate_input(clips["good"]), variant_spec(), hero_with_reference(tmp_path), llm=fake)
        assert [c.name for c in checks] == ["vision_judge"] and checks[0].skipped


def synthetic_frames(folder: Path, count: int = 4) -> list[tuple[float, Path]]:
    folder.mkdir(parents=True, exist_ok=True)
    frames: list[tuple[float, Path]] = []
    gradient = np.tile(np.linspace(0, 255, 540, dtype=np.uint8), (960, 1))
    for index in range(count):
        pixels = cv2.merge([gradient, np.flipud(gradient), np.full_like(gradient, 40 * index)])
        cv2.rectangle(pixels, (100 + index * 20, 200), (400, 600), (255, 255, 255), 6)
        if index % 2:
            pixels = cv2.GaussianBlur(pixels, (15, 15), 5)
        path = folder / f"frame {index:02d}.jpg"
        cv2.imwrite(str(path), pixels)
        frames.append((index * 2.0, path))
    return frames


def test_identity_degrades_when_models_cannot_be_fetched(tmp_path, monkeypatch):
    def offline(asset, directory):
        raise identity.ModelUnavailable("network unreachable")

    monkeypatch.setattr(identity, "fetch_model", offline)
    settings = make_settings(tmp_path)
    frames = synthetic_frames(tmp_path / "work demo" / "samples")
    assert identity.face_box(frames[0][1], settings) is None
    hero = identity.build_hero(frames, tmp_path / "work demo" / "hero", "a man in his thirties", 7, settings)
    assert hero.embedding is None and hero.embedding_model is None
    assert [p.name for p in hero.references] == ["ref_01.jpg", "ref_02.jpg", "ref_03.jpg"]
    assert all(p.is_file() for p in hero.references)
    assert hero.id.startswith("hero-") and "different person" in hero.negative
    locked = HeroPack(id="hero-x", description="d", embedding=[1.0] + [0.0] * 127, embedding_model=identity.EMBEDDING_MODEL)
    keyframe = identity.check_image(frames[0][1], locked, 0.45, settings)
    assert keyframe.skipped and keyframe.name == "identity_keyframe" and "network unreachable" in keyframe.detail
    clip = identity.check_video(frames[0][1], locked, 0.42, settings)
    assert clip.skipped and clip.name == "identity_hook"
    assert identity.check_image(frames[0][1], hero, 0.45, settings).skipped


def test_identity_threshold_logic_flags_flicker_and_missing_faces(tmp_path, monkeypatch):
    readings = iter([[(0.2, 0.7), (0.6, 0.68), (1.0, 0.25), (1.4, 0.71)], [(0.2, 0.7), (0.6, None), (1.0, None), (1.4, 0.72)]])

    class ReadyEngine:
        available = True
        failure = ""

    monkeypatch.setattr(identity, "face_engine", lambda settings: ReadyEngine())
    monkeypatch.setattr(identity, "video_duration", lambda video: 2.0)
    monkeypatch.setattr(identity, "frame_readings", lambda engine, video, times, anchor: next(readings))
    hero = HeroPack(id="hero-x", description="d", embedding=[1.0] + [0.0] * 127, embedding_model=identity.EMBEDDING_MODEL)
    flicker = identity.check_video(tmp_path / "clip.mp4", hero, 0.42, make_settings(tmp_path))
    assert not flicker.passed and "flicker at 1.00s" in flicker.detail and flicker.retry == "shot"
    assert flicker.score == pytest.approx(0.69, abs=0.01)
    missing = identity.check_video(tmp_path / "clip.mp4", hero, 0.42, make_settings(tmp_path))
    assert not missing.passed and "no face in 2/4" in missing.detail


@pytest.fixture
def live_settings(tmp_path) -> Settings:
    settings = make_settings(tmp_path, SHARED_MODELS)
    if not identity.face_engine(settings).available:
        pytest.skip("face models could not be downloaded")
    return settings


def test_engine_finds_no_face_in_synthetic_image(live_settings, tmp_path):
    frames = synthetic_frames(tmp_path / "plain")
    assert identity.face_box(frames[0][1], live_settings) is None
    hero = HeroPack(id="hero-x", description="d", embedding=[1.0] + [0.0] * 127, embedding_model=identity.EMBEDDING_MODEL)
    check = identity.check_image(frames[0][1], hero, 0.45, live_settings)
    assert not check.passed and not check.skipped and check.detail == "no face detected in keyframe"


@pytest.mark.skipif(not SAMPLE_WINNER.is_file(), reason="sample winner not present")
def test_hero_lock_on_sample_winner(live_settings, tmp_path):
    samples = tmp_path / "work demo" / "samples"
    samples.mkdir(parents=True)
    subprocess.run(
        [ffmpeg(), "-v", "error", "-i", str(SAMPLE_WINNER), "-vf", "fps=0.5,scale=960:-2", str(samples / "f_%03d.jpg")],
        check=True,
        capture_output=True,
        timeout=120,
    )
    frames = [(index * 2.0, path) for index, path in enumerate(sorted(samples.glob("f_*.jpg")))]
    hero = identity.build_hero(frames, tmp_path / "work demo" / "hero", "bearded man", 7, live_settings)
    assert hero.embedding is not None and len(hero.embedding) == 128
    assert np.linalg.norm(hero.embedding) == pytest.approx(1.0, abs=1e-3)
    assert hero.embedding_model == identity.EMBEDDING_MODEL
    assert min(cv2.imread(str(hero.references[1])).shape[:2]) >= 768
    box = identity.face_box(hero.references[0], live_settings)
    assert box is not None and 0.0 <= box.x <= 1.0 and 0.0 < box.w <= 1.0
    assert identity.check_image(hero.references[0], hero, 0.45, live_settings).passed
    hook = identity.check_video(SAMPLE_WINNER, hero, 0.42, live_settings, 0.0, 3.0)
    assert hook.passed, hook.detail
