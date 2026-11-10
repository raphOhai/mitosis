from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mitosis.cache import Cache
from mitosis.config import ModelSpec, Settings, load_brief
from mitosis.ledger import Ledger
from mitosis.models import Aspect, Check, MediaInfo, QAReport
from mitosis.providers.base import FACTORIES, GenResult, PermanentError, Provider, TransientError, register
from mitosis.registry import Breaker, Registry

CALLS: dict[str, int] = {}


@register("fake_ok", "image")
class FakeImage(Provider):
    def generate(self, prompt, references, aspect, seed, out, negative=""):
        CALLS[self.spec.id] = CALLS.get(self.spec.id, 0) + 1
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(f"{prompt}-{seed}".encode())
        return GenResult(path=out, usage={"image": 1})


@register("fake_flaky", "image")
class FlakyImage(Provider):
    def generate(self, prompt, references, aspect, seed, out, negative=""):
        CALLS[self.spec.id] = CALLS.get(self.spec.id, 0) + 1
        raise TransientError("503 upstream busy")


@register("fake_refuse", "image")
class RefusingImage(Provider):
    def generate(self, prompt, references, aspect, seed, out, negative=""):
        CALLS[self.spec.id] = CALLS.get(self.spec.id, 0) + 1
        raise PermanentError("content policy")


def make_settings(tmp_path: Path, models: dict, chains: dict) -> Settings:
    config = tmp_path / "config"
    config.mkdir()
    (config / "models.yaml").write_text(yaml.safe_dump({"chains": chains, "models": models}))
    (config / "brand.yaml").write_text(yaml.safe_dump({"name": "B", "product": "B", "category": "c", "audience": "a", "cta": "Try", "tone": "t"}))
    return Settings.load(root=tmp_path, models_dir=tmp_path / "models")


@pytest.fixture(autouse=True)
def reset_calls():
    CALLS.clear()


def test_model_spec_snaps_to_supported_durations():
    spec = ModelSpec(id="v", provider="p", kind="video", durations=[5, 10])
    assert spec.snap(3.2) == 5
    assert spec.snap(5.0) == 5
    assert spec.snap(7.5) == 10
    assert spec.snap(14) == 10


def test_model_spec_prices_usage_units():
    spec = ModelSpec(id="llm", provider="p", kind="llm", price={"input_mtok": 4.0, "output_mtok": 20.0})
    assert spec.cost({"input_mtok": 0.01, "output_mtok": 0.002}) == pytest.approx(0.08)


def test_cache_round_trip_links_file(tmp_path):
    cache = Cache(tmp_path / "cache")
    source = tmp_path / "a.png"
    source.write_bytes(b"pixels")
    key = cache.key("image", {"seed": 1, "ref": source, "aspect": Aspect.VERTICAL})
    cache.put(key, GenResult(path=source, usage={"image": 1}, shadow_usd=0.15))
    target = tmp_path / "out" / "b.png"
    hit = cache.get(key, target)
    assert hit is not None and hit.cached
    assert target.read_bytes() == b"pixels"
    assert hit.shadow_usd == 0.15


def test_cache_key_tracks_file_content(tmp_path):
    cache = Cache(tmp_path / "cache")
    source = tmp_path / "ref.jpg"
    source.write_bytes(b"one")
    first = cache.key("image", source)
    source.write_bytes(b"two!")
    assert cache.key("image", source) != first


def test_ledger_scopes_and_summary(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    with ledger.scope(variant="V01", stage="keyframe"):
        ledger.record(kind="image", provider="fal", model="m", usd=0.15, shadow_usd=0.15)
        with ledger.scope(stage="animate", attempt=2):
            ledger.record(kind="video", provider="fal", model="k", usd=0.35, shadow_usd=0.35)
    ledger.record(kind="image", provider="fal", model="m", cached=True, saved_usd=0.15)
    rows = Ledger.read(tmp_path / "ledger.jsonl")
    assert [r.stage for r in rows] == ["keyframe", "animate", None]
    assert rows[1].attempt == 2
    assert ledger.for_variant("V01")[0] == pytest.approx(0.5)
    summary = ledger.summary()
    assert summary["cache_hits"] == 1
    assert summary["saved_usd"] == pytest.approx(0.15)


def test_breaker_opens_after_threshold_and_recovers():
    breaker = Breaker(threshold=2, cooldown=0.0)
    breaker.failure()
    assert breaker.closed
    breaker.failure()
    assert breaker.closed
    breaker.success()
    assert breaker.failures == 0


def test_registry_falls_back_and_caches(tmp_path, monkeypatch):
    models = {
        "flaky": {"provider": "fake_flaky", "kind": "image", "price": {"image": 0.2}, "params": {"retries": 1}},
        "refuse": {"provider": "fake_refuse", "kind": "image", "price": {"image": 0.1}},
        "good": {"provider": "fake_ok", "kind": "image", "price": {"image": 0.05}},
    }
    settings = make_settings(tmp_path, models, {"image": ["flaky", "refuse", "good"]})
    monkeypatch.setattr("mitosis.registry.time.sleep", lambda _: None)
    ledger = Ledger(tmp_path / "ledger.jsonl")
    registry = Registry(settings, ledger, Cache(tmp_path / "cache"))
    out = tmp_path / "shot.png"
    result = registry.image.generate("hero", [], Aspect.VERTICAL, 3, out)
    assert result.model == "good" and result.usd == pytest.approx(0.05)
    assert CALLS == {"flaky": 2, "refuse": 1, "good": 1}
    again = registry.image.generate("hero", [], Aspect.VERTICAL, 3, tmp_path / "again.png")
    assert again.cached and CALLS["good"] == 1
    failures = [e for e in ledger.entries if not e.ok]
    assert len(failures) == 3
    assert registry.mode == "live"


def test_registry_prefers_live_and_hides_offline(tmp_path, monkeypatch):
    models = {
        "good": {"provider": "fake_ok", "kind": "image", "price": {"image": 0.05}, "requires": ["FAKE_KEY"]},
        "local": {"provider": "fake_ok", "kind": "image", "offline": True},
    }
    settings = make_settings(tmp_path, models, {"image": ["good", "local"]})
    monkeypatch.delenv("FAKE_KEY", raising=False)
    registry = Registry(settings, Ledger(), Cache(tmp_path / "cache"))
    assert registry.image.names == ["local"]
    result = registry.image.generate("x", [], Aspect.SQUARE, 1, tmp_path / "x.png")
    assert result.usd == 0 and result.shadow_usd == pytest.approx(0.05)
    monkeypatch.setenv("FAKE_KEY", "k")
    live = Registry(settings, Ledger(), Cache(tmp_path / "cache2"))
    assert live.image.names == ["good"]
    forced = Registry(settings, Ledger(), Cache(tmp_path / "cache3"), offline=True)
    assert forced.image.names == ["local"]


def test_qa_report_passes_on_warnings_and_skips():
    report = QAReport(
        variant_id="V01",
        aspect=Aspect.VERTICAL,
        checks=[
            Check(name="a", passed=True),
            Check(name="b", passed=False, severity="warn"),
            Check(name="c", passed=False, skipped=True),
        ],
    )
    assert report.passed
    report.checks.append(Check(name="d", passed=False, retry="shot"))
    assert not report.passed and [c.name for c in report.blocking] == ["d"]


def test_load_brief_resolves_relative_paths(tmp_path):
    (tmp_path / "media").mkdir()
    (tmp_path / "media" / "w.mp4").write_bytes(b"")
    (tmp_path / "media" / "script.txt").write_text("Hello there. Try it free.")
    brief_file = tmp_path / "briefs" / "b.yaml"
    brief_file.parent.mkdir()
    brief_file.write_text(
        yaml.safe_dump(
            {"winner": {"id": "W1", "video": "../media/w.mp4", "script": "../media/script.txt"}, "aspects": ["9:16", "1:1"], "variants": 4}
        )
    )
    brief = load_brief(brief_file)
    assert brief.script == "Hello there. Try it free."
    assert brief.video == (tmp_path / "media" / "w.mp4").resolve()
    assert brief.aspects == [Aspect.VERTICAL, Aspect.SQUARE]


def test_shipped_config_is_consistent():
    settings = Settings.load()
    catalog = settings.catalog
    for kind, names in catalog.chains.items():
        for name in names:
            assert name in catalog.models, name
            assert catalog.models[name].kind == kind
    assert set(settings.voices) and set(settings.locations)
    assert any(location.original for location in settings.locations.values())
    assert settings.brand.safe_zone(Aspect.VERTICAL).bottom >= 0.3


def test_hook_shot_withholds_video_when_no_keyframe_matches(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from mitosis import pipeline as pipe
    from mitosis.events import EventBus

    calls = {"image": 0}

    def generate(prompt, references, aspect, seed, out, negative=""):
        calls["image"] += 1
        return GenResult(path=out, usage={"image": 1})

    def animate(*args, **kwargs):
        raise AssertionError("video must not be generated for an unmatched hero")

    rejected = Check(name="identity_keyframe", passed=False, score=0.2, threshold=0.45, retry="shot")
    monkeypatch.setattr(pipe.identity, "check_image", lambda *a, **k: rejected)
    monkeypatch.setattr(pipe.Pipeline, "shot_prompt", lambda self, run, spec: "hero in kitchen")
    events = []
    bus = EventBus()
    bus.subscribe(events.append)
    runner = pipe.Pipeline(Settings.load(), bus)
    info = MediaInfo(path=tmp_path / "w.mp4", width=1080, height=1920, fps=30.0, duration=25.0, has_audio=True)
    winner = SimpleNamespace(video=info.path, hook_end=1.4, info=info, shots=[], talking_head=True)
    run = SimpleNamespace(
        winner=winner,
        hero=SimpleNamespace(references=[], negative=""),
        ledger=Ledger(),
        registry=SimpleNamespace(image=SimpleNamespace(generate=generate), video=SimpleNamespace(animate=animate)),
    )
    spec = SimpleNamespace(id="V01", new_shot=True)
    segment, checks = runner.hook_shot(run, spec, tmp_path, 17, 3.0, tmp_path / "hook.wav")
    assert calls["image"] == 3
    assert segment.path == winner.video and segment.reframe == "shots"
    assert len(checks) == 3 and all(c.retry == "shot" and not c.passed for c in checks)
    assert any(e.status == "shot_withheld" for e in events)
