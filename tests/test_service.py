from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import mitosis.pipeline as pipeline_module
from mitosis.config import Settings
from mitosis.events import EventBus
from mitosis.intelligence import priors as priors_module
from mitosis.models import (
    Aspect,
    BetaArm,
    Brief,
    Check,
    CostEstimate,
    CreativeDNA,
    HeroPack,
    Hook,
    Location,
    MediaInfo,
    PolicyFlag,
    Prelaunch,
    Priors,
    QAReport,
    Render,
    RunManifest,
    Totals,
    VariantPlan,
    VariantResult,
    VariantSpec,
    Voice,
    Winner,
)
from mitosis.service.api import create_app, normalize_archetypes
from mitosis.service.jobs import JobRecord, initial_stages

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPARE_STAGES = ("ingest", "understand", "hero", "hooks", "plan")
SCRIPT = "I used to think we were fine. Then she stopped telling me things. Betterly gave us ten minutes a night that changed it."
RENDER_BYTES = bytes(range(256)) * 4


@dataclass
class FakeBehaviour:
    release: threading.Event | None = None
    failure: str | None = None
    briefs: list[Brief] = field(default_factory=list)


def write_file(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def accepted_hook() -> Hook:
    return Hook(
        id="h1",
        archetype="confession",
        text="I almost lost her over something this small.",
        on_screen_text="I almost lost her",
        visual="looks into the lens, exhales",
        motion="slow push in",
        score=0.82,
    )


def blocked_hook() -> Hook:
    return Hook(
        id="h9",
        archetype="callout",
        text="Is your sex life failing?",
        on_screen_text="Failing in bed?",
        visual="",
        motion="",
        flags=[PolicyFlag(rule="personal_attributes", severity="block", message="Asserts a personal attribute of the viewer")],
    )


def build_result(out_dir: Path, index: int) -> VariantResult:
    variant_id = f"v{index:02d}"
    spec = VariantSpec(
        id=variant_id,
        index=index,
        hook=accepted_hook(),
        voice=Voice(id="warm_30s", label="Warm 30s, reassuring", tone="warm, steady"),
        location=Location(id="parked_car", label="Parked car", prompt="parked car"),
        aspects=[Aspect.VERTICAL],
        seed=7 + index,
        changed=["hook", "voice"],
    )
    renders = out_dir / "variants" / variant_id / "renders"
    video = write_file(renders / f"{variant_id}_9x16.mp4", RENDER_BYTES)
    thumbnail = write_file(renders / f"{variant_id}_9x16.jpg", b"\xff\xd8\xff\xd9")
    passed = index == 1
    check = Check(name="identity", passed=passed, score=0.61 if passed else 0.31, threshold=0.42, detail="hero similarity")
    render = Render(
        aspect=Aspect.VERTICAL,
        path=video,
        thumbnail=thumbnail,
        duration=19.5,
        qa=QAReport(variant_id=variant_id, aspect=Aspect.VERTICAL, checks=[check]),
        name=f"{variant_id}_9x16",
    )
    return VariantResult(
        spec=spec,
        renders=[render],
        status="passed" if passed else "flagged",
        attempts=index,
        usd=0.0,
        shadow_usd=1.55,
        seconds=4.2,
        prelaunch=Prelaunch(score=0.71, reasons=["hook lands inside 2 seconds"]),
    )


def build_manifest(out_dir: Path, run_id: str, brief: Brief, created_at: str = "2026-01-01T09:00:00+00:00") -> RunManifest:
    results = [build_result(out_dir, 1), build_result(out_dir, 2)]
    reference = write_file(out_dir / "hero" / "ref_00.jpg", b"\xff\xd8\xff\xd9")
    exports = out_dir / "exports"
    return RunManifest(
        run_id=run_id,
        created_at=created_at,
        out_dir=out_dir,
        brief=brief,
        winner=Winner(
            id=brief.winner_id,
            video=brief.video,
            script=brief.script,
            info=MediaInfo(path=brief.video, width=1080, height=1920, fps=30.0, duration=21.0, has_audio=True),
            audio=out_dir / "ingest" / "audio.wav",
            hook_end=2.4,
            cta_start=17.0,
        ),
        dna=CreativeDNA(
            angle="quiet distance",
            audience="men in long relationships",
            promise="ten minutes a night",
            hook_archetype="confession",
            hook_text="I used to think we were fine.",
            body_text="Then she stopped telling me things.",
            cta_text="Try Betterly free",
            hero_description="man in his 30s, short beard, grey hoodie",
            setting="living room",
            tone="warm",
        ),
        hero=HeroPack(id="hero", description="man in his 30s", references=[reference], embedding=[0.1, 0.2]),
        hooks=[accepted_hook()],
        rejected_hooks=[blocked_hook()],
        plan=VariantPlan(
            winner_id=brief.winner_id,
            design="fractional",
            variants=[result.spec for result in results],
            estimate=CostEstimate(usd=3.1, per_variant_usd=1.55),
        ),
        results=results,
        totals=Totals(variants=2, passed=1, flagged=1, renders=2, usd=0.0, shadow_usd=3.1, seconds=9.0),
        mode="offline",
        models={"video": ["offline_kenburns"]},
        report=write_file(out_dir / "report.html", b"<!doctype html><title>Report</title>"),
        exports={
            "bundle": write_file(exports / f"{run_id}.zip", b"PK\x05\x06" + b"\x00" * 18),
            "meta_csv": write_file(exports / "meta_bulk_upload.csv", b"Ad Name,Video\n"),
        },
    )


class FakePipeline:
    behaviour = FakeBehaviour()

    def __init__(self, settings: Settings, bus: EventBus) -> None:
        self.settings = settings
        self.bus = bus

    def run(self, brief: Brief, run_id: str) -> RunManifest:
        out_dir = Path(brief.out or self.settings.runs_dir / run_id).resolve()
        for stage in PREPARE_STAGES:
            with self.bus.stage(stage):
                self.bus.emit("log", stage=stage, message=f"{stage} finished")
        if self.behaviour.release is not None:
            self.behaviour.release.wait(5)
        if self.behaviour.failure:
            with self.bus.stage("produce"):
                raise RuntimeError(self.behaviour.failure)
        manifest = build_manifest(out_dir, run_id, brief)
        with self.bus.stage("produce", "2 variants x 1 formats"):
            for position, result in enumerate(manifest.results, start=1):
                self.bus.emit(
                    "variant",
                    variant=result.spec.id,
                    status=result.status,
                    progress=position / len(manifest.results),
                    message=result.spec.hook.text,
                    data={"usd": 0.0, "shadow_usd": 1.55, "seconds": 4.2, "attempts": 1, "score": 0.71, "error": None},
                )
        with self.bus.stage("report"):
            pipeline_module.save_manifest(manifest)
        self.bus.emit("done", message=str(manifest.report), data=manifest.totals.model_dump())
        return manifest


def fake_estimate(brief: Brief, settings: Settings | None = None, bus: EventBus | None = None) -> dict[str, Any]:
    FakePipeline.behaviour.briefs.append(brief)
    for stage in PREPARE_STAGES:
        with bus.stage(stage):
            pass
    plan = VariantPlan(winner_id=brief.winner_id, design="fractional", variants=[], estimate=CostEstimate(usd=4.5, per_variant_usd=1.5))
    return {
        "run": "ignored",
        "out": str(brief.out),
        "mode": "offline",
        "plan": plan.model_dump(mode="json"),
        "hooks": [accepted_hook().model_dump(mode="json")],
        "rejected": [blocked_hook().model_dump(mode="json")],
    }


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        root=PROJECT_ROOT,
        config_dir=PROJECT_ROOT / "config",
        runs_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        models_dir=tmp_path / "models",
        priors_path=tmp_path / "priors.json",
        workers=1,
        offline=True,
    )


@pytest.fixture
def behaviour(monkeypatch: pytest.MonkeyPatch) -> FakeBehaviour:
    current = FakeBehaviour()
    monkeypatch.setattr(FakePipeline, "behaviour", current)
    monkeypatch.setattr(pipeline_module, "Pipeline", FakePipeline)
    monkeypatch.setattr(pipeline_module, "estimate", fake_estimate)
    return current


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def submit_run(client: TestClient, **overrides: str) -> dict[str, Any]:
    form = {
        "script": SCRIPT,
        "winner_id": "W01",
        "variants": "2",
        "aspects": "9:16",
        "voices": "warm_30s",
        "locations": "parked_car",
        **overrides,
    }
    files = {"video": ("winner.mp4", io.BytesIO(b"\x00" * 64), "video/mp4")}
    response = client.post("/api/runs", data=form, files=files)
    assert response.status_code == 202, response.text
    return response.json()


def wait_for_finish(client: TestClient, run_id: str, timeout: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        detail = client.get(f"/api/runs/{run_id}").json()
        if detail["status"] in {"done", "error"}:
            return detail
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish in {timeout}s")


def read_stream(client: TestClient, run_id: str, headers: dict[str, str] | None = None) -> str:
    with client.stream("GET", f"/api/runs/{run_id}/events", headers=headers or {}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        return "".join(response.iter_text())


def parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    messages: list[tuple[str, dict[str, Any]]] = []
    for block in text.split("\n\n"):
        lines = block.splitlines()
        data = [line[5:].strip() for line in lines if line.startswith("data:")]
        if not data:
            continue
        name = next((line[6:].strip() for line in lines if line.startswith("event:")), "message")
        messages.append((name, json.loads("".join(data))))
    return messages


def seed_finished_run(settings: Settings, run_id: str = "20260101-090000-w01") -> tuple[str, Path]:
    run_dir = (settings.runs_dir / run_id).resolve()
    video = write_file(settings.runs_dir / "uploads" / run_id / "winner.mp4", b"\x00" * 64)
    brief = Brief(winner_id="w01", video=video, script=SCRIPT, variants=2, aspects=[Aspect.VERTICAL])
    pipeline_module.save_manifest(build_manifest(run_dir, run_id, brief))
    return run_id, run_dir


def test_import_is_fast_and_side_effect_free(tmp_path: Path) -> None:
    code = (
        "import json, sys, time\n"
        "started = time.perf_counter()\n"
        "import mitosis.service.api\n"
        "elapsed = time.perf_counter() - started\n"
        "heavy = [name for name in ('mitosis.pipeline', 'mitosis.registry', 'cv2', 'numpy') if name in sys.modules]\n"
        "print(json.dumps({'elapsed': elapsed, 'heavy': heavy}))\n"
    )
    environment = {**os.environ, "MITOSIS_HOME": str(tmp_path)}
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, env=environment, capture_output=True, text=True, timeout=60, check=True
    )
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["heavy"] == []
    assert report["elapsed"] < 2.0
    assert list(tmp_path.iterdir()) == []


def test_index_and_static_assets_are_served(client: TestClient) -> None:
    index = client.get("/")
    assert index.status_code == 200
    assert index.headers["content-type"].startswith("text/html")
    assert "Mitosis" in index.text
    for asset in ("/static/app.js", "/static/styles.css"):
        assert client.get(asset).status_code == 200


def test_health_reports_mode_chains_and_ffmpeg(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["mode"] in {"offline", "hybrid", "live"}
    assert "video" in body["chains"]
    assert isinstance(body["ffmpeg"], bool)
    assert body["jobs"] == {"queued": 0, "running": 0, "done": 0, "error": 0}


def test_config_lists_brand_voices_locations_and_formats(client: TestClient) -> None:
    body = client.get("/api/config").json()
    assert body["brand"]["name"] == "Betterly"
    assert {"id", "label", "tone"} <= set(body["voices"][0])
    assert "warm_30s" in {voice["id"] for voice in body["voices"]}
    assert "parked_car" in {place["id"] for place in body["locations"]}
    assert [aspect["value"] for aspect in body["aspects"]] == ["9:16", "4:5", "1:1"]
    assert isinstance(body["archetypes"], list)
    assert body["defaults"]["variants"] == 10


def test_archetypes_are_normalized_from_any_shape() -> None:
    assert normalize_archetypes(["pattern_interrupt"]) == [{"id": "pattern_interrupt", "label": "Pattern interrupt", "description": ""}]
    assert normalize_archetypes({"confession": "I have to admit..."})[0]["description"] == "I have to admit..."
    assert normalize_archetypes([{"id": "callout", "label": "Call-out"}])[0]["label"] == "Call-out"


def test_run_lifecycle_from_upload_to_manifest(client: TestClient, settings: Settings, behaviour: FakeBehaviour) -> None:
    created = submit_run(client)
    run_id = created["run_id"]
    assert re.fullmatch(r"\d{8}-\d{6}-w01", run_id)
    assert created["kind"] == "run"
    assert (settings.runs_dir / "uploads" / run_id / "winner.mp4").read_bytes() == b"\x00" * 64

    detail = wait_for_finish(client, run_id)
    assert detail["status"] == "done", detail["error"]
    assert detail["progress"] == 1.0
    assert [stage["status"] for stage in detail["stages"]] == ["done"] * 7
    assert detail["variants_done"] == 2 and detail["variant_progress"] == 1.0
    assert detail["brief"]["winner_id"] == "w01"

    manifest = detail["manifest"]
    assert manifest["totals"]["variants"] == 2
    assert manifest["rejected_hooks"][0]["flags"][0]["rule"] == "personal_attributes"
    assert manifest["hero"]["references"] == [f"/files/{run_id}/hero/ref_00.jpg"]
    first, second = manifest["results"]
    assert first["renders"][0]["url"] == f"/files/{run_id}/variants/v01/renders/v01_9x16.mp4"
    assert first["renders"][0]["thumbnail"] == f"/files/{run_id}/variants/v01/renders/v01_9x16.jpg"
    assert first["qa"]["passed"] is True
    assert second["qa"]["issues"][0]["name"] == "identity"
    assert manifest["links"]["bundle"] == f"/api/runs/{run_id}/bundle"
    assert manifest["links"]["report"] == f"/files/{run_id}/report.html"
    assert manifest["links"]["meta_csv"] == f"/files/{run_id}/exports/meta_bulk_upload.csv"

    persisted = json.loads((settings.runs_dir / run_id / "job.json").read_text())
    assert persisted["status"] == "done"
    assert persisted["kind"] == "run"
    assert client.get("/api/runs").json()["runs"][0]["run_id"] == run_id

    bundle = client.get(f"/api/runs/{run_id}/bundle")
    assert bundle.status_code == 200
    assert bundle.headers["content-type"] == "application/zip"
    assert f"{run_id}.zip" in bundle.headers["content-disposition"]


def test_dry_run_returns_plan_and_hooks(client: TestClient, settings: Settings, behaviour: FakeBehaviour) -> None:
    created = submit_run(client, dry_run="true")
    assert created["kind"] == "estimate"
    detail = wait_for_finish(client, created["run_id"])
    assert detail["status"] == "done"
    assert len(detail["stages"]) == 5
    assert detail["estimate"]["plan"]["estimate"]["usd"] == 4.5
    assert detail["estimate"]["hooks"][0]["archetype"] == "confession"
    assert detail["estimate"]["rejected_hooks"][0]["id"] == "h9"
    assert detail["events"][-1]["type"] == "done"
    assert behaviour.briefs[0].out == (settings.runs_dir / created["run_id"]).resolve()


def test_failed_job_reports_error_and_stage(client: TestClient, settings: Settings, behaviour: FakeBehaviour) -> None:
    behaviour.failure = "video provider timed out"
    created = submit_run(client)
    detail = wait_for_finish(client, created["run_id"])
    assert detail["status"] == "error"
    assert "timed out" in detail["error"]
    produce = next(stage for stage in detail["stages"] if stage["name"] == "produce")
    assert produce["status"] == "error"
    assert "manifest" not in detail
    assert json.loads((settings.runs_dir / created["run_id"] / "job.json").read_text())["status"] == "error"


def test_events_replay_history_then_close(client: TestClient, behaviour: FakeBehaviour) -> None:
    run_id = submit_run(client)["run_id"]
    wait_for_finish(client, run_id)
    messages = parse_sse(read_stream(client, run_id))
    assert messages[-1] == ("end", {"status": "done", "error": None})
    types = [data["type"] for name, data in messages if name == "message"]
    assert types.count("stage") == 14
    assert types.count("variant") == 2
    assert "done" in types

    resumed = parse_sse(read_stream(client, run_id, headers={"Last-Event-ID": "3"}))
    assert resumed[0][1]["seq"] == 4


def test_events_stream_follows_a_live_job(client: TestClient, behaviour: FakeBehaviour) -> None:
    behaviour.release = threading.Event()
    client.app.state.service.keepalive_seconds = 0.05
    run_id = submit_run(client)["run_id"]
    timer = threading.Timer(0.4, behaviour.release.set)
    timer.start()
    try:
        text = read_stream(client, run_id)
    finally:
        behaviour.release.set()
        timer.cancel()
    messages = parse_sse(text)
    assert ": keepalive" in text
    assert messages[-1] == ("end", {"status": "done", "error": None})
    sequences = [data["seq"] for name, data in messages if name == "message"]
    assert sequences == list(range(1, len(sequences) + 1))
    assert [data["status"] for name, data in messages if data.get("type") == "variant"] == ["passed", "flagged"]


def test_runs_list_merges_disk_manifests_and_interrupted_jobs(client: TestClient, settings: Settings) -> None:
    finished_id, _ = seed_finished_run(settings)
    interrupted_dir = settings.runs_dir / "20260102-090000-w02"
    interrupted_dir.mkdir(parents=True)
    record = JobRecord(
        run_id=interrupted_dir.name,
        kind="run",
        status="running",
        created_at="2026-01-02T09:00:00+00:00",
        run_dir=interrupted_dir,
        brief={"winner_id": "w02", "variants": 8},
        stages=initial_stages("run"),
    )
    (interrupted_dir / "job.json").write_text(record.model_dump_json())
    (settings.runs_dir / "uploads").mkdir(exist_ok=True)

    runs = client.get("/api/runs").json()["runs"]
    assert [run["run_id"] for run in runs] == [interrupted_dir.name, finished_id]
    assert runs[0]["status"] == "error"
    assert runs[1]["status"] == "done" and runs[1]["passed"] == 1
    assert client.get(f"/api/runs/{interrupted_dir.name}").json()["error"].startswith("The server stopped")
    assert client.get("/api/runs/uploads").status_code == 404
    assert client.get("/api/runs/does-not-exist").status_code == 404


def test_review_persists_into_manifest(client: TestClient, settings: Settings) -> None:
    run_id, run_dir = seed_finished_run(settings)
    response = client.post(f"/api/runs/{run_id}/variants/v02/review", json={"decision": "approved", "note": "Ship it"})
    assert response.status_code == 200, response.text
    variant = response.json()
    assert variant["review"] == "approved"
    assert variant["note"] == "Ship it"
    assert variant["renders"][0]["url"].startswith(f"/files/{run_id}/")

    stored = pipeline_module.load_manifest(run_dir / "manifest.json")
    assert {result.spec.id: result.review for result in stored.results} == {"v01": "pending", "v02": "approved"}

    rejected = client.post(f"/api/runs/{run_id}/variants/v01/review", json={"decision": "rejected"})
    assert rejected.json()["review"] == "rejected"
    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["manifest"]["review"] == {"approved": 1, "rejected": 1, "pending": 0}
    assert detail["manifest"]["results"][1]["note"] == "Ship it"

    assert client.post(f"/api/runs/{run_id}/variants/v02/review", json={"decision": "maybe"}).status_code == 422
    assert client.post(f"/api/runs/{run_id}/variants/v99/review", json={"decision": "approved"}).status_code == 404


def test_files_are_served_with_media_types_and_ranges(client: TestClient, settings: Settings) -> None:
    run_id, _ = seed_finished_run(settings)
    url = f"/files/{run_id}/variants/v01/renders/v01_9x16.mp4"
    full = client.get(url)
    assert full.status_code == 200
    assert full.headers["content-type"] == "video/mp4"
    assert full.headers["accept-ranges"] == "bytes"
    assert full.content == RENDER_BYTES

    partial = client.get(url, headers={"Range": "bytes=10-19"})
    assert partial.status_code == 206
    assert partial.content == RENDER_BYTES[10:20]
    assert partial.headers["content-range"] == f"bytes 10-19/{len(RENDER_BYTES)}"

    assert client.get(f"/files/{run_id}/variants/v01/renders/v01_9x16.jpg").headers["content-type"] == "image/jpeg"
    assert client.get(f"/files/{run_id}/report.html").headers["content-type"].startswith("text/html")
    assert client.get(f"/files/{run_id}/exports/meta_bulk_upload.csv").headers["content-type"].startswith("text/csv")
    assert client.get(f"/files/{run_id}/missing.mp4").status_code == 404


def test_files_reject_traversal_and_symlink_escapes(client: TestClient, settings: Settings, tmp_path: Path) -> None:
    run_id, run_dir = seed_finished_run(settings)
    secret = write_file(tmp_path / "secret.txt", b"top secret")
    (run_dir / "escape.txt").symlink_to(secret)
    (run_dir / "escape_dir").symlink_to(tmp_path, target_is_directory=True)
    for path in ("..%2F..%2Fsecret.txt", "%2e%2e/%2e%2e/secret.txt", "escape.txt", "escape_dir/secret.txt", "%2Fetc%2Fpasswd"):
        response = client.get(f"/files/{run_id}/{path}")
        assert response.status_code in {403, 404}, path
        assert "top secret" not in response.text
    assert client.get(f"/files/{run_id}/escape.txt").status_code == 403
    assert client.get(f"/files/uploads/{run_id}/winner.mp4").status_code == 404
    assert client.get("/files/unknown-run/report.html").status_code == 404


def test_run_creation_validates_input(client: TestClient, behaviour: FakeBehaviour) -> None:
    video = {"video": ("winner.mp4", io.BytesIO(b"\x00" * 16), "video/mp4")}
    base = {"script": SCRIPT}
    cases = [
        ({"aspects": "16:9"}, video, "Unknown format"),
        ({"voices": "robot"}, video, "Unknown voice"),
        ({"locations": "moon"}, video, "Unknown location"),
        ({"script": "   "}, video, "script"),
        ({}, {"video": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}, "video file"),
        ({}, {"video": ("winner.mp4", io.BytesIO(b""), "video/mp4")}, "empty"),
    ]
    for overrides, files, message in cases:
        for upload in files.values():
            upload[1].seek(0)
        response = client.post("/api/runs", data={**base, **overrides}, files=files)
        assert response.status_code == 422, overrides
        assert message in response.json()["detail"]
    assert client.post("/api/runs", data={}, files=video).status_code == 422
    assert client.get("/api/runs").json()["runs"] == []


def test_learn_updates_priors_from_csv(client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    saved: dict[str, Any] = {}
    learned = Priors(
        factors={
            "voice": {
                "deep_40s": BetaArm(alpha=2, beta=8, trials=10, wins=1),
                "warm_30s": BetaArm(alpha=8, beta=2, trials=10, wins=7),
            }
        },
        updated_at="2026-11-10T12:00:00+00:00",
    )

    def fake_learn(priors: Priors, results_csv: Path) -> Priors:
        assert results_csv.read_text().startswith("variant,ctr")
        return learned

    def fake_save(priors: Priors, path: Path) -> Path:
        saved.update(priors=priors, path=path)
        return path

    monkeypatch.setattr(priors_module, "load", lambda path: Priors())
    monkeypatch.setattr(priors_module, "learn", fake_learn)
    monkeypatch.setattr(priors_module, "save", fake_save)

    response = client.post("/api/learn", files={"file": ("results.csv", b"variant,ctr\nv01,0.031\n", "text/csv")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert [row["level"] for row in body["factors"]["voice"]] == ["warm_30s", "deep_40s"]
    assert body["levels"] == 2
    assert saved == {"priors": learned, "path": settings.priors_path}

    rejected = client.post("/api/learn", files={"file": ("clip.mp4", b"\x00", "video/mp4")})
    assert rejected.status_code == 422
