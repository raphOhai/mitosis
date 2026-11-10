from __future__ import annotations

import base64
import csv
import io
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from mitosis.config import PACKAGE_ROOT, Brand, read_yaml
from mitosis.models import (
    Aspect,
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
    QAReport,
    Render,
    RunManifest,
    Totals,
    VariantPlan,
    VariantResult,
    VariantSpec,
    Voice,
    Winner,
    Word,
)
from mitosis.report import export, html

FFMPEG = shutil.which("ffmpeg")
NODE = shutil.which("node")
WORKFLOWS = PACKAGE_ROOT / "workflows" / "n8n"
WORKFLOW_FILES = ("mitosis_winner_multiplier.json", "mitosis_learning_loop.json")
RUN_ID = "20261110-101500-W017"
ASSETS = (Aspect.VERTICAL, Aspect.PORTRAIT, Aspect.SQUARE)

HOOKS = {
    "H02": Hook(
        id="H02",
        archetype="confession",
        text="I almost lost her over something stupid & small",
        on_screen_text="I almost lost her",
        visual="sits in a parked car, looks at the camera",
        motion="slow push in",
        score=0.82,
    ),
    "H05": Hook(
        id="H05",
        archetype="curiosity",
        text="Nobody tells you this about year seven.",
        on_screen_text="Year 7 <b>secret</b>",
        visual="leans on the kitchen counter",
        motion="handheld drift",
        score=74.0,
        flags=[PolicyFlag(rule="soft_claim", severity="warn", message="Implies a universal outcome")],
    ),
}
VOICES = {
    "warm_30s": Voice(id="warm_30s", label="Warm 30s, reassuring"),
    "deep_40s": Voice(id="deep_40s", label="Deep 40s, grounded"),
}
LOCATIONS = {
    "parked_car": Location(id="parked_car", label="Parked car", prompt="driver seat of a parked car"),
    "original": Location(id="original", label="Original set", prompt="same room", original=True),
}
VARIANTS = (
    ("V01", "H02", "warm_30s", "parked_car", "passed", "pending"),
    ("V02", "H05", "deep_40s", "original", "flagged", "approved"),
    ("V03", "H02", "deep_40s", "parked_car", "passed", "rejected"),
    ("V04", "H05", "warm_30s", "parked_car", "flagged", "pending"),
    ("V05", "H02", "warm_30s", "original", "failed", "pending"),
)


@pytest.fixture(scope="session")
def media(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    folder = tmp_path_factory.mktemp("media")
    clip = folder / "clip.mp4"
    still = folder / "still.jpg"
    if FFMPEG:
        lavfi = [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i"]
        subprocess.run(lavfi + ["testsrc=size=64x112:rate=10:duration=0.5", "-pix_fmt", "yuv420p", str(clip)], check=True)
        subprocess.run(lavfi + ["color=c=gray:size=64x64", "-frames:v", "1", str(still)], check=True)
    else:
        clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        still.write_bytes(b"\xff\xd8\xff\xd9")
    return {"clip": clip, "still": still}


@pytest.fixture
def brand() -> Brand:
    return Brand(**read_yaml(PACKAGE_ROOT / "config" / "brand.yaml"))


def make_spec(variant_id: str, index: int, hook_id: str, voice_id: str, location_id: str) -> VariantSpec:
    changed = ["hook", "voice"] + (["location"] if location_id != "original" else [])
    return VariantSpec(
        id=variant_id,
        index=index,
        hook=HOOKS[hook_id],
        voice=VOICES[voice_id],
        location=LOCATIONS[location_id],
        aspects=list(ASSETS),
        seed=7 + index,
        new_shot=location_id != "original",
        changed=changed,
    )


def caption_words() -> list[Word]:
    script = "I almost lost her over something stupid. It was not the fight, it was the silence after. Try Betterly free."
    return [Word(text=token, start=round(i * 0.3, 2), end=round(i * 0.3 + 0.26, 2)) for i, token in enumerate(script.split())]


def qa_report(variant_id: str, aspect: Aspect, blocking_failure: bool) -> QAReport:
    return QAReport(
        variant_id=variant_id,
        aspect=aspect,
        checks=[
            Check(name="identity", passed=True, score=0.61, threshold=0.42, detail="same hero"),
            Check(name="safe_zones", passed=False, severity="warn", detail="caption touches the bottom zone"),
            Check(name="loudness", passed=not blocking_failure, detail="-13.8 LUFS" if not blocking_failure else "-19.2 LUFS"),
            Check(name="judge", passed=False, skipped=True, detail="no llm"),
        ],
    )


def make_renders(out_dir: Path, spec: VariantSpec, brand: Brand, media: dict[str, Path], status: str) -> list[Render]:
    folder = out_dir / "variants" / spec.id / "renders"
    folder.mkdir(parents=True, exist_ok=True)
    renders = []
    for aspect in spec.aspects:
        name = export.asset_name(brand, "W017", spec, aspect)
        video = folder / f"{name}.mp4"
        thumbnail = folder / f"{name}.jpg"
        shutil.copy(media["clip"], video)
        shutil.copy(media["still"], thumbnail)
        renders.append(
            Render(
                aspect=aspect,
                path=video,
                thumbnail=thumbnail,
                duration=10.4,
                qa=qa_report(spec.id, aspect, status == "flagged"),
                name=name,
            )
        )
    return renders


def make_result(out_dir: Path, brand: Brand, media: dict[str, Path], row: tuple[str, ...], index: int) -> VariantResult:
    variant_id, hook_id, voice_id, location_id, status, review = row
    spec = make_spec(variant_id, index, hook_id, voice_id, location_id)
    if status == "failed":
        return VariantResult(spec=spec, status="failed", attempts=3, error="ProviderError: all models failed for 'video'")
    return VariantResult(
        spec=spec,
        renders=make_renders(out_dir, spec, brand, media, status),
        status=status,
        attempts=1 if status == "passed" else 3,
        usd=0.0,
        shadow_usd=1.42,
        seconds=48.3,
        prelaunch=Prelaunch(score=0.71, reasons=["hook lands in 1.9 s", "face visible in first frame"]),
        review=review,
        hook_seconds=2.1,
        words=caption_words(),
    )


@pytest.fixture
def manifest(tmp_path: Path, brand: Brand, media: dict[str, Path]) -> RunManifest:
    out_dir = tmp_path / "runs" / RUN_ID
    hero_dir = out_dir / "hero"
    hero_dir.mkdir(parents=True)
    references = []
    for index in range(2):
        reference = hero_dir / f"ref_{index:02d}.jpg"
        shutil.copy(media["still"], reference)
        references.append(reference)
    winner_video = tmp_path / "samples" / "W017.mp4"
    winner_video.parent.mkdir(parents=True)
    shutil.copy(media["clip"], winner_video)
    results = [make_result(out_dir, brand, media, row, index) for index, row in enumerate(VARIANTS, start=1)]
    ledger = out_dir / "ledger.jsonl"
    ledger.write_text('{"ts": 1.0, "kind": "video", "provider": "offline", "model": "offline_kenburns"}\n')
    built = RunManifest(
        run_id=RUN_ID,
        created_at="2026-11-10T10:15:00+00:00",
        out_dir=out_dir,
        brief=Brief(winner_id="W017", video=winner_video, script="I almost lost her over something stupid."),
        winner=Winner(
            id="W017",
            video=winner_video,
            script="I almost lost her over something stupid.",
            info=MediaInfo(path=winner_video, width=1080, height=1920, fps=30.0, duration=24.3, has_audio=True),
            audio=out_dir / "ingest" / "audio.wav",
            hook_end=2.4,
        ),
        dna=CreativeDNA(
            angle="Confessional near-miss",
            audience="US men 28-45 in long-term relationships",
            promise="Fix the distance before it becomes a breakup",
            hook_archetype="confession",
            hook_text="I almost lost her over something stupid.",
            body_text="It was not the fight, it was the silence after.",
            cta_text="Try Betterly free",
            beats=["confession", "turn", "proof", "cta"],
            hero_description="man in his 30s, short beard, grey hoodie",
            setting="living room",
            tone="honest",
            format="talking_head",
        ),
        hero=HeroPack(
            id="hero",
            description="Man in his 30s, short beard, grey hoodie",
            references=references,
            embedding=[0.1, 0.2],
            embedding_model="arcface",
        ),
        hooks=list(HOOKS.values()),
        rejected_hooks=[
            Hook(
                id="H09",
                archetype="fear",
                text="Your marriage is failing and you know it.",
                on_screen_text="Is your marriage failing?",
                visual="close up",
                motion="static",
                flags=[
                    PolicyFlag(
                        rule="personal_attributes",
                        severity="block",
                        span="Your marriage is failing",
                        message="Asserts a personal attribute of the viewer",
                        suggestion="Speak from your own experience: I thought my marriage was failing.",
                    )
                ],
            )
        ],
        plan=VariantPlan(
            winner_id="W017",
            design="greedy_d_optimal",
            variants=[result.spec for result in results],
            levels={"hook": ["H02", "H05"], "voice": list(VOICES), "location": list(LOCATIONS)},
            coverage={"hook": 1.0, "pairs": 0.67},
            estimate=CostEstimate(usd=7.1, per_variant_usd=1.42),
            notes=["Two hooks carry the plan; voices rotate across them."],
        ),
        results=results,
        totals=Totals(
            variants=5,
            passed=2,
            flagged=2,
            failed=1,
            renders=12,
            usd=0.0,
            shadow_usd=7.1,
            seconds=412.0,
            usd_per_variant=0.0,
            shadow_per_variant=1.42,
            seconds_per_variant=82.4,
            cache_hits=6,
            saved_usd=1.8,
            by_stage={"animate": 3.5, "keyframe": 1.2, "voice": 1.9, "render": 0.0},
            by_provider={"offline:offline_kenburns": 3.5, "offline:offline_still": 1.2, "offline:offline_say": 1.9},
        ),
        mode="offline",
        models={"image": ["offline_still"], "video": ["offline_kenburns"], "speech": ["offline_say"], "llm": []},
        ledger=ledger,
    )
    (out_dir / "manifest.json").write_text(built.model_dump_json(indent=2))
    return built


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_asset_name_matches_contract(brand: Brand) -> None:
    spec = make_spec("V03", 3, "H02", "warm_30s", "parked_car")
    assert export.asset_name(brand, "W017", spec, Aspect.VERTICAL) == "Betterly_W017_V03_H02-confession_warm-30s_parked-car_9x16"


def test_asset_name_sanitizes_every_token(brand: Brand) -> None:
    spec = make_spec("V01", 1, "H02", "warm_30s", "parked_car").model_copy(
        update={
            "id": "V 07",
            "hook": HOOKS["H02"].model_copy(update={"id": "H_09", "archetype": "us vs. them"}),
            "location": Location(id="café rooftop/night", label="Rooftop", prompt="rooftop"),
        }
    )
    name = export.asset_name(brand, "W 17", spec, Aspect.PORTRAIT)
    assert name == "Betterly_W-17_V-07_H-09-us-vs-them_warm-30s_cafe-rooftop-night_4x5"


def test_asset_name_is_capped_and_deterministic(brand: Brand) -> None:
    spec = make_spec("V01", 1, "H02", "warm_30s", "parked_car").model_copy(
        update={"location": Location(id="l" * 200, label="Long", prompt="long"), "voice": Voice(id="v_" * 60, label="Long")}
    )
    first = export.asset_name(brand, "W017", spec, Aspect.SQUARE)
    assert first == export.asset_name(brand, "W017", spec, Aspect.SQUARE)
    assert len(first) <= 120
    assert re.fullmatch(r"[A-Za-z0-9_-]+", first)
    assert first.endswith("_1x1")
    assert len(first.split("_")) == 7
    assert first.startswith("Betterly_W017_V01_H02-confession_")


def test_meta_csv_keeps_passed_and_approved_and_drops_rejected(manifest: RunManifest, brand: Brand, tmp_path: Path) -> None:
    out = export.export_meta_csv(manifest, brand, tmp_path / "exports" / "nested" / "meta.csv")
    rows = read_rows(out)
    assert {row["variant_id"] for row in rows} == {"V01", "V02"}
    assert len(rows) == 6
    required = {"Ad Name", "Campaign Name", "Ad Set Name", "Body", "Title", "Call to Action", "Video File Name"}
    tracking = {
        "variant_id",
        "hook_id",
        "archetype",
        "voice",
        "location",
        "aspect",
        "qa_status",
        "prelaunch_score",
        "utm_content",
        "local_path",
    }
    assert required | tracking | {"placement_hint"} <= set(rows[0])
    hints = {row["aspect"]: row["placement_hint"] for row in rows}
    assert hints == {"9:16": "Stories/Reels", "4:5": "Feed", "1:1": "Feed/Marketplace"}
    first = rows[0]
    assert first["Ad Name"] == "Betterly_W017_V01_H02-confession_warm-30s_parked-car_9x16"
    assert first["utm_content"] == first["Ad Name"]
    assert first["Video File Name"] == f"{first['Ad Name']}.mp4" == Path(first["local_path"]).name
    assert Path(first["local_path"]).is_absolute()
    assert first["Body"] == f"I almost lost her over something stupid & small. {brand.cta}"
    assert first["Title"] == "I almost lost her"
    assert "W017" in first["Campaign Name"] and RUN_ID in first["Ad Set Name"]
    assert first["prelaunch_score"] == "0.710"
    approved = next(row for row in rows if row["variant_id"] == "V02")
    assert (approved["qa_status"], approved["review"]) == ("flagged", "approved")


def test_meta_csv_writes_header_when_nothing_is_launchable(manifest: RunManifest, brand: Brand, tmp_path: Path) -> None:
    rejected = [result.model_copy(update={"review": "rejected"}) for result in manifest.results]
    out = export.export_meta_csv(manifest.model_copy(update={"results": rejected}), brand, tmp_path / "empty.csv")
    assert read_rows(out) == []
    assert out.read_text().startswith("Campaign Name,")


def jsx_payload(script: str) -> dict:
    match = re.search(r"var data = (\{.*?\});\n", script, re.DOTALL)
    assert match
    return json.loads(match.group(1))


def test_ae_jsx_builds_comps_in_plain_extendscript(manifest: RunManifest, brand: Brand, tmp_path: Path) -> None:
    out = export.export_ae_jsx(manifest, brand, tmp_path / "ae" / "import_variants.jsx")
    script = out.read_text()
    assert "app.beginUndoGroup" in script and "app.endUndoGroup" in script
    assert "addComp" in script and "importFile" in script and "addText" in script
    assert '"hook end"' in script
    assert "//" not in script and "/*" not in script and "=>" not in script
    assert not re.search(r"\b(let|const|class)\b", script)
    payload = jsx_payload(script)
    assert payload["projectFolder"] == f"Mitosis {RUN_ID}"
    comps = payload["comps"]
    assert {comp["name"].split("_")[2] for comp in comps} == {"V01", "V02", "V04"}
    assert len(comps) == 9
    for comp in comps:
        assert [comp["width"], comp["height"]] in ([1080, 1920], [1080, 1350], [1080, 1080])
        assert comp["fps"] == 30 and comp["duration"] == 10.4 and comp["hookSeconds"] == 2.1
        assert Path(comp["footage"]).is_absolute() and Path(comp["footage"]).exists()
        assert comp["captions"] and comp["captions"][0]["name"] == f"{comp['name']} caption 01"
        assert all(c["start"] < c["end"] <= comp["duration"] for c in comp["captions"])
        assert all(len(c["text"].split()) <= brand.subtitle.max_words for c in comp["captions"])
        assert comp["headline"]["end"] == 2.1
    ready = {comp["group"] for comp in comps if "_V01_" in comp["name"]}
    review = {comp["group"] for comp in comps if "_V04_" in comp["name"]}
    assert (ready, review) == ({"Ready"}, {"Needs review"})


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_ae_jsx_is_syntactically_valid(manifest: RunManifest, brand: Brand, tmp_path: Path) -> None:
    script = export.export_ae_jsx(manifest, brand, tmp_path / "import_variants.jsx").read_text()
    assert script.startswith("#target aftereffects\n")
    plain = tmp_path / "plain.js"
    plain.write_text(script.split("\n", 1)[1])
    subprocess.run([NODE, "--check", str(plain)], check=True, capture_output=True)


def test_bundle_collects_deliverables_with_relative_paths(manifest: RunManifest, brand: Brand) -> None:
    exports = manifest.out_dir / "exports"
    manifest.exports["meta_csv"] = export.export_meta_csv(manifest, brand, exports / "meta_bulk_upload.csv")
    manifest.exports["after_effects"] = export.export_ae_jsx(manifest, brand, exports / "import_variants.jsx")
    manifest.report = html.write_report(manifest, manifest.out_dir / "report.html")
    missing_thumbnail = manifest.results[3].renders[2].thumbnail
    missing_thumbnail.unlink()
    archive_path = export.bundle(manifest, exports / f"{RUN_ID}.zip")
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        bundled_report = archive.read("report.html").decode()
    v01 = export.asset_name(brand, "W017", manifest.results[0].spec, Aspect.VERTICAL)
    v04 = export.asset_name(brand, "W017", manifest.results[3].spec, Aspect.VERTICAL)
    v03 = export.asset_name(brand, "W017", manifest.results[2].spec, Aspect.VERTICAL)
    assert f"renders/passed/{v01}.mp4" in names
    assert f"renders/flagged/{v04}.mp4" in names
    assert not any(v03 in name for name in names)
    assert f"thumbnails/{v01}.jpg" in names
    assert f"thumbnails/{missing_thumbnail.name}" not in names
    assert {"report.html", "manifest.json", "ledger.jsonl", "exports/meta_bulk_upload.csv", "exports/import_variants.jsx"} <= names
    assert not any(name.startswith("/") or ".." in name or name.endswith(".zip") for name in names)
    assert len([name for name in names if name.startswith("renders/")]) == 9
    assert f'src="renders/passed/{v01}.mp4"' in bundled_report
    assert str(manifest.out_dir) not in bundled_report


def media_references(document: str) -> list[str]:
    return re.findall(r'(?:src|poster|href)="([^"#][^"]*)"', document)


def test_report_has_every_section_and_portable_media(manifest: RunManifest) -> None:
    out = html.write_report(manifest, manifest.out_dir / "report.html")
    document = out.read_text()
    for anchor in ("outcome", "dna", "hooks", "plan", "variants", "spend"):
        assert f'id="{anchor}"' in document
    assert "Blocked before Meta could reject them" in document
    assert "personal_attributes" in document and "Speak from your own experience" in document
    assert "offline_kenburns" in document and "local stand-in" in document
    assert "Production-equivalent" in document and "$7.10" in document
    assert document.count("<video ") == 12
    assert document.count('preload="none"') == 12
    assert 'class="changed"' in document
    assert "Year 7 &lt;b&gt;secret&lt;/b&gt;" in document and "<b>secret</b>" not in document
    assert "something stupid &amp; small" in document
    references = media_references(document)
    assert references
    assert all(not ref.startswith(("/", "file:", "http:", "https:")) for ref in references)
    assert str(manifest.out_dir) not in document
    assert "variants/V01/renders/" in document and "hero/ref_00.jpg" in document
    assert "<link" not in document and "<script src" not in document and "https://" not in document


def test_report_survives_a_sparse_manifest(manifest: RunManifest, tmp_path: Path) -> None:
    sparse = manifest.model_copy(
        update={
            "hooks": [],
            "rejected_hooks": [],
            "results": [],
            "plan": VariantPlan(winner_id="W017", design="", variants=[]),
            "models": {},
            "totals": Totals(),
            "hero": HeroPack(id="hero", description="unknown"),
            "ledger": None,
        }
    )
    document = html.write_report(sparse, tmp_path / "sparse" / "report.html").read_text()
    assert "No variants were produced." in document
    assert "No hook was blocked in this run." in document
    assert "No hero reference frames are available." in document
    assert "No spend recorded." in document


def load_workflow(filename: str) -> dict:
    return json.loads((WORKFLOWS / filename).read_text())


def by_name(workflow: dict) -> dict[str, dict]:
    return {node["name"]: node for node in workflow["nodes"]}


def targets(workflow: dict, source: str, output: int = 0) -> list[str]:
    return [link["node"] for link in workflow["connections"][source]["main"][output]]


@pytest.mark.parametrize("filename", WORKFLOW_FILES)
def test_n8n_workflow_is_importable_and_wired(filename: str) -> None:
    raw = (WORKFLOWS / filename).read_text()
    workflow = json.loads(raw)
    assert {"name", "nodes", "connections", "settings"} <= set(workflow)
    nodes = by_name(workflow)
    assert len(nodes) == len(workflow["nodes"])
    assert len({node["id"] for node in workflow["nodes"]}) == len(nodes)
    for node in workflow["nodes"]:
        assert {"id", "name", "type", "typeVersion", "position", "parameters"} <= set(node)
        assert node["type"].startswith("n8n-nodes-base.")
        assert isinstance(node["typeVersion"], (int, float))
        assert len(node["position"]) == 2
    reached = set()
    for source, outputs in workflow["connections"].items():
        assert source in nodes
        for branch in outputs["main"]:
            for link in branch:
                assert link["node"] in nodes
                assert link["type"] == "main"
                reached.add(link["node"])
    triggers = {name for name, node in nodes.items() if node["type"].endswith("Trigger")}
    assert len(triggers) == 1
    assert set(nodes) - triggers == reached
    for node in workflow["nodes"]:
        code = node["parameters"].get("jsCode", "")
        assert "//" not in code and "/*" not in code
    assert "access_token" not in raw and "Bearer " not in raw and "xoxb-" not in raw


def test_winner_multiplier_flow() -> None:
    workflow = load_workflow("mitosis_winner_multiplier.json")
    nodes = by_name(workflow)
    assert nodes["New Winner in Drive"]["type"] == "n8n-nodes-base.googleDriveTrigger"
    start = nodes["Start Mitosis Run"]["parameters"]
    assert start["contentType"] == "multipart-form-data" and start["url"].endswith("/api/runs")
    fields = {item["name"]: item for item in start["bodyParameters"]["parameters"]}
    assert fields["video"]["parameterType"] == "formBinaryData"
    assert fields["variants"]["value"] == "10" and fields["aspects"]["value"] == "9:16,4:5,1:1"
    assert {"script", "winner_id"} <= set(fields)
    assert targets(workflow, "Start Mitosis Run") == ["Wait 30s"]
    assert nodes["Wait 30s"]["parameters"]["amount"] == 30
    assert targets(workflow, "Wait 30s") == ["Get Run Status"]
    assert targets(workflow, "Run Finished?", 1) == ["Wait 30s"]
    assert targets(workflow, "Run Succeeded?", 1) == ["Alert Failure"]
    assert "Post Run Summary" in targets(workflow, "Summarize Run")
    assert nodes["Post Run Summary"]["parameters"]["channelId"]["value"] == "#creative-pipeline"
    assert "MITOSIS_PUSH_TO_META" in json.dumps(nodes["Push to Meta?"]["parameters"])
    assert targets(workflow, "Push to Meta?") == ["Split Out Ads"]
    assert nodes["Split Out Ads"]["type"] == "n8n-nodes-base.splitOut"
    assert "/advideos" in nodes["Upload to Meta"]["parameters"]["url"]
    assert "META_AD_ACCOUNT_ID" in nodes["Upload to Meta"]["parameters"]["url"]
    assert "report.html" in nodes["Summarize Run"]["parameters"]["jsCode"]


def test_learning_loop_flow() -> None:
    workflow = load_workflow("mitosis_learning_loop.json")
    nodes = by_name(workflow)
    rule = nodes["Every Day 06:00"]["parameters"]["rule"]["interval"][0]
    assert (rule["field"], rule["triggerAtHour"]) == ("days", 6)
    insights = nodes["Fetch Ad Insights"]["parameters"]
    query = {item["name"]: item["value"] for item in insights["queryParameters"]["parameters"]}
    assert query["level"] == "ad" and query["date_preset"] == "last_7d"
    assert query["fields"] == "ad_name,spend,impressions,actions,purchase_roas"
    assert insights["options"]["pagination"]["pagination"]["paginationMode"] == "responseContainsNextURL"
    learn = nodes["Send Results to Mitosis"]["parameters"]
    assert learn["url"].endswith("/api/learn") and learn["contentType"] == "multipart-form-data"
    assert targets(workflow, "Score Ads") == ["Send Results to Mitosis"]
    assert targets(workflow, "Rank Archetypes") == ["Post Learning Summary"]


def run_code_node(code: str, items: list[dict], env: dict[str, str], tmp_path: Path, nodes: dict | None = None) -> list[dict]:
    harness = tmp_path / "harness.js"
    harness.write_text(
        "const fs = require('fs');\n"
        "const payload = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));\n"
        "const wrap = (json) => ({ json });\n"
        "const $input = { all: () => payload.items.map(wrap), first: () => wrap(payload.items[0]) };\n"
        "const $ = (name) => ({ first: () => wrap(payload.nodes[name]) });\n"
        "const run = new Function('$input', '$env', '$', 'Buffer', payload.code);\n"
        "process.stdout.write(JSON.stringify(run($input, payload.env, $, Buffer)));\n"
    )
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps({"code": code, "items": items, "env": env, "nodes": nodes or {}}))
    completed = subprocess.run([NODE, str(harness), str(payload)], check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def insight(name: str, spend: str, purchases: str, roas: str) -> dict:
    return {
        "ad_name": name,
        "spend": spend,
        "impressions": "12000",
        "actions": [{"action_type": "link_click", "value": "90"}, {"action_type": "omni_purchase", "value": purchases}],
        "purchase_roas": [{"action_type": "omni_purchase", "value": roas}],
    }


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_learning_loop_scores_mitosis_ads(brand: Brand, tmp_path: Path) -> None:
    names = [
        export.asset_name(brand, "W017", make_spec(f"V0{index}", index, "H02", "warm_30s", "parked_car"), Aspect.VERTICAL)
        for index in range(1, 5)
    ]
    pages = [
        {
            "data": [
                insight(names[0], "120.50", "6", "2.4"),
                insight(names[1], "80", "1", "0.7"),
                insight("Spring promo static", "300", "9", "3.1"),
            ]
        },
        {"data": [insight(names[2], "5", "0", "0"), insight(names[3], "60", "3", "1.6")]},
    ]
    code = by_name(load_workflow("mitosis_learning_loop.json"))["Score Ads"]["parameters"]["jsCode"]
    output = run_code_node(code, pages, {"MITOSIS_MIN_SPEND": "20", "MITOSIS_TARGET_ROAS": "1"}, tmp_path)
    assert len(output) == 1
    summary = output[0]["json"]
    assert (summary["ads"], summary["eligible"], summary["wins"], summary["roas_bar"]) == (4, 3, 2, 1.6)
    rows = list(csv.DictReader(io.StringIO(base64.b64decode(output[0]["binary"]["results"]["data"]).decode())))
    assert list(rows[0]) == ["ad_name", "spend", "purchases", "roas", "win"]
    wins = {row["ad_name"]: row["win"] for row in rows}
    assert wins == {names[0]: "1", names[1]: "0", names[3]: "1"}

    rank = by_name(load_workflow("mitosis_learning_loop.json"))["Rank Archetypes"]["parameters"]["jsCode"]
    service_rows = [
        {"level": "curiosity", "mean": 0.3333, "trials": 4, "wins": 1},
        {"level": "confession", "mean": 0.7, "trials": 8, "wins": 6},
    ]
    arms = {"confession": {"alpha": 7, "beta": 3, "trials": 8, "wins": 6}, "curiosity": {"alpha": 2, "beta": 4, "trials": 4, "wins": 1}}
    for factors in ({"archetype": service_rows}, {"archetype": arms}):
        ranked = run_code_node(rank, [{"factors": factors, "levels": 2}], {}, tmp_path, {"Score Ads": summary})
        text = ranked[0]["json"]["slack_text"]
        assert "1. confession: 70% expected win rate (6/8 wins)" in text
        assert "2. curiosity: 33%" in text


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_winner_multiplier_summarizes_run(manifest: RunManifest, tmp_path: Path) -> None:
    code = by_name(load_workflow("mitosis_winner_multiplier.json"))["Summarize Run"]["parameters"]["jsCode"]
    output = run_code_node(code, [manifest.model_dump(mode="json")], {"MITOSIS_URL": "http://mitosis.local:8000/"}, tmp_path)
    summary = output[0]["json"]
    assert summary["report_url"] == f"http://mitosis.local:8000/files/{RUN_ID}/report.html"
    assert len(summary["uploads"]) == 6
    first = summary["uploads"][0]
    assert first["file_url"] == f"http://mitosis.local:8000/files/{RUN_ID}/variants/V01/renders/{first['ad_name']}.mp4"
    assert "passed 2" in summary["slack_text"] and "$7.10 production-equivalent" in summary["slack_text"]
