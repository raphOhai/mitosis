from __future__ import annotations

import contextvars
import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mitosis.cache import Cache
from mitosis.config import Settings
from mitosis.events import EventBus
from mitosis.intelligence import deconstruct as dnalib
from mitosis.intelligence import hooks as hooklib
from mitosis.intelligence import planner, predict, structure
from mitosis.intelligence import priors as priorlib
from mitosis.ledger import Ledger
from mitosis.media import align, assemble, audio, frames, probe, scenes, subtitles
from mitosis.models import (
    Aspect,
    AudioPart,
    Brief,
    Check,
    CreativeDNA,
    GateInput,
    Headline,
    HeroPack,
    Hook,
    Priors,
    Render,
    RenderJob,
    RunManifest,
    Segment,
    Shot,
    Totals,
    VariantPlan,
    VariantResult,
    VariantSpec,
    Winner,
    Word,
)
from mitosis.qa import gate, identity
from mitosis.registry import Chain, Registry
from mitosis.report import export, html

log = logging.getLogger(__name__)


@dataclass
class Body:
    path: Path
    words: list[Word]
    seconds: float


@dataclass
class Run:
    run_id: str
    out: Path
    brief: Brief
    registry: Registry
    ledger: Ledger
    priors: Priors
    winner: Winner | None = None
    dna: CreativeDNA | None = None
    hero: HeroPack | None = None
    hooks: list[Hook] = field(default_factory=list)
    rejected: list[Hook] = field(default_factory=list)
    plan: VariantPlan | None = None
    started: float = field(default_factory=time.monotonic)

    @property
    def llm(self) -> Chain | None:
        chain = self.registry.llm
        return chain if chain.available else None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class Pipeline:
    def __init__(self, settings: Settings | None = None, bus: EventBus | None = None) -> None:
        self.settings = settings or Settings.load()
        self.bus = bus or EventBus()
        self.brand = self.settings.brand

    def open(self, brief: Brief, run_id: str | None = None) -> Run:
        run_id = run_id or f"{datetime.now():%Y%m%d-%H%M%S}-{brief.winner_id}"
        out = (brief.out or self.settings.runs_dir / run_id).resolve()
        out.mkdir(parents=True, exist_ok=True)
        ledger = Ledger(out / "ledger.jsonl")
        registry = Registry(self.settings, ledger, Cache(self.settings.cache_dir), offline=brief.offline)
        priors = priorlib.load(self.settings.priors_path)
        run = Run(run_id=run_id, out=out, brief=brief, registry=registry, ledger=ledger, priors=priors)
        self.bus.emit("log", message=f"run {run_id} in {registry.mode} mode", data={"models": registry.describe(), "out": str(out)})
        return run

    def run(self, brief: Brief, run_id: str | None = None) -> RunManifest:
        run = self.open(brief, run_id)
        self.prepare(run)
        with self.bus.stage("produce", f"{len(run.plan.variants)} variants x {len(brief.aspects)} formats"):
            results = self.produce(run)
        with self.bus.stage("report"):
            manifest = self.finish(run, results)
        self.bus.emit("done", message=str(manifest.report), data=manifest.totals.model_dump())
        return manifest

    def prepare(self, run: Run) -> Run:
        with self.bus.stage("ingest", str(run.brief.video.name)):
            run.winner = self.ingest(run)
        with self.bus.stage("understand"):
            run.dna = self.understand(run)
        with self.bus.stage("hero"):
            run.hero = self.lock_hero(run)
        with self.bus.stage("hooks"):
            run.hooks, run.rejected = self.write_hooks(run)
        with self.bus.stage("plan"):
            run.plan = self.design(run)
        return run

    def ingest(self, run: Run) -> Winner:
        brief = run.brief
        folder = run.out / "ingest"
        folder.mkdir(parents=True, exist_ok=True)
        info = probe.probe(brief.video)
        spans = scenes.detect_shots(brief.video, info)
        shots: list[Shot] = []
        for index, (start, end) in enumerate(spans):
            key = frames.extract_frame(brief.video, (start + end) / 2, folder / f"shot_{index:02d}.jpg", width=720)
            shots.append(
                Shot(
                    index=index,
                    start=start,
                    end=end,
                    keyframe=key,
                    face=identity.face_box(key, self.settings),
                    sharpness=frames.sharpness(key),
                )
            )
        wav = audio.extract_audio(brief.video, folder / "audio.wav")
        stems = audio.separate_stems(wav, folder / "stems")
        transcript: list[Word] | None = None
        if run.registry.transcriber.available:
            with run.ledger.scope(stage="transcribe"):
                result = run.registry.transcriber.transcribe(stems.vocals if stems else wav, folder / "transcript.json", hint=brief.script)
            transcript = result.words or None
        words = align.align_script(brief.script, wav, transcript)
        hook_end, cta_start = structure.segment(words, spans, info.duration)
        face_time = sum(s.length for s in shots if s.face)
        winner = Winner(
            id=brief.winner_id,
            video=brief.video,
            script=brief.script,
            info=info,
            audio=wav,
            stems=stems,
            words=words,
            shots=shots,
            hook_end=hook_end,
            cta_start=cta_start,
            talking_head=face_time / max(info.duration, 1e-6) >= 0.5,
        )
        (folder / "winner.json").write_text(winner.model_dump_json(indent=2))
        self.bus.emit(
            "log",
            stage="ingest",
            message=f"{len(shots)} shots, {len(words)} words, hook ends at {hook_end:.2f}s, CTA from {cta_start:.2f}s",
            data={"stems": bool(stems), "talking_head": winner.talking_head},
        )
        return winner

    def understand(self, run: Run) -> CreativeDNA:
        keyframes = [s.keyframe for s in run.winner.shots if s.keyframe][:6]
        with run.ledger.scope(stage="understand"):
            dna = dnalib.deconstruct(run.winner, self.brand, llm=run.llm, frames=keyframes)
        (run.out / "dna.json").write_text(dna.model_dump_json(indent=2))
        self.bus.emit("log", stage="understand", message=f"angle: {dna.angle} | archetype: {dna.hook_archetype}")
        return dna

    def lock_hero(self, run: Run) -> HeroPack:
        samples = frames.sample_frames(run.winner.video, run.out / "hero" / "samples", every=0.5, width=960)
        hero = identity.build_hero(samples, run.out / "hero", run.dna.hero_description, run.brief.seed, self.settings)
        (run.out / "hero" / "hero.json").write_text(hero.model_dump_json(indent=2))
        self.bus.emit(
            "log",
            stage="hero",
            message=f"{len(hero.references)} reference frames, identity embedding {'locked' if hero.embedding else 'unavailable'}",
        )
        return hero

    def write_hooks(self, run: Run) -> tuple[list[Hook], list[Hook]]:
        with run.ledger.scope(stage="hooks"):
            hooks, rejected = hooklib.generate_hooks(
                run.dna,
                run.brief.hook_count,
                self.brand,
                run.priors,
                llm=run.llm,
                archetypes=run.brief.archetypes,
                seed=run.brief.seed,
            )
        payload = {"accepted": [h.model_dump(mode="json") for h in hooks], "rejected": [h.model_dump(mode="json") for h in rejected]}
        (run.out / "hooks.json").write_text(json.dumps(payload, indent=2))
        self.bus.emit("log", stage="hooks", message=f"{len(hooks)} hooks accepted, {len(rejected)} blocked by policy or quality")
        return hooks, rejected

    def body_mode(self, run: Run) -> str:
        mode = run.brief.body_mode
        if mode != "auto":
            return mode
        return "sts" if run.winner.talking_head or run.dna.format == "talking_head" else "tts"

    def estimate_hook_seconds(self, run: Run, text: str) -> float:
        words = max(1, len(text.split()))
        return clamp(words / max(run.dna.pacing_wps, 1.2) + 0.35, self.brand.min_hook_seconds, self.brand.max_hook_seconds)

    def cost(self, run: Run, spec: VariantSpec) -> float:
        registry = run.registry
        hook_seconds = self.estimate_hook_seconds(run, spec.hook.text)
        reference = registry.reference("video")
        clip_seconds = reference.snap(math.ceil(hook_seconds)) if reference else math.ceil(hook_seconds)
        body_seconds = max(0.0, run.winner.info.duration - run.winner.hook_end)
        usd = registry.estimate("speech", {"kchar": len(spec.hook.text) / 1000})
        if spec.new_shot:
            usd += registry.estimate("image", {"image": 1.4})
            usd += registry.estimate("video", {"second": clip_seconds})
        if spec.body_mode == "sts":
            usd += registry.estimate("voice", {"minute": body_seconds / 60})
        elif spec.body_mode == "tts":
            usd += registry.estimate("speech", {"kchar": len(run.dna.body_text + run.dna.cta_text) / 1000})
        if spec.new_shot and run.winner.talking_head:
            usd += registry.estimate("lipsync", {"second": hook_seconds})
        usd += registry.estimate("llm", {"input_mtok": 0.006, "output_mtok": 0.0012})
        return round(usd, 4)

    def design(self, run: Run) -> VariantPlan:
        catalog = self.settings.voices
        voices = [catalog[v] for v in run.brief.voices if v in catalog] or list(catalog.values())
        places = self.settings.locations
        locations = [places[p] for p in run.brief.locations if p in places] or list(places.values())
        result = planner.plan(
            run.winner.id,
            run.dna,
            run.hooks,
            voices,
            locations,
            run.brief.aspects,
            run.brief.variants,
            run.brief.budget_usd,
            run.priors,
            cost=lambda spec: self.cost(run, spec),
            body_mode=self.body_mode(run),
            seed=run.brief.seed,
        )
        self.protect_lipsync(run, result)
        (run.out / "plan.json").write_text(result.model_dump_json(indent=2))
        self.bus.emit(
            "log",
            stage="plan",
            message=f"{len(result.variants)} variants, est. ${result.estimate.usd:.2f} at production prices ({result.design})",
            data=result.coverage,
        )
        return result

    def protect_lipsync(self, run: Run, plan_result: VariantPlan) -> None:
        if not run.winner.talking_head:
            return
        reshot = [spec for spec in plan_result.variants if not spec.new_shot]
        if not reshot:
            return
        for spec in reshot:
            spec.new_shot = True
        plan_result.estimate.usd = round(sum(self.cost(run, spec) for spec in plan_result.variants), 4)
        plan_result.estimate.per_variant_usd = round(plan_result.estimate.usd / max(1, len(plan_result.variants)), 4)
        plan_result.notes.append(
            f"talking-head winner: {len(reshot)} original-location variants get a fresh opening shot in the original set, "
            "because reusing the original opening under a new line would break lip-sync"
        )

    def produce(self, run: Run) -> list[VariantResult]:
        variants = run.plan.variants
        done: dict[str, VariantResult] = {}
        with ThreadPoolExecutor(max_workers=max(1, self.settings.workers)) as pool:
            futures = {pool.submit(contextvars.copy_context().run, self.variant, run, spec): spec for spec in variants}
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    log.exception("variant %s crashed", spec.id)
                    result = VariantResult(spec=spec, status="failed", error=f"{type(exc).__name__}: {exc}")
                done[spec.id] = result
                self.bus.emit(
                    "variant",
                    variant=spec.id,
                    status=result.status,
                    progress=len(done) / len(variants),
                    message=spec.hook.text,
                    data={
                        "usd": result.usd,
                        "shadow_usd": result.shadow_usd,
                        "seconds": result.seconds,
                        "attempts": result.attempts,
                        "score": result.prelaunch.score if result.prelaunch else None,
                        "error": result.error,
                    },
                )
        return [done[s.id] for s in variants]

    def variant(self, run: Run, spec: VariantSpec) -> VariantResult:
        started = time.monotonic()
        folder = run.out / "variants" / spec.id
        result = VariantResult(spec=spec)
        seed = spec.seed
        with run.ledger.scope(variant=spec.id):
            self.bus.emit("variant", variant=spec.id, status="start", message=spec.hook.text)
            speech, hook_words = self.hook_voice(run, spec, folder)
            hook_seconds = self.hook_window(speech.duration or 0.0)
            body = self.body_voice(run, spec, folder)
            words = [w for w in hook_words if w.start < hook_seconds] + [w.shifted(hook_seconds) for w in body.words]
            track = self.mix(run, folder, speech.path, body, hook_seconds)
            renders: list[Render] = []
            for attempt in range(1, run.brief.max_attempts + 1):
                result.attempts = attempt
                with run.ledger.scope(attempt=attempt):
                    hook_segment, shot_checks = self.hook_shot(run, spec, folder, seed, hook_seconds, speech.path)
                    self.bus.emit("variant", variant=spec.id, status="rendering", data={"attempt": attempt})
                    renders = self.render_all(run, spec, folder, hook_segment, shot_checks, body, track, words, hook_seconds)
                decision = gate.decide([r.qa for r in renders if r.qa], attempt, run.brief.max_attempts)
                self.bus.emit("variant", variant=spec.id, status=f"qa:{decision}", data={"attempt": attempt})
                if decision == "accept":
                    result.status = "passed"
                    break
                if decision == "retry_shot" and spec.new_shot:
                    seed += 7919
                    continue
                if decision == "retry_render":
                    continue
                result.status = "flagged"
                break
            else:
                result.status = "flagged"
            result.renders = renders
            result.hook_seconds = round(hook_seconds, 3)
            result.words = words
            primary = next((r for r in renders if r.aspect == Aspect.VERTICAL), renders[0] if renders else None)
            if primary:
                result.prelaunch = predict.prelaunch_score(primary.path, spec, words, hook_seconds, run.priors)
        result.usd, result.shadow_usd, _ = run.ledger.for_variant(spec.id)
        result.seconds = round(time.monotonic() - started, 2)
        (folder / "result.json").write_text(result.model_dump_json(indent=2))
        return result

    def hook_window(self, spoken: float) -> float:
        padded = clamp(spoken + 0.3, self.brand.min_hook_seconds, self.brand.max_hook_seconds)
        return round(max(padded, spoken + 0.12), 3)

    def hook_voice(self, run: Run, spec: VariantSpec, folder: Path):
        with run.ledger.scope(stage="voice"):
            speech = run.registry.speech.speak(spec.hook.text, spec.voice, folder / "audio" / "hook.wav")
        if not speech.duration:
            speech.duration = probe.duration(speech.path)
        words = speech.words or align.proportional_words(spec.hook.text, 0.0, speech.duration)
        return speech, words

    def body_voice(self, run: Run, spec: VariantSpec, folder: Path) -> Body:
        winner = run.winner
        start, end = winner.hook_end, winner.info.duration
        source_words = align.window(winner.words, start, end, offset=-start)
        target = folder / "audio"
        target.mkdir(parents=True, exist_ok=True)
        if spec.body_mode == "tts":
            text = " ".join(w.text for w in winner.words if w.start >= start - 0.02)
            with run.ledger.scope(stage="voice"):
                speech = run.registry.speech.speak(text, spec.voice, target / "body_tts.wav")
            seconds = speech.duration or probe.duration(speech.path)
            words = speech.words or align.proportional_words(text, 0.0, seconds)
            return Body(path=speech.path, words=words, seconds=seconds)
        source = winner.stems.vocals if winner.stems else winner.audio
        clip = audio.extract_audio(source, target / "body_source.wav", start, end)
        if spec.body_mode == "keep":
            return Body(path=clip, words=source_words, seconds=end - start)
        with run.ledger.scope(stage="voice"):
            converted = run.registry.voice.convert(clip, spec.voice, target / "body_voice.wav")
        return Body(path=converted.path, words=source_words, seconds=end - start)

    def mix(self, run: Run, folder: Path, hook_audio: Path, body: Body, hook_seconds: float) -> Path:
        parts = [AudioPart(path=hook_audio, offset=0.0), AudioPart(path=body.path, offset=hook_seconds)]
        bed = run.brief.music or (run.winner.stems.accompaniment if run.winner.stems else None)
        with run.ledger.scope(stage="mix"):
            return audio.build_track(parts, hook_seconds + body.seconds, folder / "audio" / "mix.wav", self.brand.loudness, bed=bed)

    def shot_prompt(self, run: Run, spec: VariantSpec) -> str:
        place = "the same room and lighting as the reference images" if spec.location.original else spec.location.prompt
        return (
            f"{run.hero.description}. Setting: {place}. Action: {spec.hook.visual}. {self.brand.shot_style}. "
            "Keep the exact same person as the reference images: same face, hairline, facial hair, build, skin tone and wardrobe."
        )

    def motion_prompt(self, spec: VariantSpec) -> str:
        return f"{spec.hook.motion}. Subtle natural movement, handheld micro-shake, one continuous shot, the same person throughout, no cuts, no on-screen text."

    def hook_shot(
        self, run: Run, spec: VariantSpec, folder: Path, seed: int, hook_seconds: float, hook_audio: Path
    ) -> tuple[Segment, list[Check]]:
        if not spec.new_shot:
            return self.original_opening(run, hook_seconds), []
        shot_dir = folder / "shot"
        prompt = self.shot_prompt(run, spec)
        checks: list[Check] = []
        best: tuple[float, Path] | None = None
        with run.ledger.scope(stage="keyframe"):
            for tries in range(3):
                key_seed = seed + tries * 101
                image = run.registry.image.generate(
                    prompt, run.hero.references, Aspect.VERTICAL, key_seed, shot_dir / f"key_{key_seed}.png", negative=run.hero.negative
                )
                check = identity.check_image(image.path, run.hero, self.brand.image_identity_threshold, self.settings)
                score = check.score if check.score is not None else 1.0
                if best is None or score > best[0]:
                    best = (score, image.path)
                if check.passed or check.skipped:
                    checks = [check]
                    break
                checks.append(check)
                self.bus.emit("variant", variant=spec.id, status="keyframe_rejected", data={"score": check.score, "seed": key_seed})
        if not any(c.passed or c.skipped for c in checks):
            self.bus.emit("variant", variant=spec.id, status="shot_withheld", data={"best": best[0] if best else None})
            return self.original_opening(run, hook_seconds), checks
        with run.ledger.scope(stage="animate"):
            clip = run.registry.video.animate(
                best[1], self.motion_prompt(spec), hook_seconds, Aspect.VERTICAL, seed, shot_dir / f"clip_{seed}.mp4"
            )
        with run.ledger.scope(stage="identity"):
            checks.append(identity.check_video(clip.path, run.hero, self.brand.identity_threshold, self.settings, 0.0, hook_seconds))
        clip_path, clip_seconds = clip.path, clip.duration or hook_seconds
        if run.winner.talking_head and run.registry.lipsync.available:
            with run.ledger.scope(stage="lipsync"):
                synced = run.registry.lipsync.sync(clip.path, hook_audio, shot_dir / f"lipsync_{seed}.mp4")
            clip_path, clip_seconds = synced.path, synced.duration or clip_seconds
        segment = Segment(path=clip_path, start=0.0, end=min(hook_seconds, clip_seconds), seconds=hook_seconds, reframe="cover")
        return segment, checks

    def original_opening(self, run: Run, hook_seconds: float) -> Segment:
        winner = run.winner
        return Segment(
            path=winner.video,
            start=0.0,
            end=max(winner.hook_end, min(hook_seconds / 1.25, winner.info.duration)),
            seconds=hook_seconds,
            reframe="shots",
            shots=winner.shots,
            info=winner.info,
        )

    def render_all(
        self,
        run: Run,
        spec: VariantSpec,
        folder: Path,
        hook_segment: Segment,
        shot_checks: list[Check],
        body: Body,
        track: Path,
        words: list[Word],
        hook_seconds: float,
    ) -> list[Render]:
        winner = run.winner
        body_segment = Segment(
            path=winner.video,
            start=winner.hook_end,
            end=winner.info.duration,
            seconds=body.seconds,
            reframe="shots",
            shots=winner.shots,
            info=winner.info,
        )
        headline = Headline(text=spec.hook.on_screen_text, start=0.0, end=hook_seconds)
        total = hook_seconds + body.seconds
        renders: list[Render] = []
        out_dir = folder / "renders"
        out_dir.mkdir(parents=True, exist_ok=True)
        for index, aspect in enumerate(spec.aspects):
            name = export.asset_name(self.brand, winner.id, spec, aspect)
            ass = subtitles.write_ass(words, aspect, self.brand, out_dir / f"{name}.ass", headline)
            target = out_dir / f"{name}.mp4"
            with run.ledger.scope(stage="render"):
                started = time.monotonic()
                assemble.render(RenderJob(segments=[hook_segment, body_segment], audio=track, aspect=aspect, out=target, subtitles=ass), self.brand)
                run.ledger.record(kind="render", provider="ffmpeg", model=aspect.value, seconds=round(time.monotonic() - started, 3))
            inp = GateInput(
                variant_id=spec.id,
                aspect=aspect,
                video=target,
                subtitles=ass,
                expected_seconds=total,
                hook_seconds=hook_seconds,
                hook_text=spec.hook.text,
                on_screen_text=spec.hook.on_screen_text,
                spoken_text=" ".join(w.text for w in words),
            )
            with run.ledger.scope(stage="qa"):
                report = gate.evaluate(inp, spec, run.hero, self.brand, self.settings, llm=run.llm if index == 0 else None)
            report.checks = shot_checks + report.checks
            renders.append(
                Render(
                    aspect=aspect,
                    path=target,
                    thumbnail=assemble.thumbnail(target, out_dir / f"{name}.jpg"),
                    subtitles=ass,
                    duration=round(total, 3),
                    qa=report,
                    name=name,
                )
            )
        return renders

    def totals(self, run: Run, results: list[VariantResult]) -> Totals:
        summary = run.ledger.summary()
        count = max(1, len(results))
        wall = time.monotonic() - run.started
        return Totals(
            variants=len(results),
            passed=sum(1 for r in results if r.status == "passed"),
            flagged=sum(1 for r in results if r.status == "flagged"),
            failed=sum(1 for r in results if r.status == "failed"),
            renders=sum(len(r.renders) for r in results),
            usd=summary["usd"],
            shadow_usd=summary["shadow_usd"],
            seconds=round(wall, 2),
            usd_per_variant=round(summary["usd"] / count, 4),
            shadow_per_variant=round(summary["shadow_usd"] / count, 4),
            seconds_per_variant=round(wall / count, 2),
            cache_hits=summary["cache_hits"],
            saved_usd=summary["saved_usd"],
            by_stage=summary["by_stage"],
            by_provider=summary["by_provider"],
        )

    def finish(self, run: Run, results: list[VariantResult]) -> RunManifest:
        manifest = RunManifest(
            run_id=run.run_id,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            out_dir=run.out,
            brief=run.brief,
            winner=run.winner,
            dna=run.dna,
            hero=run.hero,
            hooks=run.hooks,
            rejected_hooks=run.rejected,
            plan=run.plan,
            results=results,
            totals=self.totals(run, results),
            mode=run.registry.mode,
            models=run.registry.describe(),
            ledger=run.ledger.path,
        )
        exports = run.out / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        manifest.exports["meta_csv"] = export.export_meta_csv(manifest, self.brand, exports / "meta_bulk_upload.csv")
        manifest.exports["after_effects"] = export.export_ae_jsx(manifest, self.brand, exports / "import_variants.jsx")
        manifest.report = html.write_report(manifest, run.out / "report.html")
        save_manifest(manifest)
        manifest.exports["bundle"] = export.bundle(manifest, exports / f"{run.run_id}.zip")
        save_manifest(manifest)
        return manifest


def save_manifest(manifest: RunManifest) -> Path:
    path = Path(manifest.out_dir) / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2))
    return path


def load_manifest(path: Path) -> RunManifest:
    return RunManifest.model_validate_json(Path(path).read_text())


def estimate(brief: Brief, settings: Settings | None = None, bus: EventBus | None = None) -> dict[str, Any]:
    pipeline = Pipeline(settings, bus)
    run = pipeline.open(brief)
    pipeline.prepare(run)
    return {
        "run": run.run_id,
        "out": str(run.out),
        "mode": run.registry.mode,
        "plan": run.plan.model_dump(mode="json"),
        "hooks": [h.model_dump(mode="json") for h in run.hooks],
        "rejected": [h.model_dump(mode="json") for h in run.rejected],
    }
