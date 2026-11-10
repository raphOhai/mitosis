from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import re
import shutil
import threading
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeVar
from urllib.parse import quote

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from mitosis.config import Settings
from mitosis.models import (
    Aspect,
    Brief,
    Check,
    CreativeDNA,
    Event,
    HeroPack,
    Hook,
    Priors,
    QAReport,
    Render,
    RunManifest,
    VariantPlan,
    VariantResult,
    Winner,
)
from mitosis.service.jobs import (
    MANIFEST_FILE,
    UPLOADS_DIR,
    Envelope,
    Job,
    JobManager,
    JobRecord,
    JobStatus,
    StreamEnd,
    Subscription,
    utc_now,
    write_atomic,
)

if TYPE_CHECKING:
    from mitosis.registry import Registry

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
REVIEWS_FILE = "reviews.json"
DETAIL_EVENT_TAIL = 120
KEEPALIVE_SECONDS = 15.0
UPLOAD_CHUNK_BYTES = 1 << 20
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".webm", ".mkv"})
AUDIO_SUFFIXES = frozenset({".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"})
CSV_SUFFIXES = frozenset({".csv", ".txt"})
CSV_CONTENT_TYPES = frozenset({"text/csv", "application/csv", "application/vnd.ms-excel", "text/plain"})
VARIANT_LIMITS = (1, 100)
HOOK_LIMITS = (1, 24)
NO_CACHE = {"Cache-Control": "no-cache"}
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
SSE_PREAMBLE = "retry: 3000\n: connected\n\n"
SLUG_PATTERN = re.compile(r"[^a-z0-9_-]+")
MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".csv": "text/csv; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".ass": "text/plain; charset=utf-8",
    ".srt": "text/plain; charset=utf-8",
    ".vtt": "text/vtt; charset=utf-8",
    ".jsx": "text/plain; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".zip": "application/zip",
}
BODY_MODES = (
    {"id": "auto", "label": "Auto", "description": "Voice swap for talking heads, re-voice for voiceovers"},
    {"id": "sts", "label": "Voice swap", "description": "Keep the delivery, change the voice"},
    {"id": "tts", "label": "Re-voice", "description": "Read the body again with the new voice"},
    {"id": "keep", "label": "Keep original", "description": "Only the hook is new"},
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class ReviewRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    note: str = Field(default="", max_length=2000)


class Service:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings
        self._jobs: JobManager | None = None
        self._lock = threading.Lock()
        self._run_locks: dict[str, threading.Lock] = {}
        self.keepalive_seconds = KEEPALIVE_SECONDS

    @property
    def settings(self) -> Settings:
        with self._lock:
            if self._settings is None:
                self._settings = Settings.load()
            return self._settings

    @property
    def jobs(self) -> JobManager:
        settings = self.settings
        with self._lock:
            if self._jobs is None:
                self._jobs = JobManager(settings)
            return self._jobs

    def run_lock(self, run_id: str) -> threading.Lock:
        with self._lock:
            return self._run_locks.setdefault(run_id, threading.Lock())

    def require_record(self, run_id: str) -> JobRecord:
        record = self.jobs.record(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.")
        return record

    def require_run_dir(self, run_id: str) -> Path:
        run_dir = self.jobs.run_dir(run_id)
        if run_dir is None or not run_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.")
        return run_dir

    def shutdown(self) -> None:
        with self._lock:
            jobs = self._jobs
        if jobs is not None:
            jobs.shutdown()


def get_service(request: Request) -> Service:
    return request.app.state.service


ServiceDep = Annotated[Service, Depends(get_service)]


class UrlMapper:
    def __init__(self, run_id: str, run_dir: Path, recorded_dir: Path | None = None) -> None:
        self.run_dir = run_dir
        self.prefix = f"/files/{quote(run_id, safe='')}/"
        candidates = [Path(recorded_dir) if recorded_dir else None, run_dir, run_dir.resolve()]
        self._bases = list(dict.fromkeys(base for base in candidates if base is not None))

    def __call__(self, path: Path | str | None) -> str | None:
        relative = self.relative(path)
        return None if relative is None else self.prefix + quote(relative.as_posix())

    def relative(self, path: Path | str | None) -> PurePosixPath | None:
        if path is None or str(path) == "":
            return None
        candidate = Path(path)
        if ".." in candidate.parts:
            return None
        if not candidate.is_absolute():
            return PurePosixPath(candidate.as_posix())
        for base in self._bases:
            if candidate.is_relative_to(base):
                return PurePosixPath(candidate.relative_to(base).as_posix())
        return None

    def existing(self, path: Path | str | None) -> Path | None:
        relative = self.relative(path)
        if relative is None:
            return None
        target = self.run_dir / relative
        return target if target.is_file() else None

    def link(self, path: Path | str | None) -> str | None:
        return self(path) if self.existing(path) else None


def describe_failure(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def tool_available(name: str) -> bool:
    return shutil.which(name) is not None


def build_registry(settings: Settings) -> Registry:
    from mitosis.cache import Cache
    from mitosis.ledger import Ledger
    from mitosis.registry import Registry

    return Registry(settings, Ledger(), Cache(settings.cache_dir))


def read_manifest(path: Path) -> RunManifest:
    from mitosis.pipeline import load_manifest

    return load_manifest(path)


def write_manifest(manifest: RunManifest, run_dir: Path) -> None:
    from mitosis.pipeline import save_manifest

    if Path(manifest.out_dir).resolve() == run_dir.resolve():
        save_manifest(manifest)
        return
    write_atomic(run_dir / MANIFEST_FILE, manifest.model_dump_json(indent=2))


def learn_priors(settings: Settings, results_csv: Path) -> Priors:
    from mitosis.intelligence import priors as priorlib

    updated = priorlib.learn(priorlib.load(settings.priors_path), results_csv)
    priorlib.save(updated, settings.priors_path)
    return updated


def load_archetypes() -> list[dict[str, str]]:
    try:
        from mitosis.intelligence.hooks import ARCHETYPES
    except ImportError:
        return []
    return normalize_archetypes(ARCHETYPES)


def humanize(identifier: str) -> str:
    return identifier.replace("_", " ").replace("-", " ").strip().capitalize()


def archetype_entry(key: str, value: Any) -> dict[str, str]:
    if isinstance(value, Mapping):
        label = str(value.get("label") or value.get("name") or humanize(key))
        description = str(value.get("description") or value.get("pattern") or value.get("example") or "")
        return {"id": key, "label": label, "description": description}
    description = value if isinstance(value, str) else ""
    return {"id": key, "label": humanize(key), "description": description}


def normalize_archetypes(source: Any) -> list[dict[str, str]]:
    if isinstance(source, Mapping):
        return [archetype_entry(str(key), value) for key, value in source.items()]
    entries: list[dict[str, str]] = []
    for item in source or []:
        if isinstance(item, str):
            entries.append(archetype_entry(item, None))
        elif isinstance(item, Mapping) and (item.get("id") or item.get("name")):
            entries.append(archetype_entry(str(item.get("id") or item.get("name")), item))
    return entries


def read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        log.warning("unreadable json %s: %s", path, exc)
        return None


def read_model(path: Path, model: type[ModelT]) -> ModelT | None:
    if not path.is_file():
        return None
    try:
        return model.model_validate_json(path.read_text())
    except (OSError, ValueError) as exc:
        log.warning("unreadable %s at %s: %s", model.__name__, path, exc)
        return None


def validate_hooks(items: Iterable[Any]) -> list[dict[str, Any]]:
    hooks: list[dict[str, Any]] = []
    for item in items:
        try:
            hooks.append(hook_view(Hook.model_validate(item)))
        except ValueError:
            log.warning("skipping malformed hook entry")
    return hooks


def hook_view(hook: Hook) -> dict[str, Any]:
    return hook.model_dump(mode="json")


def hero_view(hero: HeroPack, url: UrlMapper) -> dict[str, Any]:
    references = [link for link in (url(path) for path in hero.references) if link]
    return {
        "id": hero.id,
        "description": hero.description,
        "references": references,
        "identity_locked": hero.embedding is not None,
        "embedding_model": hero.embedding_model,
    }


def winner_view(winner: Winner) -> dict[str, Any]:
    return {
        "id": winner.id,
        "duration": winner.info.duration,
        "width": winner.info.width,
        "height": winner.info.height,
        "fps": winner.info.fps,
        "hook_end": winner.hook_end,
        "cta_start": winner.cta_start,
        "talking_head": winner.talking_head,
        "shots": len(winner.shots),
        "words": len(winner.words),
    }


def plan_view(plan: VariantPlan) -> dict[str, Any]:
    view = plan.model_dump(mode="json", exclude={"variants"})
    view["variant_count"] = len(plan.variants)
    return view


def check_view(check: Check) -> dict[str, Any]:
    return check.model_dump(mode="json")


def qa_view(report: QAReport) -> dict[str, Any]:
    return {"passed": report.passed, "checks": [check_view(check) for check in report.checks]}


def render_view(render: Render, url: UrlMapper) -> dict[str, Any]:
    return {
        "aspect": render.aspect.value,
        "name": render.name,
        "url": url(render.path),
        "thumbnail": url(render.thumbnail),
        "subtitles": url(render.subtitles),
        "duration": render.duration,
        "qa": qa_view(render.qa) if render.qa else None,
    }


def qa_issue(report: QAReport, check: Check) -> dict[str, Any]:
    return {
        "aspect": report.aspect.value,
        "name": check.name,
        "severity": check.severity,
        "detail": check.detail,
        "score": check.score,
        "threshold": check.threshold,
    }


def qa_summary(renders: list[Render]) -> dict[str, Any]:
    reports = [render.qa for render in renders if render.qa]
    issues = [qa_issue(report, check) for report in reports for check in report.checks if not check.passed and not check.skipped]
    return {
        "passed": bool(reports) and all(report.passed for report in reports),
        "reports": len(reports),
        "checks": sum(len(report.checks) for report in reports),
        "issues": issues,
    }


def variant_view(result: VariantResult, url: UrlMapper, note: str) -> dict[str, Any]:
    spec = result.spec
    return {
        "id": spec.id,
        "index": spec.index,
        "status": result.status,
        "review": result.review,
        "note": note,
        "attempts": result.attempts,
        "usd": result.usd,
        "shadow_usd": result.shadow_usd,
        "seconds": result.seconds,
        "hook_seconds": result.hook_seconds,
        "error": result.error,
        "hook": hook_view(spec.hook),
        "voice": {"id": spec.voice.id, "label": spec.voice.label, "tone": spec.voice.tone},
        "location": {"id": spec.location.id, "label": spec.location.label, "original": spec.location.original},
        "changed": list(spec.changed),
        "body_mode": spec.body_mode,
        "new_shot": spec.new_shot,
        "aspects": [aspect.value for aspect in spec.aspects],
        "prelaunch": result.prelaunch.model_dump(mode="json") if result.prelaunch else None,
        "renders": [render_view(render, url) for render in result.renders],
        "qa": qa_summary(result.renders),
    }


def review_counts(results: list[VariantResult]) -> dict[str, int]:
    decisions = [result.review for result in results]
    return {decision: decisions.count(decision) for decision in ("approved", "rejected", "pending")}


def links_view(run_id: str, manifest: RunManifest, url: UrlMapper) -> dict[str, str | None]:
    exports = manifest.exports
    bundle = url.existing(exports.get("bundle"))
    return {
        "report": url.link(manifest.report),
        "bundle": f"/api/runs/{quote(run_id, safe='')}/bundle" if bundle else None,
        "meta_csv": url.link(exports.get("meta_csv")),
        "after_effects": url.link(exports.get("after_effects")),
        "ledger": url.link(manifest.ledger),
        "manifest": url.link(Path(MANIFEST_FILE)),
    }


def manifest_view(run_id: str, manifest: RunManifest, run_dir: Path, notes: dict[str, str]) -> dict[str, Any]:
    url = UrlMapper(run_id, run_dir, manifest.out_dir)
    return {
        "run_id": run_id,
        "created_at": manifest.created_at,
        "mode": manifest.mode,
        "models": manifest.models,
        "totals": manifest.totals.model_dump(mode="json"),
        "winner": winner_view(manifest.winner),
        "dna": manifest.dna.model_dump(mode="json"),
        "hero": hero_view(manifest.hero, url),
        "hooks": [hook_view(hook) for hook in manifest.hooks],
        "rejected_hooks": [hook_view(hook) for hook in manifest.rejected_hooks],
        "plan": plan_view(manifest.plan),
        "results": [variant_view(result, url, notes.get(result.spec.id, "")) for result in manifest.results],
        "review": review_counts(manifest.results),
        "links": links_view(run_id, manifest, url),
    }


def estimate_view(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": result.get("mode"),
        "plan": result.get("plan"),
        "hooks": validate_hooks(result.get("hooks") or []),
        "rejected_hooks": validate_hooks(result.get("rejected") or []),
    }


def partial_view(run_id: str, run_dir: Path) -> dict[str, Any]:
    url = UrlMapper(run_id, run_dir)
    view: dict[str, Any] = {}
    hooks = read_json(run_dir / "hooks.json")
    if isinstance(hooks, dict):
        view["hooks"] = validate_hooks(hooks.get("accepted") or [])
        view["rejected_hooks"] = validate_hooks(hooks.get("rejected") or [])
    hero = read_model(run_dir / "hero" / "hero.json", HeroPack)
    if hero is not None:
        view["hero"] = hero_view(hero, url)
    dna = read_model(run_dir / "dna.json", CreativeDNA)
    if dna is not None:
        view["dna"] = dna.model_dump(mode="json")
    return view


def event_view(sequence: int, event: Event) -> dict[str, Any]:
    return {"seq": sequence, **event.model_dump(mode="json")}


def record_events(record: JobRecord) -> list[dict[str, Any]]:
    first = record.sequence - len(record.events) + 1
    views = [event_view(first + offset, event) for offset, event in enumerate(record.events)]
    return views[-DETAIL_EVENT_TAIL:]


def record_view(record: JobRecord) -> dict[str, Any]:
    return {
        "run_id": record.run_id,
        "kind": record.kind,
        "status": record.status,
        "error": record.error,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "brief": record.brief,
        "progress": record.progress,
        "variants_total": record.variants_total,
        "variants_done": record.variants_done,
        "variant_progress": record.variant_progress,
        "stages": [stage.model_dump(mode="json") for stage in record.stages],
        "variants": [variant.model_dump(mode="json") for variant in sorted(record.variants.values(), key=lambda v: v.id)],
        "events": record_events(record),
        "mode": record.mode,
        "totals": record.totals,
    }


def summary_view(record: JobRecord) -> dict[str, Any]:
    totals = record.totals or {}
    return {
        "run_id": record.run_id,
        "kind": record.kind,
        "status": record.status,
        "created_at": record.created_at,
        "winner_id": record.brief.get("winner_id", ""),
        "variants": totals.get("variants", record.brief.get("variants", 0)),
        "passed": totals.get("passed"),
        "usd": totals.get("usd"),
        "shadow_usd": totals.get("shadow_usd"),
        "mode": record.mode,
        "progress": record.progress,
        "error": record.error,
    }


def read_review_notes(run_dir: Path) -> dict[str, str]:
    data = read_json(run_dir / REVIEWS_FILE)
    if not isinstance(data, dict):
        return {}
    return {key: str(value.get("note", "")) for key, value in data.items() if isinstance(value, dict)}


def store_review_note(run_dir: Path, variant_id: str, review: ReviewRequest) -> dict[str, str]:
    data = read_json(run_dir / REVIEWS_FILE)
    reviews = data if isinstance(data, dict) else {}
    reviews[variant_id] = {"decision": review.decision, "note": review.note.strip(), "at": utc_now()}
    write_atomic(run_dir / REVIEWS_FILE, json.dumps(reviews, indent=2))
    return {key: str(value.get("note", "")) for key, value in reviews.items() if isinstance(value, dict)}


def results_view(record: JobRecord) -> dict[str, Any]:
    manifest_path = record.run_dir / MANIFEST_FILE
    if manifest_path.is_file():
        try:
            manifest = read_manifest(manifest_path)
        except (OSError, ValueError) as exc:
            log.warning("could not read manifest for %s: %s", record.run_id, exc)
            return {"manifest_error": "The results file for this run could not be read."}
        notes = read_review_notes(record.run_dir)
        return {"manifest": manifest_view(record.run_id, manifest, record.run_dir, notes)}
    partial = partial_view(record.run_id, record.run_dir)
    if record.estimate is not None:
        return {"estimate": estimate_view(record.estimate), "partial": partial}
    return {"partial": partial}


def priors_summary(priors: Priors) -> dict[str, Any]:
    factors = {
        factor: sorted(
            (
                {"level": level, "mean": round(arm.mean, 4), "trials": arm.trials, "wins": arm.wins}
                for level, arm in arms.items()
            ),
            key=lambda row: row["mean"],
            reverse=True,
        )
        for factor, arms in priors.factors.items()
    }
    return {"updated_at": priors.updated_at, "factors": factors, "levels": sum(len(rows) for rows in factors.values())}


def parse_csv(value: str) -> list[str]:
    return list(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))


def parse_aspect(token: str) -> Aspect:
    try:
        return Aspect(token.lower().replace("x", ":"))
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Unknown format '{token}'. Use 9:16, 4:5 or 1:1.") from None


def parse_aspects(value: str) -> list[Aspect]:
    aspects = list(dict.fromkeys(parse_aspect(token) for token in parse_csv(value)))
    if not aspects:
        raise HTTPException(status_code=422, detail="Pick at least one format.")
    return aspects


def pick_known(values: list[str], known: Iterable[str], label: str) -> list[str]:
    known_ids = set(known)
    unknown = [value for value in values if value not in known_ids]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown {label}: {', '.join(unknown)}.")
    return values


def slugify(value: str) -> str:
    return SLUG_PATTERN.sub("-", value.lower()).strip("-_")[:48].strip("-_")


def upload_suffix(upload: UploadFile, allowed: frozenset[str], fallback: str) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    return suffix if suffix in allowed else fallback


def is_video_upload(upload: UploadFile) -> bool:
    content_type = (upload.content_type or "").lower()
    return content_type.startswith("video/") or Path(upload.filename or "").suffix.lower() in VIDEO_SUFFIXES


def is_csv_upload(upload: UploadFile) -> bool:
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    return content_type in CSV_CONTENT_TYPES or Path(upload.filename or "").suffix.lower() in CSV_SUFFIXES


def has_file(upload: UploadFile | None) -> bool:
    return upload is not None and bool(upload.filename)


def save_upload(upload: UploadFile, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    upload.file.seek(0)
    with target.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle, UPLOAD_CHUNK_BYTES)
    return target


def learn_upload_path(settings: Settings, filename: str) -> Path:
    stem = slugify(Path(filename).stem) or "results"
    return Path(settings.runs_dir) / UPLOADS_DIR / "learn" / f"{datetime.now():%Y%m%d-%H%M%S}-{stem}.csv"


def media_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    return MEDIA_TYPES.get(suffix) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def resolve_inside(root: Path, relative: str) -> Path:
    candidate = PurePosixPath(relative)
    if not relative or "\x00" in relative or candidate.is_absolute() or ".." in candidate.parts:
        raise HTTPException(status_code=403, detail="That path is outside this run.")
    base = root.resolve()
    try:
        target = (base / candidate).resolve(strict=True)
    except (OSError, RuntimeError):
        raise HTTPException(status_code=404, detail="File not found.") from None
    if not target.is_relative_to(base):
        raise HTTPException(status_code=403, detail="That path is outside this run.")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return target


def parse_last_event_id(request: Request) -> int:
    value = request.headers.get("last-event-id", "")
    return int(value) if value.isdigit() else 0


def sse_message(envelope: Envelope) -> str:
    return f"id: {envelope.sequence}\ndata: {json.dumps(event_view(envelope.sequence, envelope.event))}\n\n"


def sse_end(status: JobStatus, error: str | None) -> str:
    return f"event: end\ndata: {json.dumps({'status': status, 'error': error})}\n\n"


async def follow(subscription: Subscription, keepalive: float) -> AsyncIterator[str]:
    while True:
        item = await subscription.next(keepalive)
        if item is None:
            yield ": keepalive\n\n"
        elif isinstance(item, StreamEnd):
            yield sse_end(item.status, item.error)
            return
        else:
            yield sse_message(item)


async def live_stream(job: Job, last_seen: int, keepalive: float) -> AsyncIterator[str]:
    history, subscription = job.subscribe(asyncio.get_running_loop())
    try:
        yield SSE_PREAMBLE
        for envelope in history:
            if envelope.sequence > last_seen:
                yield sse_message(envelope)
        if subscription is None:
            record = job.snapshot(with_events=False)
            yield sse_end(record.status, record.error)
            return
        async for chunk in follow(subscription, keepalive):
            yield chunk
    finally:
        if subscription is not None:
            job.unsubscribe(subscription)


async def replay_stream(record: JobRecord, last_seen: int) -> AsyncIterator[str]:
    yield SSE_PREAMBLE
    first = record.sequence - len(record.events) + 1
    for offset, event in enumerate(record.events):
        sequence = first + offset
        if sequence > last_seen:
            yield sse_message(Envelope(sequence, event))
    yield sse_end(record.status, record.error)


router = APIRouter()


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html; charset=utf-8", headers=NO_CACHE)


@router.get("/api/health")
def health(service: ServiceDep) -> dict[str, Any]:
    settings = service.settings
    payload: dict[str, Any] = {
        "status": "ok",
        "ffmpeg": tool_available("ffmpeg"),
        "ffprobe": tool_available("ffprobe"),
        "offline_default": settings.offline,
        "workers": settings.workers,
        "jobs": service.jobs.counts(),
    }
    try:
        registry = build_registry(settings)
    except Exception as exc:
        log.exception("model registry unavailable")
        payload.update(status="degraded", mode="unknown", chains={}, error=describe_failure(exc))
        return payload
    payload.update(mode=registry.mode, chains=registry.describe())
    if not payload["ffmpeg"]:
        payload["status"] = "degraded"
    return payload


@router.get("/api/config")
def config(service: ServiceDep) -> dict[str, Any]:
    settings = service.settings
    brand = settings.brand
    defaults = Brief.model_fields
    return {
        "brand": {"name": brand.name, "product": brand.product, "cta": brand.cta, "audience": brand.audience, "tone": brand.tone},
        "voices": [
            {"id": voice.id, "label": voice.label, "tone": voice.tone, "gender": voice.gender, "age": voice.age}
            for voice in settings.voices.values()
        ],
        "locations": [
            {"id": place.id, "label": place.label, "grade": place.grade, "original": place.original}
            for place in settings.locations.values()
        ],
        "aspects": [{"value": aspect.value, "slug": aspect.slug, "width": aspect.size[0], "height": aspect.size[1]} for aspect in Aspect],
        "archetypes": load_archetypes(),
        "body_modes": list(BODY_MODES),
        "defaults": {
            "variants": defaults["variants"].default,
            "hook_count": defaults["hook_count"].default,
            "budget_usd": defaults["budget_usd"].default,
            "body_mode": defaults["body_mode"].default,
            "offline": settings.offline,
        },
        "limits": {"variants": [3, 30], "hook_count": [2, 12]},
    }


@router.post("/api/runs", status_code=202)
def create_run(
    service: ServiceDep,
    video: Annotated[UploadFile, File()],
    script: Annotated[str, Form()],
    music: Annotated[UploadFile | None, File()] = None,
    winner_id: Annotated[str, Form()] = "",
    variants: Annotated[int, Form(ge=VARIANT_LIMITS[0], le=VARIANT_LIMITS[1])] = 10,
    aspects: Annotated[str, Form()] = "9:16,4:5,1:1",
    voices: Annotated[str, Form()] = "",
    locations: Annotated[str, Form()] = "",
    archetypes: Annotated[str, Form()] = "",
    hook_count: Annotated[int, Form(ge=HOOK_LIMITS[0], le=HOOK_LIMITS[1])] = 6,
    budget_usd: Annotated[float, Form(ge=0, le=100_000)] = 40.0,
    offline: Annotated[bool, Form()] = False,
    body_mode: Annotated[Literal["auto", "keep", "sts", "tts"], Form()] = "auto",
    dry_run: Annotated[bool, Form()] = False,
    notes: Annotated[str, Form(max_length=4000)] = "",
) -> dict[str, str]:
    settings = service.settings
    if not is_video_upload(video):
        raise HTTPException(status_code=422, detail="The winner must be a video file (MP4, MOV or WebM).")
    if video.size == 0:
        raise HTTPException(status_code=422, detail="The video file is empty.")
    script_text = script.strip()
    if not script_text:
        raise HTTPException(status_code=422, detail="Paste the winner's script so new hooks can match its voice.")
    aspect_list = parse_aspects(aspects)
    voice_ids = pick_known(parse_csv(voices), settings.voices, "voice")
    location_ids = pick_known(parse_csv(locations), settings.locations, "location")
    winner = slugify(winner_id) or slugify(Path(video.filename or "").stem) or "winner"
    run_id = service.jobs.reserve_run_id(winner)
    upload_dir = Path(settings.runs_dir) / UPLOADS_DIR / run_id
    video_path = save_upload(video, upload_dir / f"winner{upload_suffix(video, VIDEO_SUFFIXES, '.mp4')}")
    music_path = save_upload(music, upload_dir / f"music{upload_suffix(music, AUDIO_SUFFIXES, '.wav')}") if has_file(music) else None
    brief = Brief(
        winner_id=winner,
        video=video_path,
        script=script_text,
        variants=variants,
        aspects=aspect_list,
        voices=voice_ids,
        locations=location_ids,
        hook_count=hook_count,
        archetypes=parse_csv(archetypes) or None,
        budget_usd=budget_usd,
        offline=True if offline else None,
        music=music_path,
        body_mode=body_mode,
        notes=notes.strip(),
    )
    job = service.jobs.submit(brief, "estimate" if dry_run else "run", run_id)
    return {"run_id": job.run_id, "kind": job.kind, "status": job.status}


@router.get("/api/runs")
def list_runs(service: ServiceDep) -> dict[str, list[dict[str, Any]]]:
    return {"runs": [summary_view(record) for record in service.jobs.records()]}


@router.get("/api/runs/{run_id}")
def get_run(run_id: str, service: ServiceDep) -> dict[str, Any]:
    record = service.require_record(run_id)
    return {**record_view(record), **results_view(record)}


@router.get("/api/runs/{run_id}/events")
async def stream_run_events(run_id: str, request: Request, service: ServiceDep) -> StreamingResponse:
    last_seen = parse_last_event_id(request)
    job = service.jobs.get(run_id)
    if job is not None:
        stream = live_stream(job, last_seen, service.keepalive_seconds)
    else:
        record = await run_in_threadpool(service.require_record, run_id)
        stream = replay_stream(record, last_seen)
    return StreamingResponse(stream, media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/api/runs/{run_id}/variants/{variant_id}/review")
def review_variant(run_id: str, variant_id: str, review: ReviewRequest, service: ServiceDep) -> dict[str, Any]:
    run_dir = service.require_run_dir(run_id)
    manifest_path = run_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise HTTPException(status_code=409, detail="This run has no finished variants to review yet.")
    with service.run_lock(run_id):
        manifest = read_manifest(manifest_path)
        result = next((item for item in manifest.results if item.spec.id == variant_id), None)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Variant '{variant_id}' is not part of this run.")
        result.review = review.decision
        write_manifest(manifest, run_dir)
        notes = store_review_note(run_dir, variant_id, review)
    log.info("variant %s/%s %s", run_id, variant_id, review.decision)
    return variant_view(result, UrlMapper(run_id, run_dir, manifest.out_dir), notes.get(variant_id, ""))


@router.get("/api/runs/{run_id}/bundle")
def download_bundle(run_id: str, service: ServiceDep) -> FileResponse:
    run_dir = service.require_run_dir(run_id)
    manifest_path = run_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail="This run has not produced a bundle yet.")
    manifest = read_manifest(manifest_path)
    bundle = UrlMapper(run_id, run_dir, manifest.out_dir).existing(manifest.exports.get("bundle"))
    if bundle is None:
        raise HTTPException(status_code=404, detail="This run has not produced a bundle yet.")
    target = resolve_inside(run_dir, bundle.relative_to(run_dir).as_posix())
    return FileResponse(target, media_type="application/zip", filename=target.name)


@router.api_route("/files/{run_id}/{relative:path}", methods=["GET", "HEAD"])
def serve_run_file(run_id: str, relative: str, service: ServiceDep) -> FileResponse:
    run_dir = service.require_run_dir(run_id)
    target = resolve_inside(run_dir, relative)
    return FileResponse(target, media_type=media_type_for(target), headers={"X-Content-Type-Options": "nosniff"})


@router.post("/api/learn")
def learn_from_results(service: ServiceDep, file: Annotated[UploadFile, File()]) -> dict[str, Any]:
    if not is_csv_upload(file):
        raise HTTPException(status_code=422, detail="Upload the results as a CSV export.")
    settings = service.settings
    target = save_upload(file, learn_upload_path(settings, file.filename or ""))
    try:
        updated = learn_priors(settings, target)
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="Learning from results is not available in this build.") from None
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not learn from this file: {exc}") from exc
    summary = priors_summary(updated)
    log.info("priors updated from %s (%d levels)", target.name, summary["levels"])
    return summary


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    yield
    application.state.service.shutdown()


def create_app(settings: Settings | None = None) -> FastAPI:
    application = FastAPI(
        title="Mitosis",
        summary="Winner multiplier",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
    application.state.service = Service(settings)
    application.include_router(router)
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return application


app = create_app()
