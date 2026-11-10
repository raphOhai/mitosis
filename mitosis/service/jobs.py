from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import re
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from mitosis.config import Settings
from mitosis.events import EventBus
from mitosis.models import Brief, Event, RunManifest

log = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "done", "error"]
JobKind = Literal["run", "estimate"]
StageStatus = Literal["pending", "running", "done", "error"]

RUN_STAGES: tuple[str, ...] = ("ingest", "understand", "hero", "hooks", "plan", "produce", "report")
ESTIMATE_STAGES: tuple[str, ...] = RUN_STAGES[:5]
STAGE_WEIGHTS: dict[str, float] = {
    "ingest": 0.06,
    "understand": 0.06,
    "hero": 0.06,
    "hooks": 0.05,
    "plan": 0.02,
    "produce": 0.70,
    "report": 0.05,
}
DEFAULT_STAGE_WEIGHT = 0.05
FINISHED_VARIANT_STATES = frozenset({"passed", "flagged", "failed"})
TERMINAL_STATES = frozenset({"done", "error"})
PERSISTED_EVENT_TYPES = frozenset({"stage", "done", "error"})
EVENT_TAIL = 300
JOB_FILE = "job.json"
MANIFEST_FILE = "manifest.json"
UPLOADS_DIR = "uploads"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
VARIANT_COUNT_PATTERN = re.compile(r"^\s*(\d+)\s+variants?\b")
INTERRUPTED_MESSAGE = "The server stopped before this job finished. Start it again from the panel."


class StageState(BaseModel):
    name: str
    status: StageStatus = "pending"
    started_at: float | None = None
    seconds: float | None = None
    message: str = ""


class VariantState(BaseModel):
    id: str
    status: str = "queued"
    hook: str = ""
    attempt: int = 0
    usd: float | None = None
    shadow_usd: float | None = None
    seconds: float | None = None
    score: float | None = None
    error: str | None = None


class JobRecord(BaseModel):
    run_id: str
    kind: JobKind
    status: JobStatus = "queued"
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    run_dir: Path
    brief: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    progress: float = 0.0
    variants_total: int = 0
    variants_done: int = 0
    stages: list[StageState] = Field(default_factory=list)
    variants: dict[str, VariantState] = Field(default_factory=dict)
    events: list[Event] = Field(default_factory=list)
    sequence: int = 0
    estimate: dict[str, Any] | None = None
    totals: dict[str, Any] | None = None
    mode: str | None = None

    @property
    def variant_progress(self) -> float:
        if self.variants_total <= 0:
            return 0.0
        return round(min(1.0, self.variants_done / self.variants_total), 4)


@dataclass(frozen=True)
class Envelope:
    sequence: int
    event: Event


@dataclass(frozen=True)
class StreamEnd:
    status: JobStatus
    error: str | None


StreamItem = Envelope | StreamEnd


class Subscription:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue: asyncio.Queue[StreamItem] = asyncio.Queue()

    def push(self, item: StreamItem) -> bool:
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)
        except RuntimeError:
            return False
        return True

    async def next(self, timeout: float) -> StreamItem | None:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout)
        except TimeoutError:
            return None


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def is_valid_run_id(run_id: str) -> bool:
    return bool(RUN_ID_PATTERN.match(run_id)) and run_id != UPLOADS_DIR


def write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def describe_error(exc: BaseException) -> str:
    message = str(exc).strip()
    return message or type(exc).__name__


def summarize_brief(brief: Brief) -> dict[str, Any]:
    words = brief.script.split()
    return {
        "winner_id": brief.winner_id,
        "video": Path(brief.video).name,
        "script_preview": " ".join(words[:40]),
        "script_words": len(words),
        "variants": brief.variants,
        "aspects": [aspect.value for aspect in brief.aspects],
        "voices": list(brief.voices),
        "locations": list(brief.locations),
        "archetypes": list(brief.archetypes or []),
        "hook_count": brief.hook_count,
        "budget_usd": brief.budget_usd,
        "offline": brief.offline,
        "body_mode": brief.body_mode,
        "music": Path(brief.music).name if brief.music else None,
        "notes": brief.notes,
    }


def initial_stages(kind: JobKind) -> list[StageState]:
    names = ESTIMATE_STAGES if kind == "estimate" else RUN_STAGES
    return [StageState(name=name) for name in names]


def find_stage(record: JobRecord, name: str) -> StageState:
    for stage in record.stages:
        if stage.name == name:
            return stage
    stage = StageState(name=name)
    record.stages.append(stage)
    return stage


def as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def parse_variant_count(message: str) -> int | None:
    match = VARIANT_COUNT_PATTERN.match(message or "")
    return int(match.group(1)) if match else None


def apply_stage(record: JobRecord, event: Event) -> None:
    stage = find_stage(record, event.stage or "")
    if event.status == "start":
        stage.status = "running"
        stage.started_at = event.ts
        stage.message = event.message
        if stage.name == "produce":
            record.variants_total = parse_variant_count(event.message) or record.variants_total
    elif event.status == "done":
        stage.status = "done"
        seconds = as_float(event.data.get("seconds"))
        if seconds is None and stage.started_at is not None:
            seconds = round(event.ts - stage.started_at, 2)
        stage.seconds = seconds


def apply_variant(record: JobRecord, event: Event) -> None:
    variant_id = event.variant or ""
    state = record.variants.setdefault(variant_id, VariantState(id=variant_id))
    if event.status:
        state.status = event.status
    if event.message:
        state.hook = event.message
    attempt = event.data.get("attempt", event.data.get("attempts"))
    if isinstance(attempt, int):
        state.attempt = attempt
    if event.status not in FINISHED_VARIANT_STATES:
        return
    state.usd = as_float(event.data.get("usd"))
    state.shadow_usd = as_float(event.data.get("shadow_usd"))
    state.seconds = as_float(event.data.get("seconds"))
    state.score = as_float(event.data.get("score"))
    state.error = event.data.get("error") or None
    record.variants_done = sum(1 for variant in record.variants.values() if variant.status in FINISHED_VARIANT_STATES)
    if event.progress:
        record.variants_total = max(record.variants_done, round(record.variants_done / event.progress))


def apply_error(record: JobRecord, event: Event) -> None:
    if event.stage:
        find_stage(record, event.stage).status = "error"
    record.error = event.message or record.error


def overall_progress(record: JobRecord) -> float:
    total = sum(STAGE_WEIGHTS.get(stage.name, DEFAULT_STAGE_WEIGHT) for stage in record.stages)
    if total <= 0:
        return 0.0
    earned = 0.0
    for stage in record.stages:
        weight = STAGE_WEIGHTS.get(stage.name, DEFAULT_STAGE_WEIGHT)
        if stage.status == "done":
            earned += weight
        elif stage.status == "running" and stage.name == "produce":
            earned += weight * record.variant_progress
    return round(min(1.0, earned / total), 4)


def apply_event(record: JobRecord, event: Event) -> None:
    if event.type == "stage" and event.stage:
        apply_stage(record, event)
    elif event.type == "variant" and event.variant:
        apply_variant(record, event)
    elif event.type == "error":
        apply_error(record, event)
    record.progress = overall_progress(record)


def should_persist(event: Event) -> bool:
    return event.type in PERSISTED_EVENT_TYPES or (event.type == "variant" and event.status in FINISHED_VARIANT_STATES)


def estimate_digest(result: dict[str, Any]) -> dict[str, Any]:
    plan = result.get("plan") or {}
    return {
        "variants": len(plan.get("variants") or []),
        "usd": (plan.get("estimate") or {}).get("usd"),
        "hooks": len(result.get("hooks") or []),
        "rejected": len(result.get("rejected") or []),
    }


class Job:
    def __init__(self, record: JobRecord) -> None:
        self._events: deque[Event] = deque(record.events, maxlen=EVENT_TAIL)
        self._record = record.model_copy(update={"events": []})
        self._subscribers: list[Subscription] = []
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._revision = 0
        self._written_revision = -1

    @property
    def run_id(self) -> str:
        return self._record.run_id

    @property
    def kind(self) -> JobKind:
        return self._record.kind

    @property
    def run_dir(self) -> Path:
        return self._record.run_dir

    @property
    def status(self) -> JobStatus:
        with self._lock:
            return self._record.status

    def handle(self, event: Event) -> None:
        with self._lock:
            self._append(event)
            apply_event(self._record, event)
        if should_persist(event):
            self.persist()

    def start(self) -> None:
        with self._lock:
            self._record.status = "running"
            self._record.started_at = utc_now()
            self._revision += 1
        self.persist()

    def complete_run(self, manifest: RunManifest) -> None:
        with self._lock:
            self._record.totals = manifest.totals.model_dump(mode="json")
            self._record.mode = manifest.mode
            if not self._has_event("done"):
                self._append(Event(type="done", message="Run complete", data=self._record.totals))
            self._finish("done", None)
        self.persist()

    def complete_estimate(self, result: dict[str, Any]) -> None:
        with self._lock:
            self._record.estimate = result
            self._record.mode = result.get("mode")
            if not self._has_event("done"):
                self._append(Event(type="done", message="Estimate ready", data=estimate_digest(result)))
            self._finish("done", None)
        self.persist()

    def fail(self, error: str) -> None:
        with self._lock:
            if not self._has_event("error"):
                self._append(Event(type="error", status="error", message=error))
            for stage in self._record.stages:
                if stage.status == "running":
                    stage.status = "error"
            self._finish("error", error)
        self.persist()

    def subscribe(self, loop: asyncio.AbstractEventLoop) -> tuple[list[Envelope], Subscription | None]:
        with self._lock:
            history = self._history()
            if self._record.status in TERMINAL_STATES:
                return history, None
            subscription = Subscription(loop)
            self._subscribers.append(subscription)
            return history, subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        with self._lock:
            if subscription in self._subscribers:
                self._subscribers.remove(subscription)

    def snapshot(self, with_events: bool = True) -> JobRecord:
        with self._lock:
            record = self._record.model_copy(deep=True)
            if with_events:
                record.events = list(self._events)
            return record

    def persist(self) -> None:
        with self._lock:
            revision = self._revision
            payload = self.snapshot().model_dump_json(indent=2)
        with self._write_lock:
            if revision <= self._written_revision:
                return
            try:
                write_atomic(self.run_dir / JOB_FILE, payload)
            except OSError as exc:
                log.warning("could not persist job %s: %s", self.run_id, exc)
                return
            self._written_revision = revision

    def _append(self, event: Event) -> None:
        self._record.sequence += 1
        self._events.append(event)
        self._revision += 1
        self._broadcast(Envelope(self._record.sequence, event))

    def _finish(self, status: JobStatus, error: str | None) -> None:
        record = self._record
        record.status = status
        record.error = error
        record.finished_at = utc_now()
        if status == "done":
            record.progress = 1.0
        self._revision += 1
        self._broadcast(StreamEnd(status, error))
        self._subscribers = []

    def _broadcast(self, item: StreamItem) -> None:
        self._subscribers = [subscription for subscription in self._subscribers if subscription.push(item)]

    def _has_event(self, kind: str) -> bool:
        return any(event.type == kind for event in self._events)

    def _history(self) -> list[Envelope]:
        first = self._record.sequence - len(self._events) + 1
        return [Envelope(first + offset, event) for offset, event in enumerate(self._events)]


def run_pipeline(brief: Brief, run_id: str, settings: Settings, bus: EventBus) -> RunManifest:
    from mitosis.pipeline import Pipeline

    return Pipeline(settings, bus).run(brief, run_id)


def run_estimate(brief: Brief, settings: Settings, bus: EventBus) -> dict[str, Any]:
    from mitosis.pipeline import estimate

    return estimate(brief, settings, bus)


def record_from_manifest(path: Path, run_dir: Path) -> JobRecord | None:
    try:
        data = json.loads(path.read_text())
        brief = Brief.model_validate(data["brief"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning("unreadable manifest %s: %s", path, exc)
        return None
    totals = data.get("totals") or {}
    variants = int(totals.get("variants") or 0)
    created_at = data.get("created_at") or datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds")
    return JobRecord(
        run_id=run_dir.name,
        kind="run",
        status="done",
        created_at=created_at,
        finished_at=created_at,
        run_dir=run_dir,
        brief=summarize_brief(brief),
        progress=1.0,
        variants_total=variants,
        variants_done=variants,
        stages=[StageState(name=name, status="done") for name in RUN_STAGES],
        totals=totals,
        mode=data.get("mode"),
    )


def load_record(run_dir: Path) -> JobRecord | None:
    job_file = run_dir / JOB_FILE
    manifest_file = run_dir / MANIFEST_FILE
    if job_file.is_file():
        try:
            record = JobRecord.model_validate_json(job_file.read_text())
        except (OSError, ValueError) as exc:
            log.warning("unreadable job file %s: %s", job_file, exc)
        else:
            record.run_dir = run_dir
            if record.status not in TERMINAL_STATES:
                interrupted(record, finished=manifest_file.is_file())
            return record
    if manifest_file.is_file():
        return record_from_manifest(manifest_file, run_dir)
    return None


def interrupted(record: JobRecord, finished: bool) -> None:
    if finished:
        record.status = "done"
        record.progress = 1.0
        return
    record.status = "error"
    record.error = INTERRUPTED_MESSAGE
    for stage in record.stages:
        if stage.status == "running":
            stage.status = "error"


class JobManager:
    def __init__(self, settings: Settings, max_workers: int = 2) -> None:
        self.settings = settings
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mitosis-job")

    @property
    def runs_dir(self) -> Path:
        return Path(self.settings.runs_dir)

    def reserve_run_id(self, winner_id: str) -> str:
        base = f"{datetime.now():%Y%m%d-%H%M%S}-{winner_id}"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        candidates = itertools.chain([base], (f"{base}-{suffix}" for suffix in itertools.count(2)))
        return next(candidate for candidate in candidates if self._claim(candidate))

    def submit(self, brief: Brief, kind: JobKind, run_id: str) -> Job:
        run_dir = Path(brief.out or self.runs_dir / run_id).resolve()
        if kind == "estimate" and brief.out is None:
            brief = brief.model_copy(update={"out": run_dir})
        record = JobRecord(
            run_id=run_id,
            kind=kind,
            created_at=utc_now(),
            run_dir=run_dir,
            brief=summarize_brief(brief),
            variants_total=brief.variants,
            stages=initial_stages(kind),
        )
        job = Job(record)
        with self._lock:
            self._jobs[run_id] = job
        job.persist()
        self._executor.submit(self._execute, job, brief)
        log.info("queued %s job %s", kind, run_id)
        return job

    def get(self, run_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(run_id)

    def record(self, run_id: str) -> JobRecord | None:
        job = self.get(run_id)
        if job is not None:
            return job.snapshot()
        run_dir = self.run_dir(run_id)
        return load_record(run_dir) if run_dir else None

    def records(self) -> list[JobRecord]:
        with self._lock:
            jobs = list(self._jobs.values())
        found = {job.run_id: job.snapshot(with_events=False) for job in jobs}
        for folder in self._run_folders():
            if folder.name in found:
                continue
            record = load_record(folder)
            if record is not None:
                record.events = []
                found[folder.name] = record
        return sorted(found.values(), key=lambda record: (record.created_at, record.run_id), reverse=True)

    def run_dir(self, run_id: str) -> Path | None:
        job = self.get(run_id)
        if job is not None:
            return job.run_dir
        if not is_valid_run_id(run_id):
            return None
        candidate = self.runs_dir / run_id
        if (candidate / JOB_FILE).is_file() or (candidate / MANIFEST_FILE).is_file():
            return candidate
        return None

    def counts(self) -> dict[str, int]:
        with self._lock:
            statuses = [job.status for job in self._jobs.values()]
        return {status: statuses.count(status) for status in ("queued", "running", "done", "error")}

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _execute(self, job: Job, brief: Brief) -> None:
        bus = EventBus()
        bus.subscribe(job.handle)
        job.start()
        try:
            if job.kind == "estimate":
                job.complete_estimate(run_estimate(brief, self.settings, bus))
            else:
                job.complete_run(run_pipeline(brief, job.run_id, self.settings, bus))
        except Exception as exc:
            log.exception("job %s failed", job.run_id)
            job.fail(describe_error(exc))

    def _claim(self, run_id: str) -> bool:
        try:
            (self.runs_dir / run_id).mkdir()
        except FileExistsError:
            return False
        return True

    def _run_folders(self) -> list[Path]:
        if not self.runs_dir.is_dir():
            return []
        return sorted(child for child in self.runs_dir.iterdir() if child.is_dir() and is_valid_run_id(child.name))
