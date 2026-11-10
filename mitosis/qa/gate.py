from __future__ import annotations

import contextvars
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from mitosis.config import Brand, Settings
from mitosis.models import Check, GateInput, HeroPack, PolicyFlag, QAReport, VariantSpec
from mitosis.qa import judge, safezones, technical

log = logging.getLogger(__name__)

Decision = Literal["accept", "retry_shot", "retry_render", "flag"]
MAX_LISTED_FLAGS = 5


def guarded(stage: str, run: Callable[[], list[Check]]) -> list[Check]:
    try:
        return run()
    except Exception as exc:
        log.exception("qa stage %s crashed", stage)
        return [Check(name="qa_error", passed=False, severity="warn", detail=f"{stage} crashed: {type(exc).__name__}: {exc}")]


def collect_flags(inp: GateInput, brand: Brand) -> list[tuple[str, PolicyFlag]]:
    from mitosis.intelligence.policy import lint

    seen: set[tuple[str, str]] = set()
    flags: list[tuple[str, PolicyFlag]] = []
    for source, text in (("spoken", inp.spoken_text), ("on-screen", inp.on_screen_text)):
        if not text.strip():
            continue
        for flag in lint(text, brand):
            key = (flag.rule, flag.span.strip().lower())
            if key not in seen:
                seen.add(key)
                flags.append((source, flag))
    return flags


def describe_flag(source: str, flag: PolicyFlag) -> str:
    span = f" '{flag.span}'" if flag.span else ""
    return f"{source} {flag.rule}{span}: {flag.message}"


def policy_checks(inp: GateInput, brand: Brand) -> list[Check]:
    try:
        flags = collect_flags(inp, brand)
    except (ImportError, NotImplementedError) as exc:
        reason = "not implemented" if isinstance(exc, NotImplementedError) else str(exc)
        return [Check(name="policy_final", passed=False, skipped=True, detail=f"policy linter unavailable ({reason})")]
    blocking = [(source, flag) for source, flag in flags if flag.severity == "block"]
    warnings = [(source, flag) for source, flag in flags if flag.severity != "block"]
    listed = blocking or warnings
    summary = "; ".join(describe_flag(source, flag) for source, flag in listed[:MAX_LISTED_FLAGS])
    if blocking:
        detail = f"{len(blocking)} blocking policy flag(s): {summary}"
    elif warnings:
        detail = f"no blocking flags, {len(warnings)} warning(s): {summary}"
    else:
        detail = "spoken and on-screen text pass the policy linter"
    return [Check(name="policy_final", passed=not blocking, retry="none", score=float(len(blocking)), threshold=0.0, detail=detail)]


def evaluate(inp: GateInput, spec: VariantSpec, hero: HeroPack, brand: Brand, settings: Settings, llm=None) -> QAReport:
    stages: list[tuple[str, Callable[[], list[Check]]]] = [
        ("technical", lambda: technical.check_render(inp, brand)),
        ("safe_zones", lambda: [safezones.check_text_placement(inp, brand)]),
        ("policy_final", lambda: policy_checks(inp, brand)),
        ("vision_judge", lambda: judge.judge(inp, spec, hero, llm)),
    ]
    with ThreadPoolExecutor(max_workers=len(stages), thread_name_prefix="qa-gate") as pool:
        futures = [pool.submit(contextvars.copy_context().run, guarded, stage, run) for stage, run in stages]
        checks = [check for future in futures for check in future.result()]
    report = QAReport(variant_id=inp.variant_id, aspect=inp.aspect, checks=checks)
    failing = [check.name for check in checks if not check.passed and not check.skipped]
    log.info(
        "qa %s %s: %s%s",
        inp.variant_id,
        inp.aspect.value,
        "passed" if report.passed else "blocked",
        f" (failing: {', '.join(failing)})" if failing else "",
    )
    return report


def decide(reports: list[QAReport], attempt: int, max_attempts: int) -> Literal["accept", "retry_shot", "retry_render", "flag"]:
    if reports and all(report.passed for report in reports):
        return "accept"
    blocking = [check for report in reports for check in report.blocking]
    decision: Decision = "flag"
    if attempt < max_attempts and any(check.retry == "shot" for check in blocking):
        decision = "retry_shot"
    elif attempt < max_attempts and any(check.retry == "render" for check in blocking):
        decision = "retry_render"
    log.info(
        "qa decision %s on attempt %d/%d (blocking: %s)",
        decision,
        attempt,
        max_attempts,
        ", ".join(sorted({check.name for check in blocking})) or "none",
    )
    return decision
