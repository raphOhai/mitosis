from __future__ import annotations

import logging
import os
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import datetime
from html import escape
from pathlib import Path
from urllib.parse import quote

from mitosis import __version__
from mitosis.models import Check, Hook, PolicyFlag, Render, RunManifest, Totals, VariantResult, VariantSpec
from mitosis.report.selection import is_launchable, ordered_renders, ordered_results

log = logging.getLogger(__name__)

Linker = Callable[[Path | None], str | None]

URL_SAFE = "/-_.~"
CHAIN_ORDER = ("llm", "image", "video", "speech", "voice", "transcribe", "lipsync", "upscale")
KIND_LABELS = {"llm": "LLM", "lipsync": "Lip sync"}
FACTORS = ("hook", "voice", "location")
STATUS_LABELS = {"passed": "Passed", "flagged": "Flagged", "failed": "Failed"}
STATUS_ICONS = {"passed": "pass", "flagged": "warn", "failed": "fail"}
BODY_MODES = {"keep": "Original voice", "sts": "Voice converted", "tts": "Re-voiced"}
EXPORT_LABELS = {"meta_csv": "Meta bulk sheet", "after_effects": "After Effects script", "bundle": "ZIP bundle"}
MODE_NOTES = {
    "offline": (
        "Offline: every generative step ran on local stand-in models (system speech, pitch-shift voice "
        "conversion, Ken Burns motion, ffmpeg), so the pipeline runs end to end without API keys. Actual spend "
        "is what providers billed; production-equivalent spend prices every call at the production model it "
        "stands in for. Lip-sync and the Claude calls only exist on the live path, so budget with the plan "
        "estimate, which includes them."
    ),
    "hybrid": (
        "Hybrid: some kinds ran on production APIs and the rest fell back to local stand-ins. Stand-in calls "
        "are priced at production rates in the production-equivalent spend."
    ),
    "live": "Live: every generative step ran on production APIs, so actual spend is billed spend.",
}
ICONS = {
    "pass": '<path d="M3.5 8.5l3 3 6-7"/>',
    "fail": '<path d="M4.5 4.5l7 7M11.5 4.5l-7 7"/>',
    "skip": '<path d="M4 8h8"/>',
    "warn": '<path d="M8 3.5v5.5M8 12.2v.3"/>',
}


class RelativeLinks:
    def __init__(self, base: Path) -> None:
        self.base = Path(base).resolve()

    def __call__(self, path: Path | None) -> str | None:
        if path is None:
            return None
        target = Path(path).resolve()
        try:
            relative = os.path.relpath(target, self.base)
        except ValueError:
            return target.as_uri()
        return quote(Path(relative).as_posix(), safe=URL_SAFE)


class MappedLinks:
    def __init__(self, locations: dict[Path, str]) -> None:
        self.locations = {Path(source).resolve(): target for source, target in locations.items()}

    def __call__(self, path: Path | None) -> str | None:
        if path is None:
            return None
        target = self.locations.get(Path(path).resolve())
        return quote(target, safe=URL_SAFE) if target else None


def write_report(manifest: RunManifest, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(manifest, RelativeLinks(out.parent)), encoding="utf-8")
    log.info("report: %d variants -> %s", len(manifest.results), out)
    return out


def render_report(manifest: RunManifest, link: Linker) -> str:
    sections = [
        masthead(manifest),
        outcome_section(manifest),
        dna_section(manifest, link),
        hooks_section(manifest),
        plan_section(manifest),
        gallery_section(manifest, link),
        spend_section(manifest.totals),
        footer(manifest, link),
    ]
    title = f"Mitosis report {manifest.winner.id} {manifest.run_id}"
    return "".join(
        [
            '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n',
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n',
            '<meta name="color-scheme" content="light dark">\n',
            f'<meta name="generator" content="Mitosis {escape(__version__)}">\n',
            f'<title>{text(title)}</title>\n<style>{STYLE}</style>\n</head>\n<body>\n<main class="page">\n',
            "\n".join(sections),
            f"\n</main>\n<script>{SCRIPT}</script>\n</body>\n</html>\n",
        ]
    )


def text(value: object) -> str:
    return "" if value is None else escape(str(value), quote=True)


def money(value: float) -> str:
    if 0 < abs(value) < 0.01:
        return f"${value:.4f}"
    return f"${value:,.2f}"


def duration_label(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def percent(part: float, whole: float) -> str:
    return f"{part / whole:.0%}" if whole else "n/a"


def ratio_label(value: float) -> str:
    return f"{value:.0%}" if 0 <= value <= 1 else f"{value:g}"


def score_fraction(score: float) -> float:
    value = score / 100 if score > 1 else score
    return min(1.0, max(0.0, value))


def humanize(key: str) -> str:
    words = key.replace("_", " ").replace("-", " ").split()
    return " ".join(words).capitalize() if words else ""


def timestamp_label(value: str) -> str:
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return value
    zone = f" {moment.tzname()}" if moment.tzinfo else ""
    return f"{moment:%b %d, %Y %H:%M}{zone}"


def icon(kind: str) -> str:
    return f'<svg class="icon" viewBox="0 0 16 16" aria-hidden="true">{ICONS[kind]}</svg>'


def status_badge(status: str) -> str:
    kind = STATUS_ICONS.get(status, "skip")
    return f'<span class="badge badge-{kind}">{icon(kind)}{text(STATUS_LABELS.get(status, humanize(status)))}</span>'


def chip(value: str, extra: str = "") -> str:
    css = f"chip {extra}" if extra else "chip"
    return f'<span class="{css}">{text(value)}</span>'


def meter(fraction: float, wide: bool = False) -> str:
    width = f"{fraction * 100:.1f}%"
    return f'<span class="meter{" meter-wide" if wide else ""}" aria-hidden="true"><span style="width:{width}"></span></span>'


def section(anchor: str, kicker: str, title: str, body: str, lede: str = "") -> str:
    intro = f'<p class="section-lede">{lede}</p>' if lede else ""
    return (
        f'<section id="{anchor}" class="section">\n<p class="kicker">{text(kicker)}</p>\n'
        f"<h2>{text(title)}</h2>\n{intro}\n{body}\n</section>"
    )


def table(headers: Iterable[str], rows: Iterable[str], css: str = "") -> str:
    head = "".join(f'<th scope="col">{text(h)}</th>' for h in headers)
    return f'<div class="table-wrap"><table class="{css}"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def masthead(manifest: RunManifest) -> str:
    winner = manifest.winner
    info = winner.info
    source = f"{Path(winner.video).name} · {info.duration:.1f} s · {info.width}×{info.height} · {info.fps:g} fps"
    facts = [
        ("Run", manifest.run_id),
        ("Winner", winner.id),
        ("Created", timestamp_label(manifest.created_at)),
        ("Source", source),
    ]
    facts_html = "".join(f"<div><dt>{text(label)}</dt><dd>{text(value)}</dd></div>" for label, value in facts)
    count = len(manifest.results) or len(manifest.plan.variants)
    nav = "".join(
        f'<a href="#{anchor}">{text(label)}</a>'
        for anchor, label in (
            ("outcome", "Outcome"),
            ("dna", "Creative DNA"),
            ("hooks", "Hooks"),
            ("plan", "Experiment plan"),
            ("variants", "Variants"),
            ("spend", "Spend"),
        )
    )
    return (
        '<header class="masthead" id="top">\n<p class="eyebrow">Mitosis run report</p>\n'
        f'<h1>{text(winner.id)} <span class="arrow" aria-hidden="true">&rarr;</span> {count} variants</h1>\n'
        '<p class="lede">One proven ad, re-cut with new hooks, voices and hook-shot locations in every Meta '
        "format while the same hero stays on screen.</p>\n"
        f'<dl class="facts">{facts_html}</dl>\n{mode_panel(manifest)}\n<nav class="toc">{nav}</nav>\n</header>'
    )


def mode_panel(manifest: RunManifest) -> str:
    note = MODE_NOTES.get(manifest.mode, "")
    return (
        f'<div class="mode mode-{text(manifest.mode)}">\n<p><span class="mode-badge">{text(manifest.mode)}</span> {text(note)}</p>\n'
        f"{chain_table(manifest.models)}\n</div>"
    )


def chain_order(models: dict[str, list[str]]) -> list[str]:
    known = [kind for kind in CHAIN_ORDER if kind in models]
    return known + sorted(kind for kind in models if kind not in CHAIN_ORDER)


def model_label(name: str) -> str:
    local = '<span class="tag">local stand-in</span>' if name.startswith("offline") else ""
    return f"<code>{text(name)}</code>{local}"


def chain_table(models: dict[str, list[str]]) -> str:
    if not models:
        return '<p class="muted">No model chains were recorded for this run.</p>'
    rows = []
    for kind in chain_order(models):
        chain = models[kind]
        served = model_label(chain[0]) if chain else '<span class="muted">unavailable</span>'
        fallbacks = " &rarr; ".join(model_label(name) for name in chain[1:]) or '<span class="muted">none</span>'
        label = KIND_LABELS.get(kind, humanize(kind))
        rows.append(f'<tr><th scope="row">{text(label)}</th><td>{served}</td><td>{fallbacks}</td></tr>')
    return table(("Kind", "Served by", "Fallback chain"), rows, "chains")


def outcome_counts(manifest: RunManifest) -> dict[str, int]:
    totals = manifest.totals
    if totals.variants:
        return {"variants": totals.variants, "passed": totals.passed, "flagged": totals.flagged, "failed": totals.failed}
    counts = Counter(result.status for result in manifest.results)
    return {"variants": len(manifest.results), "passed": counts["passed"], "flagged": counts["flagged"], "failed": counts["failed"]}


def tile(label: str, value: str, note: str = "", extra: str = "") -> str:
    note_html = f'<p class="tile-note">{note}</p>' if note else ""
    return f'<div class="tile"><p class="tile-label">{text(label)}</p><p class="tile-value">{value}</p>{extra}{note_html}</div>'


def outcome_tile(counts: dict[str, int]) -> str:
    items = "".join(
        f'<li class="status-{status}">{icon(STATUS_ICONS[status])}<b>{counts[status]}</b> {text(status)}</li>'
        for status in ("passed", "flagged", "failed")
    )
    return f'<div class="tile"><p class="tile-label">Outcome</p><ul class="outcome">{items}</ul></div>'


def outcome_section(manifest: RunManifest) -> str:
    totals = manifest.totals
    counts = outcome_counts(manifest)
    variants = counts["variants"]
    ready = sum(1 for result in manifest.results if is_launchable(result))
    renders = totals.renders or sum(len(result.renders) for result in manifest.results)
    formats = len(manifest.brief.aspects)
    pass_fraction = counts["passed"] / variants if variants else 0.0
    tiles = [
        tile("Variants", str(variants), text(f"{renders} renders in {formats} formats")),
        outcome_tile(counts),
        tile("Pass rate", percent(counts["passed"], variants), "passed every blocking QA gate", meter(pass_fraction, wide=True)),
        tile("Ready to launch", str(ready), "passed or approved, none rejected"),
        tile("Actual spend", money(totals.usd), "billed by providers"),
        tile("Production-equivalent", money(totals.shadow_usd), "the same calls at production prices"),
        tile("Cost per variant", money(totals.usd_per_variant), text(f"{money(totals.shadow_per_variant)} at production prices")),
        tile("Wall time", duration_label(totals.seconds), "brief to bundle"),
        tile("Time per variant", duration_label(totals.seconds_per_variant), "wall time / variants"),
        tile("Cache hits", str(totals.cache_hits), "generations reused, not re-billed"),
        tile("Saved by cache", money(totals.saved_usd), "spend avoided by reuse"),
    ]
    body = f'<div class="tiles">{"".join(tiles)}</div>'
    return section("outcome", "01", "Outcome in numbers", body)


def dna_rows(manifest: RunManifest) -> list[tuple[str, str]]:
    dna = manifest.dna
    hook = f"&ldquo;{text(dna.hook_text)}&rdquo; {chip(dna.hook_archetype)}"
    rows = [
        ("Angle", text(dna.angle)),
        ("Audience", text(dna.audience)),
        ("Promise", text(dna.promise)),
        ("Original hook", hook),
        ("Format", text(humanize(dna.format))),
        ("Pacing", text(f"{dna.pacing_wps:.1f} words per second")),
        ("Tone", text(dna.tone)),
        ("Call to action", text(dna.cta_text)),
    ]
    return [(label, value) for label, value in rows if value]


def hero_gallery(manifest: RunManifest, link: Linker) -> str:
    hero = manifest.hero
    images = []
    for index, reference in enumerate(hero.references, start=1):
        source = link(reference)
        if source:
            images.append(f'<img src="{text(source)}" alt="Hero reference {index}" loading="lazy">')
    grid = f'<div class="hero-grid">{"".join(images)}</div>' if images else '<p class="muted">No hero reference frames are available.</p>'
    lock = text(hero.embedding_model or "locked") if hero.embedding else "unavailable"
    return (
        f'<figure class="hero">{grid}<figcaption><b>Hero lock.</b> {text(hero.description)} '
        f'<span class="muted">Identity embedding: {lock}.</span></figcaption></figure>'
    )


def dna_section(manifest: RunManifest, link: Linker) -> str:
    rows = "".join(f"<div><dt>{text(label)}</dt><dd>{value}</dd></div>" for label, value in dna_rows(manifest))
    beats = "".join(f"<li>{text(beat)}</li>" for beat in manifest.dna.beats)
    beats_html = f'<h3>Beats</h3><ol class="beats">{beats}</ol>' if beats else ""
    body = f'<div class="dna"><div><dl class="dna-list">{rows}</dl>{beats_html}</div>{hero_gallery(manifest, link)}</div>'
    return section("dna", "02", "Creative DNA", body, "What made the winner work, extracted before anything was changed.")


def flag_notes(flags: list[PolicyFlag]) -> str:
    return "".join(f'<p class="flag-note">{icon("warn")}{text(flag.rule)}: {text(flag.message)}</p>' for flag in flags)


def hook_row(hook: Hook) -> str:
    return (
        f'<tr><td class="mono">{text(hook.id)}</td><td>{chip(hook.archetype)}</td>'
        f"<td>{text(hook.text)}{flag_notes(hook.flags)}</td><td>{text(hook.on_screen_text)}</td>"
        f'<td class="num">{meter(score_fraction(hook.score))} {hook.score:.2f}</td></tr>'
    )


def blocked_flag(flag: PolicyFlag) -> str:
    matched = f'<p class="flag-span">Matched: &ldquo;{text(flag.span)}&rdquo;</p>' if flag.span else ""
    suggestion = f'<p class="flag-suggestion"><b>Rewrite:</b> {text(flag.suggestion)}</p>' if flag.suggestion else ""
    kind = "fail" if flag.severity == "block" else "warn"
    return (
        f'<li><p class="flag-head"><span class="badge badge-{kind}">{icon(kind)}{text(flag.severity)}</span>'
        f"<code>{text(flag.rule)}</code></p><p>{text(flag.message)}</p>{matched}{suggestion}</li>"
    )


def blocked_card(hook: Hook) -> str:
    flags = "".join(blocked_flag(flag) for flag in hook.flags) or '<li class="muted">Rejected on quality, no policy flag.</li>'
    on_screen = f'<p class="muted">On screen: {text(hook.on_screen_text)}</p>' if hook.on_screen_text else ""
    return (
        f'<article class="blocked"><header><span class="mono">{text(hook.id)}</span>{chip(hook.archetype)}</header>'
        f'<p class="quote">&ldquo;{text(hook.text)}&rdquo;</p>{on_screen}<ul class="flags">{flags}</ul></article>'
    )


def hooks_section(manifest: RunManifest) -> str:
    accepted = (
        table(("ID", "Archetype", "Spoken", "On screen", "Score"), [hook_row(h) for h in manifest.hooks], "hooks")
        if manifest.hooks
        else '<p class="muted">No hooks were accepted.</p>'
    )
    blocked = (
        f'<div class="blocked-grid">{"".join(blocked_card(h) for h in manifest.rejected_hooks)}</div>'
        if manifest.rejected_hooks
        else '<p class="muted">No hook was blocked in this run.</p>'
    )
    body = (
        f'<h3>Accepted <span class="count">{len(manifest.hooks)}</span></h3>{accepted}'
        f'<h3>Blocked before Meta could reject them <span class="count">{len(manifest.rejected_hooks)}</span></h3>'
        '<p class="muted">Each line was stopped by the ad-policy screen before a single frame was generated.</p>'
        f"{blocked}"
    )
    return section("hooks", "03", "Hooks", body)


def factor_changed(spec: VariantSpec, factor: str) -> bool:
    return any(factor in item.lower() for item in spec.changed)


def factor_cell(spec: VariantSpec, factor: str, label: str) -> str:
    changed = factor_changed(spec, factor)
    css = ' class="changed"' if changed else ""
    marker = '<span class="sr"> (changed)</span>' if changed else ""
    return f"<td{css}>{label}{marker}</td>"


def plan_row(spec: VariantSpec, result: VariantResult | None) -> str:
    hook = f'<span class="mono">{text(spec.hook.id)}</span> {text(spec.hook.archetype)}'
    outcome = status_badge(result.status) if result else '<span class="muted">not run</span>'
    return (
        f'<tr><td class="mono">{text(spec.id)}</td>'
        f"{factor_cell(spec, 'hook', hook)}"
        f"{factor_cell(spec, 'voice', text(spec.voice.label or spec.voice.id))}"
        f"{factor_cell(spec, 'location', text(spec.location.label or spec.location.id))}"
        f"<td>{text(BODY_MODES.get(spec.body_mode, spec.body_mode))}</td>"
        f"<td>{'New' if spec.new_shot else 'Original'}</td><td>{outcome}</td></tr>"
    )


def factor_usage(plan_levels: dict[str, list[str]], specs: list[VariantSpec]) -> list[tuple[str, str]]:
    used = {
        "hook": {s.hook.id for s in specs},
        "voice": {s.voice.id for s in specs},
        "location": {s.location.id for s in specs},
    }
    stats = []
    for factor in FACTORS:
        levels = plan_levels.get(factor)
        value = f"{len(used[factor])} of {len(levels)}" if levels else str(len(used[factor]))
        stats.append((f"{humanize(factor)}s used", value))
    stats += [(f"{humanize(factor)} levels", str(len(levels))) for factor, levels in plan_levels.items() if factor not in FACTORS]
    return stats


def plan_stats(manifest: RunManifest, specs: list[VariantSpec]) -> list[tuple[str, str]]:
    plan = manifest.plan
    stats = [("Design", humanize(plan.design) or "n/a"), ("Variants", str(len(specs)))]
    stats += factor_usage(plan.levels, specs)
    stats += [(f"{humanize(name)} coverage", ratio_label(value)) for name, value in plan.coverage.items()]
    stats.append(("Planned cost", f"{money(plan.estimate.usd)} at production prices"))
    if plan.estimate.per_variant_usd:
        stats.append(("Planned per variant", money(plan.estimate.per_variant_usd)))
    return stats


def plan_section(manifest: RunManifest) -> str:
    specs = manifest.plan.variants or [result.spec for result in manifest.results]
    results = {result.spec.id: result for result in manifest.results}
    stats = "".join(f"<div><dt>{text(label)}</dt><dd>{text(value)}</dd></div>" for label, value in plan_stats(manifest, specs))
    matrix = (
        table(
            ("Variant", "Hook", "Voice", "Location", "Body", "Hook shot", "Outcome"),
            [plan_row(spec, results.get(spec.id)) for spec in sorted(specs, key=lambda s: s.index)],
            "matrix",
        )
        if specs
        else '<p class="muted">The planner produced no variants.</p>'
    )
    notes = "".join(f"<li>{text(note)}</li>" for note in manifest.plan.notes)
    notes_html = f'<h3>Planner notes</h3><ul class="notes">{notes}</ul>' if notes else ""
    body = (
        f'<dl class="stats">{stats}</dl>'
        '<p class="legend"><span class="swatch changed"></span>Highlighted cells are the factors this variant changes against the winner.</p>'
        f"{matrix}{notes_html}"
    )
    return section("plan", "04", "Experiment plan", body)


def check_state(check: Check) -> str:
    if check.skipped:
        return "skip"
    if check.passed:
        return "pass"
    return "warn" if check.severity == "warn" else "fail"


def check_title(check: Check) -> str:
    parts = [check.detail] if check.detail else []
    if check.score is not None:
        threshold = f" / {check.threshold:g}" if check.threshold is not None else ""
        parts.append(f"score {check.score:.3g}{threshold}")
    return " · ".join(parts) or check_state(check)


def check_mark(check: Check | None) -> str:
    if check is None:
        return '<td class="mark-cell"><span class="muted">n/a</span></td>'
    state = check_state(check)
    return (
        f'<td class="mark-cell"><span class="mark mark-{state}" title="{text(check_title(check))}">'
        f'{icon(state)}<span class="sr">{state}</span></span></td>'
    )


def qa_matrix(renders: list[Render]) -> str:
    reports = [(render.aspect.value, {check.name: check for check in render.qa.checks} if render.qa else {}) for render in renders]
    names = list(dict.fromkeys(name for _, checks in reports for name in checks))
    if not names:
        return '<p class="muted small">No QA checks were recorded.</p>'
    rows = [
        f'<tr><th scope="row">{text(humanize(name))}</th>{"".join(check_mark(checks.get(name)) for _, checks in reports)}</tr>'
        for name in names
    ]
    return table(["QA check"] + [aspect for aspect, _ in reports], rows, "qa")


def player(render: Render, link: Linker) -> str:
    source = link(render.path)
    slug = render.aspect.slug
    verdict = ""
    if render.qa is not None:
        kind = "pass" if render.qa.passed else "fail"
        verdict = f'<span class="mark mark-{kind}" title="QA {kind}">{icon(kind)}<span class="sr">QA {kind}</span></span>'
    if source is None:
        frame = f'<div class="frame ar-{slug} placeholder">Not included</div>'
        label = text(render.aspect.value)
    else:
        poster_url = link(render.thumbnail) if render.thumbnail else None
        poster = f' poster="{text(poster_url)}"' if poster_url else ""
        frame = f'<video class="frame ar-{slug}" controls preload="none" playsinline src="{text(source)}"{poster}></video>'
        label = f'<a href="{text(source)}">{text(render.aspect.value)}</a>'
    return f'<figure class="player" title="{text(render.name)}">{frame}<figcaption>{label}{verdict}</figcaption></figure>'


def prelaunch_block(result: VariantResult) -> str:
    if result.prelaunch is None:
        return '<div class="prelaunch"><p class="muted small">No prelaunch score.</p></div>'
    fraction = score_fraction(result.prelaunch.score)
    reasons = "".join(f"<li>{text(reason)}</li>" for reason in result.prelaunch.reasons)
    reasons_html = f'<ul class="reasons">{reasons}</ul>' if reasons else ""
    return (
        f'<div class="prelaunch"><p class="prelaunch-head"><span>Prelaunch score</span><b>{fraction * 100:.0f}</b></p>'
        f"{meter(fraction, wide=True)}{reasons_html}</div>"
    )


def cost_block(result: VariantResult) -> str:
    items = [
        ("Actual", money(result.usd)),
        ("Production", money(result.shadow_usd)),
        ("Time", duration_label(result.seconds)),
        ("Attempts", str(result.attempts)),
    ]
    return '<dl class="cost">' + "".join(f"<div><dt>{label}</dt><dd>{value}</dd></div>" for label, value in items) + "</dl>"


def variant_card(result: VariantResult, link: Linker) -> str:
    spec = result.spec
    renders = ordered_renders(result)
    review = chip(result.review, f"review-{result.review}") if result.review != "pending" else ""
    players = "".join(player(render, link) for render in renders) or '<p class="muted small">No renders were produced.</p>'
    error = f'<p class="error">{text(result.error)}</p>' if result.error else ""
    tags = "".join(
        chip(value) for value in (spec.hook.archetype, spec.voice.label or spec.voice.id, spec.location.label or spec.location.id)
    )
    return (
        f'<article class="variant" data-status="{text(result.status)}" id="variant-{text(spec.id)}">'
        f'<header class="variant-head"><h3>{text(spec.id)}</h3>{status_badge(result.status)}{review}</header>'
        f'<p class="quote">&ldquo;{text(spec.hook.text)}&rdquo;</p><p class="tags">{tags}</p>'
        f'<div class="players">{players}</div>{error}'
        f'<div class="variant-body">{qa_matrix(renders)}<div>{prelaunch_block(result)}{cost_block(result)}</div></div>'
        "</article>"
    )


def gallery_section(manifest: RunManifest, link: Linker) -> str:
    results = ordered_results(manifest.results)
    if not results:
        return section("variants", "05", "Variants", '<p class="muted">No variants were produced.</p>')
    counts = Counter(result.status for result in results)
    buttons = [f'<button type="button" data-filter="all" aria-pressed="true">All <span>{len(results)}</span></button>']
    buttons += [
        f'<button type="button" data-filter="{status}" aria-pressed="false">{label} <span>{counts[status]}</span></button>'
        for status, label in STATUS_LABELS.items()
        if counts[status]
    ]
    body = (
        f'<div class="filters" role="group" aria-label="Filter variants by status">{"".join(buttons)}</div>'
        f'<div class="gallery">{"".join(variant_card(result, link) for result in results)}</div>'
    )
    return section("variants", "05", "Variants", body)


def provider_label(key: str) -> str:
    provider, _, model = key.partition(":")
    return f"{provider} · {model}" if model else provider


def bar_chart(title: str, values: dict[str, float], label: Callable[[str], str]) -> str:
    rows = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    if not rows:
        return f'<div class="chart"><h3>{text(title)}</h3><p class="muted">No spend recorded.</p></div>'
    peak = max(value for _, value in rows) or 1.0
    total = sum(value for _, value in rows) or 1.0
    bars = []
    for key, value in rows:
        width = f' style="width:{value / peak * 100:.1f}%"' if value > 0 else ""
        hint = f"{label(key)}: {money(value)} ({value / total:.0%} of total)"
        bars.append(
            f'<div class="bar-row" title="{text(hint)}"><span class="bar-label">{text(label(key))}</span>'
            f'<span class="bar-track"><span class="bar-fill"{width}></span></span><span class="bar-value">{money(value)}</span></div>'
        )
    return f'<div class="chart"><h3>{text(title)}</h3>{"".join(bars)}</div>'


def spend_section(totals: Totals) -> str:
    body = (
        '<div class="charts">'
        f"{bar_chart('By stage', totals.by_stage, humanize)}"
        f"{bar_chart('By provider and model', totals.by_provider, provider_label)}"
        "</div>"
    )
    lede = "Each call counts at its billed price, or at its production-equivalent price when a free local stand-in served it."
    return section("spend", "06", "Where the money went", body, lede)


def footer(manifest: RunManifest, link: Linker) -> str:
    root = Path(manifest.out_dir)
    targets = [("Manifest", root / "manifest.json"), ("Cost ledger", manifest.ledger or root / "ledger.jsonl")]
    targets += [(EXPORT_LABELS.get(key, humanize(key)), path) for key, path in manifest.exports.items()]
    links = []
    for label, path in targets:
        href = link(path)
        if href:
            links.append(f'<a href="{text(href)}">{text(label)} <code>{text(Path(path).name)}</code></a>')
    return (
        f'<footer class="footer"><nav class="downloads">{"".join(links)}</nav>'
        f'<p class="muted">Generated by Mitosis {text(__version__)} for run {text(manifest.run_id)}.</p></footer>'
    )


LIGHT_TOKENS = """
  color-scheme: light;
  --page: #f9f9f7;
  --surface: #fcfcfb;
  --surface-2: #f2f1ed;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #6f6d68;
  --line: #e1e0d9;
  --border: rgba(11, 11, 11, 0.10);
  --accent: #2a78d6;
  --accent-wash: rgba(42, 120, 214, 0.12);
  --track: #ebeae5;
  --good: #0ca30c;
  --warn: #c98500;
  --bad: #d03b3b;
  --good-wash: rgba(12, 163, 12, 0.10);
  --warn-wash: rgba(250, 178, 25, 0.16);
  --bad-wash: rgba(208, 59, 59, 0.10);
"""

DARK_TOKENS = """
  color-scheme: dark;
  --page: #0d0d0d;
  --surface: #1a1a19;
  --surface-2: #232321;
  --ink: #f0efec;
  --ink-2: #c3c2b7;
  --muted: #9a988f;
  --line: #2c2c2a;
  --border: rgba(255, 255, 255, 0.10);
  --accent: #3987e5;
  --accent-wash: rgba(57, 135, 229, 0.18);
  --track: #2c2c2a;
  --good: #0ca30c;
  --warn: #fab219;
  --bad: #e66767;
  --good-wash: rgba(12, 163, 12, 0.16);
  --warn-wash: rgba(250, 178, 25, 0.14);
  --bad-wash: rgba(230, 103, 103, 0.14);
"""

STYLE = (
    ":root {"
    + LIGHT_TOKENS
    + """
  --radius: 12px;
  --font: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
}
@media screen and (prefers-color-scheme: dark) {
  :root {"""
    + DARK_TOKENS
    + """  }
}
*, *::before, *::after { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.55 var(--font); -webkit-font-smoothing: antialiased; }
.page { max-width: 1200px; margin: 0 auto; padding: 40px 16px 64px; }
@media (min-width: 720px) { .page { padding: 56px 32px 80px; } }
h1, h2, h3 { line-height: 1.2; margin: 0; }
h1 { font-size: clamp(28px, 4.6vw, 44px); font-weight: 700; letter-spacing: -0.025em; }
h2 { font-size: clamp(21px, 2.6vw, 27px); font-weight: 650; letter-spacing: -0.015em; }
h3 { font-size: 15px; font-weight: 650; margin: 28px 0 12px; }
p { margin: 0; }
a { color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 2px; }
code, .mono { font-family: var(--mono); font-size: 0.86em; }
.muted { color: var(--muted); }
.small { font-size: 13px; }
.sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
.icon {
  width: 14px;
  height: 14px;
  flex: none;
  fill: none;
  stroke: currentColor;
  stroke-width: 2.2;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.eyebrow, .kicker {
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 8px;
}
.arrow { color: var(--muted); font-weight: 400; }
.lede { max-width: 62ch; margin-top: 12px; font-size: 17px; color: var(--ink-2); }
.section-lede { max-width: 70ch; margin-top: 8px; color: var(--ink-2); }
.masthead { padding-bottom: 32px; border-bottom: 1px solid var(--line); }
.facts { display: flex; flex-wrap: wrap; gap: 12px 32px; margin: 24px 0 0; }
.facts div, .stats div, .dna-list div, .cost div { min-width: 0; }
dt { font-size: 12px; color: var(--muted); }
dd { margin: 2px 0 0; overflow-wrap: anywhere; }
.facts dd { font-weight: 550; }
.mode { margin-top: 24px; padding: 16px 18px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); }
.mode > p { max-width: 90ch; color: var(--ink-2); }
.mode-badge {
  display: inline-block;
  margin-right: 6px;
  padding: 1px 8px;
  border-radius: 999px;
  background: var(--accent-wash);
  color: var(--ink);
  font-size: 12px;
  font-weight: 650;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.tag {
  display: inline-block;
  margin-left: 6px;
  padding: 0 6px;
  border-radius: 4px;
  background: var(--surface-2);
  color: var(--muted);
  font-size: 11px;
}
.toc { display: flex; flex-wrap: wrap; gap: 6px 18px; margin-top: 24px; font-size: 14px; }
.toc a { color: var(--ink-2); text-decoration: none; }
.toc a:hover { color: var(--accent); }
.section { padding: 44px 0 8px; }
.section > h2 { margin-bottom: 4px; }
.section > h2 + * { margin-top: 20px; }
.count {
  display: inline-block;
  min-width: 22px;
  padding: 0 7px;
  margin-left: 4px;
  border-radius: 999px;
  background: var(--surface-2);
  color: var(--ink-2);
  font-size: 12px;
  text-align: center;
}
.tiles { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 20px; }
@media (min-width: 720px) { .tiles { grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); } }
.tile { padding: 16px 18px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); }
.tile-label { font-size: 13px; color: var(--ink-2); }
.tile-value { margin-top: 6px; font-size: clamp(24px, 6vw, 30px); font-weight: 650; letter-spacing: -0.02em; line-height: 1.1; }
.tile-note { margin-top: 6px; font-size: 12.5px; color: var(--muted); }
.tile .meter-wide { margin-top: 10px; }
.outcome { list-style: none; margin: 8px 0 0; padding: 0; display: grid; gap: 3px; }
.outcome li { display: flex; align-items: center; gap: 8px; color: var(--ink-2); }
.outcome b { min-width: 2ch; color: var(--ink); font-size: 18px; font-variant-numeric: tabular-nums; }
.status-passed .icon { color: var(--good); }
.status-flagged .icon { color: var(--warn); }
.status-failed .icon { color: var(--bad); }
.meter {
  display: inline-block;
  width: 64px;
  height: 6px;
  border-radius: 3px;
  background: var(--track);
  vertical-align: middle;
  overflow: hidden;
}
.meter > span { display: block; height: 100%; border-radius: 3px; background: var(--accent); }
.meter-wide { display: block; width: 100%; }
.table-wrap {
  overflow-x: auto;
  margin-top: 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}
table { width: 100%; border-collapse: collapse; font-size: 13.5px; font-variant-numeric: tabular-nums; }
th, td { padding: 9px 12px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--line); }
thead th {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
  background: var(--surface-2);
  white-space: nowrap;
}
tbody tr:last-child > * { border-bottom: 0; }
tbody th { font-weight: 550; white-space: nowrap; }
td.num { white-space: nowrap; text-align: right; }
.chains td { white-space: nowrap; }
.chip {
  display: inline-block;
  padding: 1px 8px;
  margin: 0 4px 2px 0;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--surface-2);
  color: var(--ink-2);
  font-size: 12px;
  white-space: nowrap;
}
.badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 2px 9px 2px 7px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  color: var(--ink);
  white-space: nowrap;
}
.badge-pass { background: var(--good-wash); } .badge-pass .icon { color: var(--good); }
.badge-warn { background: var(--warn-wash); } .badge-warn .icon { color: var(--warn); }
.badge-fail { background: var(--bad-wash); } .badge-fail .icon { color: var(--bad); }
.badge-skip { background: var(--surface-2); }
.review-approved { border-color: var(--good); } .review-rejected { border-color: var(--bad); }
.dna { display: grid; gap: 28px; margin-top: 20px; }
@media (min-width: 900px) { .dna { grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr); } }
.dna-list { display: grid; gap: 14px; margin: 0; }
.dna-list dd { font-size: 15.5px; }
.beats { margin: 0; padding-left: 20px; color: var(--ink-2); }
.hero { margin: 0; }
.hero-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(96px, 1fr)); gap: 8px; }
.hero-grid img { width: 100%; aspect-ratio: 3 / 4; object-fit: cover; border-radius: 8px; background: var(--surface-2); display: block; }
.hero figcaption { margin-top: 10px; font-size: 13.5px; color: var(--ink-2); }
.flag-note { display: flex; gap: 6px; align-items: center; margin-top: 4px; font-size: 12.5px; color: var(--muted); }
.flag-note .icon { color: var(--warn); }
.blocked-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 340px), 1fr)); gap: 12px; margin-top: 12px; }
.blocked {
  padding: 16px 18px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 3px solid var(--bad);
  border-radius: var(--radius);
}
.blocked header { display: flex; gap: 8px; align-items: center; color: var(--muted); }
.quote { margin-top: 8px; font-size: 16px; line-height: 1.45; }
.blocked .quote { color: var(--ink-2); text-decoration: line-through; text-decoration-color: var(--bad); text-decoration-thickness: 1px; }
.flags { list-style: none; margin: 12px 0 0; padding: 0; display: grid; gap: 12px; font-size: 13.5px; }
.flag-head { display: flex; gap: 8px; align-items: center; margin-bottom: 4px; }
.flag-span { color: var(--muted); margin-top: 2px; }
.flag-suggestion { margin-top: 6px; padding: 8px 10px; border-radius: 8px; background: var(--good-wash); }
.stats { display: flex; flex-wrap: wrap; gap: 12px 28px; margin: 20px 0 0; }
.stats dd { font-weight: 600; }
.legend { display: flex; align-items: center; gap: 8px; margin-top: 18px; font-size: 13px; color: var(--muted); }
.swatch { width: 14px; height: 14px; border-radius: 4px; border: 1px solid var(--border); }
.changed { background: var(--accent-wash); box-shadow: inset 3px 0 0 var(--accent); }
.matrix td { white-space: nowrap; }
.notes { margin: 0; padding-left: 20px; color: var(--ink-2); }
.filters { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 20px; }
.filters button {
  font: inherit;
  font-size: 13px;
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--ink-2);
  cursor: pointer;
}
.filters button span { color: var(--muted); font-variant-numeric: tabular-nums; }
.filters button[aria-pressed="true"] { background: var(--ink); color: var(--page); border-color: var(--ink); }
.filters button[aria-pressed="true"] span { color: inherit; opacity: 0.7; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 540px), 1fr)); gap: 16px; margin-top: 16px; }
.variant { padding: 18px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); min-width: 0; }
.variant[hidden] { display: none; }
.variant-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.variant-head h3 { margin: 0 4px 0 0; font-size: 18px; }
.tags { margin-top: 8px; }
.players { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 12px; margin-top: 14px; }
.player { margin: 0; }
.frame { display: block; height: 200px; border-radius: 8px; background: #000; object-fit: cover; }
.ar-9x16 { width: 113px; } .ar-4x5 { width: 160px; } .ar-1x1 { width: 200px; }
.placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--surface-2);
  color: var(--muted);
  font-size: 12px;
  text-align: center;
  padding: 8px;
}
.player figcaption { display: flex; align-items: center; gap: 6px; margin-top: 6px; font-size: 12.5px; font-variant-numeric: tabular-nums; }
.variant-body { display: grid; align-items: start; gap: 16px; margin-top: 16px; }
@media (min-width: 640px) { .variant-body { grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); } }
.variant-body .table-wrap { margin-top: 0; }
.qa th, .qa td { padding: 6px 10px; }
.mark-cell { text-align: center; }
.mark { display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; border-radius: 50%; }
.mark-pass { color: var(--good); background: var(--good-wash); }
.mark-fail { color: var(--bad); background: var(--bad-wash); }
.mark-warn { color: var(--warn); background: var(--warn-wash); }
.mark-skip { color: var(--muted); background: var(--surface-2); }
.prelaunch-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  font-size: 13px;
  color: var(--ink-2);
  margin-bottom: 6px;
}
.prelaunch-head b { font-size: 22px; color: var(--ink); }
.reasons { margin: 10px 0 0; padding-left: 18px; font-size: 13px; color: var(--ink-2); }
.cost { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px 16px; margin: 16px 0 0; }
.cost dd { font-weight: 600; font-variant-numeric: tabular-nums; }
.error {
  margin-top: 12px;
  padding: 8px 10px;
  border-radius: 8px;
  background: var(--bad-wash);
  font-family: var(--mono);
  font-size: 12.5px;
  overflow-wrap: anywhere;
}
.charts { display: grid; gap: 24px; margin-top: 8px; }
@media (min-width: 860px) { .charts { grid-template-columns: 1fr 1fr; } }
.chart { padding: 4px 18px 18px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); }
.chart h3 { margin-top: 16px; }
.bar-row {
  display: grid;
  grid-template-columns: minmax(90px, 38%) minmax(0, 1fr) auto;
  align-items: center;
  gap: 12px;
  padding: 5px 0;
  font-size: 13.5px;
}
.bar-row:hover { background: var(--surface-2); }
.bar-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--ink-2); }
.bar-track { height: 10px; }
.bar-fill { display: block; width: 0; height: 100%; border-radius: 0 4px 4px 0; background: var(--accent); }
.bar-value { font-variant-numeric: tabular-nums; font-weight: 600; text-align: right; }
.footer { margin-top: 56px; padding-top: 24px; border-top: 1px solid var(--line); display: grid; gap: 12px; font-size: 14px; }
.downloads { display: flex; flex-wrap: wrap; gap: 8px 20px; }
.downloads code { color: var(--muted); }
@media print {
  body { background: #fff; font-size: 11pt; }
  .page { max-width: none; padding: 0; }
  .toc, .filters { display: none; }
  .variant[hidden] { display: block; }
  .section { padding-top: 24px; }
  .tile, .variant, .blocked, .chart, .table-wrap, tr { break-inside: avoid; }
  .frame { height: 120px; }
  .ar-9x16 { width: 68px; } .ar-4x5 { width: 96px; } .ar-1x1 { width: 120px; }
  a { color: inherit; text-decoration: none; }
}
"""
)

SCRIPT = """
(() => {
  const buttons = document.querySelectorAll("[data-filter]");
  const cards = document.querySelectorAll(".variant[data-status]");
  const select = (filter) => {
    cards.forEach((card) => { card.hidden = filter !== "all" && card.dataset.status !== filter; });
    buttons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.filter === filter)));
  };
  buttons.forEach((button) => button.addEventListener("click", () => select(button.dataset.filter)));
  document.addEventListener("play", (event) => {
    document.querySelectorAll("video").forEach((video) => { if (video !== event.target) video.pause(); });
  }, true);
})();
"""
