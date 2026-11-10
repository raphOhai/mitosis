from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from mitosis.cache import Cache
from mitosis.config import Settings, load_brief
from mitosis.events import EventBus
from mitosis.ledger import Ledger
from mitosis.models import Event, RunManifest
from mitosis.registry import Registry

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Mitosis: multiply a winning ad into identity-locked, policy-safe variations.")
console = Console()

STATUS_STYLE = {"passed": "green", "flagged": "yellow", "failed": "red"}


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
    )


def printer(event: Event) -> None:
    if event.type == "stage" and event.status == "start":
        detail = f" [dim]{event.message}[/dim]" if event.message else ""
        console.print(f"[bold cyan]>[/bold cyan] {event.stage}{detail}")
    elif event.type == "stage" and event.status == "done":
        console.print(f"  [dim]{event.stage} done in {event.data.get('seconds', 0):.1f}s[/dim]")
    elif event.type == "log" and event.stage:
        console.print(f"  [dim]{event.message}[/dim]")
    elif event.type == "variant" and event.status in STATUS_STYLE:
        style = STATUS_STYLE[event.status]
        score = event.data.get("score")
        score_text = f" score {score:.0f}" if isinstance(score, (int, float)) else ""
        console.print(
            f"  [{style}]{event.status:<7}[/{style}] {event.variant}  "
            f"${event.data.get('shadow_usd', 0):.2f} prod-eq  {event.data.get('seconds', 0):.0f}s  "
            f"try {event.data.get('attempts', 1)}{score_text}  [dim]{event.message}[/dim]"
        )
        if event.data.get("error"):
            console.print(f"          [red]{event.data['error']}[/red]")
    elif event.type == "error":
        console.print(f"[red]x {event.stage}: {event.message}[/red]")


def summary_table(manifest: RunManifest) -> Table:
    table = Table(title=f"{manifest.run_id}  ({manifest.mode})", show_lines=False)
    for column in ("Variant", "Hook", "Archetype", "Voice", "Location", "Status", "Score", "Prod $", "Tries"):
        table.add_column(column)
    for result in manifest.results:
        spec = result.spec
        style = STATUS_STYLE.get(result.status, "white")
        table.add_row(
            spec.id,
            spec.hook.text[:48],
            spec.hook.archetype,
            spec.voice.id,
            spec.location.id,
            f"[{style}]{result.status}[/{style}]",
            f"{result.prelaunch.score:.0f}" if result.prelaunch else "-",
            f"{result.shadow_usd:.2f}",
            str(result.attempts),
        )
    return table


def totals_line(manifest: RunManifest) -> str:
    t = manifest.totals
    rate = t.passed / t.variants * 100 if t.variants else 0
    return (
        f"{t.variants} variants, {t.renders} renders | passed {t.passed}, flagged {t.flagged}, failed {t.failed} ({rate:.0f}% pass) | "
        f"spend ${t.usd:.2f} actual, ${t.shadow_usd:.2f} at production prices (${t.shadow_per_variant:.2f}/variant) | "
        f"{t.seconds:.0f}s wall, {t.seconds_per_variant:.1f}s/variant | cache hits {t.cache_hits}, saved ${t.saved_usd:.2f}"
    )


@app.command()
def run(
    brief_path: Path = typer.Argument(..., exists=True, help="Brief YAML describing the winner and the batch."),
    offline: Optional[bool] = typer.Option(None, "--offline/--live", help="Force local stand-in models or live APIs."),
    variants: Optional[int] = typer.Option(None, help="Override the number of variants."),
    workers: Optional[int] = typer.Option(None, help="Parallel variant workers."),
    open_report: bool = typer.Option(False, "--open", help="Open the HTML report when finished."),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    configure_logging(verbose)
    from mitosis.pipeline import Pipeline

    settings = Settings.load(workers=workers)
    brief = load_brief(brief_path)
    if offline is not None:
        brief.offline = offline
    if variants:
        brief.variants = variants
    bus = EventBus()
    bus.subscribe(printer)
    manifest = Pipeline(settings, bus).run(brief)
    console.print(summary_table(manifest))
    console.print(totals_line(manifest))
    console.print(f"report  {manifest.report}")
    console.print(f"bundle  {manifest.exports.get('bundle')}")
    if open_report and manifest.report:
        webbrowser.open(Path(manifest.report).as_uri())


@app.command()
def plan(
    brief_path: Path = typer.Argument(..., exists=True),
    offline: Optional[bool] = typer.Option(None, "--offline/--live"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    configure_logging(verbose)
    from mitosis.pipeline import estimate

    brief = load_brief(brief_path)
    if offline is not None:
        brief.offline = offline
    bus = EventBus()
    bus.subscribe(printer)
    result = estimate(brief, Settings.load(), bus)
    data = result["plan"]
    table = Table(title=f"Plan: {data['design']}")
    for column in ("Variant", "Hook", "Voice", "Location", "New shot", "Changed"):
        table.add_column(column)
    for spec in data["variants"]:
        table.add_row(
            spec["id"],
            spec["hook"]["text"][:56],
            spec["voice"]["id"],
            spec["location"]["id"],
            "yes" if spec["new_shot"] else "reuse",
            ", ".join(spec["changed"]),
        )
    console.print(table)
    estimate_data = data["estimate"]
    console.print(f"estimated ${estimate_data['usd']:.2f} at production prices, ${estimate_data['per_variant_usd']:.2f} per variant")
    for note in data.get("notes", []):
        console.print(f"[dim]- {note}[/dim]")
    for hook in result["rejected"]:
        rules = ", ".join(f["rule"] for f in hook.get("flags", []))
        console.print(f"[yellow]blocked[/yellow] {hook['text']}  [dim]{rules}[/dim]")


@app.command()
def sample(
    out: Path = typer.Option(Path("samples"), help="Where to write the sample winner."),
    voice: str = typer.Option("Reed (English (US))", help="macOS voice for the sample narration."),
) -> None:
    from mitosis.sample import build_sample

    paths = build_sample(out.resolve(), voice)
    for label, path in paths.items():
        console.print(f"{label:<8} {path}")


@app.command()
def doctor() -> None:
    settings = Settings.load()
    table = Table(title="Mitosis doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    ffmpeg = shutil.which("ffmpeg")
    table.add_row("ffmpeg", "[green]ok[/green]" if ffmpeg else "[red]missing[/red]", ffmpeg or "brew install ffmpeg")
    if ffmpeg:
        filters = subprocess.run([ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True).stdout
        for name in ("ass", "loudnorm", "rubberband", "zoompan", "scdet", "freezedetect", "sidechaincompress"):
            present = f" {name} " in filters
            table.add_row(f"ffmpeg filter {name}", "[green]ok[/green]" if present else "[yellow]missing[/yellow]", "")
    table.add_row("say (offline TTS)", "[green]ok[/green]" if shutil.which("say") else "[yellow]missing[/yellow]", "macOS speech synthesis")
    for key in ("ANTHROPIC_API_KEY", "FAL_KEY", "ELEVENLABS_API_KEY", "RUNWAYML_API_SECRET", "HIGGSFIELD_API_KEY", "COMFYUI_URL"):
        table.add_row(key, "[green]set[/green]" if os.environ.get(key) else "[dim]unset[/dim]", "")
    registry = Registry(settings, Ledger(), Cache(settings.cache_dir))
    for kind, names in registry.describe().items():
        table.add_row(f"chain {kind}", "[green]live[/green]" if names and not registry.chain(kind).offline else "[cyan]offline[/cyan]" if names else "[dim]none[/dim]", " -> ".join(names))
    table.add_row("mode", registry.mode, "")
    console.print(table)


@app.command()
def models() -> None:
    settings = Settings.load()
    table = Table(title="Model catalog")
    for column in ("Kind", "Order", "Model", "Provider", "Endpoint", "Price"):
        table.add_column(column)
    for kind, names in settings.catalog.chains.items():
        for order, name in enumerate(names, 1):
            spec = settings.catalog.models.get(name)
            if not spec:
                continue
            price = ", ".join(f"${v}/{k}" for k, v in spec.price.items()) or ("local" if spec.offline else "-")
            table.add_row(kind, str(order), name, spec.provider, spec.model or "-", price)
    console.print(table)


@app.command()
def learn(results_csv: Path = typer.Argument(..., exists=True, help="Ad-level results exported from Meta.")) -> None:
    from mitosis.intelligence import priors as priorlib

    settings = Settings.load()
    updated = priorlib.learn(priorlib.load(settings.priors_path), results_csv)
    priorlib.save(updated, settings.priors_path)
    for factor, arms in updated.factors.items():
        ranked = sorted(arms.items(), key=lambda kv: -kv[1].mean)
        console.print(f"[bold]{factor}[/bold]  " + "  ".join(f"{name} {arm.mean:.2f} ({arm.wins}/{arm.trials})" for name, arm in ranked))


@app.command()
def lint(lines: list[str] = typer.Argument(..., help="Ad copy to check against the Meta policy rules.")) -> None:
    from mitosis.intelligence import policy

    brand = Settings.load().brand
    for text in lines:
        flags = policy.lint(text, brand)
        if not flags:
            console.print(f"[green]clean[/green]    {text}")
            continue
        verdict = "[red]blocked[/red]  " if policy.blocking(flags) else "[yellow]warning[/yellow]  "
        console.print(f"{verdict}{text}")
        for flag in flags:
            console.print(f"           [dim]{flag.rule} ({flag.severity}): {flag.message}[/dim]")
        fixed = policy.repair(text, flags, brand)
        console.print(f"           [cyan]repair[/cyan] {fixed}" if fixed else "           [red]no compliant rewrite, hook dropped[/red]")


@app.command()
def report(run_dir: Path = typer.Argument(..., exists=True, file_okay=False)) -> None:
    from mitosis.pipeline import load_manifest, save_manifest
    from mitosis.report import html

    manifest = load_manifest(run_dir / "manifest.json")
    manifest.report = html.write_report(manifest, run_dir / "report.html")
    save_manifest(manifest)
    console.print(str(manifest.report))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8765),
    offline: Optional[bool] = typer.Option(None, "--offline/--live"),
) -> None:
    import uvicorn

    from mitosis.service.api import create_app

    settings = Settings.load(offline=offline)
    console.print(f"Mitosis panel on http://{host}:{port}")
    uvicorn.run(create_app(settings), host=host, port=port, log_level="warning")


@app.command()
def inspect(run_dir: Path = typer.Argument(..., exists=True, file_okay=False)) -> None:
    from mitosis.pipeline import load_manifest

    manifest = load_manifest(run_dir / "manifest.json")
    console.print(summary_table(manifest))
    console.print(totals_line(manifest))
    console.print_json(json.dumps(manifest.totals.by_stage))


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
