from __future__ import annotations

import csv
import json
import logging
import re
import textwrap
import unicodedata
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from mitosis.config import Brand
from mitosis.models import Aspect, Render, RunManifest, VariantResult, VariantSpec, Word
from mitosis.report import html
from mitosis.report.selection import (
    is_launchable,
    is_reviewable,
    ordered_renders,
    ordered_results,
)

log = logging.getLogger(__name__)

DEFAULT_NAMING = "{brand}_{winner}_{variant}_{hook}_{voice}_{location}_{aspect}"
MAX_NAME_LENGTH = 120
SHRINK_ORDER = ("location", "voice", "hook", "winner", "brand", "variant")
UNSAFE_TOKEN = re.compile(r"[^A-Za-z0-9-]+")
UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")
REPEATED_DASHES = re.compile(r"-{2,}")

CALL_TO_ACTION = "SIGN_UP"
AD_STATUS = "PAUSED"
PLACEMENT_HINTS = {
    Aspect.VERTICAL: "Stories/Reels",
    Aspect.PORTRAIT: "Feed",
    Aspect.SQUARE: "Feed/Marketplace",
}
META_COLUMNS = (
    "Campaign Name",
    "Ad Set Name",
    "Ad Name",
    "Ad Status",
    "Title",
    "Body",
    "Call to Action",
    "Video File Name",
    "URL Tags",
    "placement_hint",
    "variant_id",
    "hook_id",
    "archetype",
    "voice",
    "location",
    "aspect",
    "qa_status",
    "review",
    "prelaunch_score",
    "utm_content",
    "local_path",
)

SENTENCE_END = (".", "!", "?", "…")
MIN_PHRASE_SECONDS = 0.2
FALLBACK_COMP_SECONDS = 1.0
PRECOMPRESSED_SUFFIXES = {".mp4", ".mov", ".jpg", ".jpeg", ".png", ".zip"}


def asset_name(brand: Brand, winner_id: str, spec: VariantSpec, aspect: Aspect) -> str:
    template = naming_template(brand)
    tokens = {
        "brand": sanitize_token(brand.name),
        "winner": sanitize_token(winner_id),
        "variant": sanitize_token(spec.id),
        "hook": f"{sanitize_token(spec.hook.id)}-{sanitize_token(spec.hook.archetype)}",
        "voice": sanitize_token(spec.voice.id),
        "location": sanitize_token(spec.location.id),
        "aspect": aspect.slug,
    }
    return fit_name(template, tokens, MAX_NAME_LENGTH)


def sanitize_token(value: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return REPEATED_DASHES.sub("-", UNSAFE_TOKEN.sub("-", ascii_text)).strip("-") or "na"


def naming_template(brand: Brand) -> str:
    return validated_template(brand.naming or DEFAULT_NAMING)


@lru_cache(maxsize=16)
def validated_template(template: str) -> str:
    try:
        template.format_map(dict.fromkeys(SHRINK_ORDER + ("aspect",), ""))
    except (KeyError, ValueError, IndexError):
        usable = False
    else:
        usable = all(field in template for field in ("{variant}", "{aspect}"))
    if not usable:
        log.warning("brand naming template %r is unusable, falling back to %r", template, DEFAULT_NAMING)
        return DEFAULT_NAMING
    return template


def render_name(template: str, tokens: dict[str, str]) -> str:
    return UNSAFE_NAME.sub("-", template.format_map(tokens))


def fit_name(template: str, tokens: dict[str, str], limit: int) -> str:
    tokens = dict(tokens)
    name = render_name(template, tokens)
    while len(name) > limit:
        candidates = [key for key in SHRINK_ORDER if len(tokens[key]) > 1]
        if not candidates:
            return name[:limit]
        longest = max(candidates, key=lambda key: len(tokens[key]))
        tokens[longest] = tokens[longest][:-1].rstrip("-") or tokens[longest][0]
        name = render_name(template, tokens)
    return name


def ad_name(render: Render, brand: Brand, winner_id: str, spec: VariantSpec) -> str:
    return render.name or asset_name(brand, winner_id, spec, render.aspect)


def export_meta_csv(manifest: RunManifest, brand: Brand, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(meta_rows(manifest, brand))
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=META_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    log.info("meta bulk sheet: %d ads from %d variants -> %s", len(rows), len(manifest.results), out)
    return out


def meta_rows(manifest: RunManifest, brand: Brand) -> Iterator[dict[str, str]]:
    winner_id = manifest.winner.id
    campaign = f"{brand.name} | Mitosis | {winner_id}"
    for result in ordered_results(manifest.results):
        if not is_launchable(result):
            continue
        for render in ordered_renders(result):
            yield meta_row(manifest, brand, campaign, result, render)


def meta_row(manifest: RunManifest, brand: Brand, campaign: str, result: VariantResult, render: Render) -> dict[str, str]:
    spec = result.spec
    name = ad_name(render, brand, manifest.winner.id, spec)
    placement = PLACEMENT_HINTS[render.aspect]
    return {
        "Campaign Name": campaign,
        "Ad Set Name": f"{manifest.winner.id} | {manifest.run_id} | {placement}",
        "Ad Name": name,
        "Ad Status": AD_STATUS,
        "Title": spec.hook.on_screen_text.strip(),
        "Body": primary_text(spec.hook.text, brand.cta),
        "Call to Action": CALL_TO_ACTION,
        "Video File Name": Path(render.path).name,
        "URL Tags": url_tags(manifest.winner.id, name),
        "placement_hint": placement,
        "variant_id": spec.id,
        "hook_id": spec.hook.id,
        "archetype": spec.hook.archetype,
        "voice": spec.voice.id,
        "location": spec.location.id,
        "aspect": render.aspect.value,
        "qa_status": result.status,
        "review": result.review,
        "prelaunch_score": f"{result.prelaunch.score:.3f}" if result.prelaunch else "",
        "utm_content": name,
        "local_path": str(Path(render.path).resolve()),
    }


def primary_text(hook_text: str, cta: str) -> str:
    hook = " ".join(hook_text.split())
    if hook and not hook.endswith(SENTENCE_END):
        hook = f"{hook}."
    return f"{hook} {cta.strip()}".strip()


def url_tags(winner_id: str, name: str) -> str:
    campaign = f"mitosis-{sanitize_token(winner_id)}"
    return f"utm_source=meta&utm_medium=paid_social&utm_campaign={campaign}&utm_content={name}"


@dataclass(frozen=True)
class Phrase:
    text: str
    start: float
    end: float


def group_phrases(words: list[Word], max_words: int, max_chars: int, max_gap: float, boundary: float) -> list[Phrase]:
    phrases: list[Phrase] = []
    current: list[Word] = []
    for word in (w for w in words if w.text.strip()):
        if current and starts_new_phrase(current, word, max_words, max_chars, max_gap, boundary):
            phrases.append(join_phrase(current))
            current = []
        current.append(word)
    if current:
        phrases.append(join_phrase(current))
    return phrases


def starts_new_phrase(current: list[Word], word: Word, max_words: int, max_chars: int, max_gap: float, boundary: float) -> bool:
    previous = current[-1]
    length = sum(len(w.text.strip()) + 1 for w in current) + len(word.text.strip())
    return (
        len(current) >= max(1, max_words)
        or length > max_chars
        or word.start - previous.end > max_gap
        or previous.text.strip().endswith(SENTENCE_END)
        or previous.start < boundary <= word.start
    )


def join_phrase(words: list[Word]) -> Phrase:
    return Phrase(text=" ".join(w.text.strip() for w in words), start=words[0].start, end=words[-1].end)


def hex_to_rgb(color: str) -> list[float]:
    digits = color.strip().lstrip("#")
    if len(digits) == 3:
        digits = "".join(ch * 2 for ch in digits)
    try:
        return [round(int(digits[i : i + 2], 16) / 255, 4) for i in (0, 2, 4)]
    except ValueError:
        log.warning("invalid color %r, using white", color)
        return [1.0, 1.0, 1.0]


def postscript_font(font: str) -> str:
    return "-".join(font.split())


def comp_duration(result: VariantResult, render: Render, fps: float) -> float:
    candidates = [render.duration, max((w.end for w in result.words), default=0.0), result.hook_seconds]
    return round(max(max(candidates) or FALLBACK_COMP_SECONDS, 1.0 / fps), 3)


def caption_entries(result: VariantResult, brand: Brand, name: str, duration: float) -> list[dict[str, object]]:
    style = brand.subtitle
    phrases = group_phrases(result.words, style.max_words, style.max_chars, style.max_gap, result.hook_seconds)
    entries: list[dict[str, object]] = []
    for phrase in phrases:
        start = round(max(0.0, phrase.start), 3)
        end = round(min(duration, max(phrase.end, start + MIN_PHRASE_SECONDS)), 3)
        if end <= start:
            continue
        entries.append(
            {
                "name": f"{name} caption {len(entries) + 1:02d}",
                "text": phrase.text.upper() if style.uppercase else phrase.text,
                "start": start,
                "end": end,
            }
        )
    return entries


def headline_entry(result: VariantResult, brand: Brand, name: str, duration: float) -> dict[str, object] | None:
    text = result.spec.hook.on_screen_text.strip()
    end = round(min(duration, result.hook_seconds), 3)
    if not text or end <= 0:
        return None
    lines = textwrap.wrap(text, width=max(8, brand.headline.max_chars_per_line)) or [text]
    return {"name": f"{name} headline", "text": "\r".join(lines), "start": 0.0, "end": end}


def comp_entry(manifest: RunManifest, brand: Brand, result: VariantResult, render: Render) -> dict[str, object]:
    width, height = render.aspect.size
    fps = float(brand.video.fps)
    name = ad_name(render, brand, manifest.winner.id, result.spec)
    duration = comp_duration(result, render, fps)
    caption_size = round(width * brand.subtitle.size_ratio, 1)
    headline_size = round(width * brand.headline.size_ratio, 1)
    hook_seconds = round(min(result.hook_seconds, duration), 3)
    return {
        "name": name,
        "group": "Ready" if is_launchable(result) else "Needs review",
        "footage": str(Path(render.path).resolve()),
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "hookSeconds": hook_seconds,
        "captionSize": caption_size,
        "captionStroke": round(caption_size * brand.subtitle.outline_ratio, 1),
        "captionPosition": [width / 2, round(height * brand.subtitle.anchor.get(render.aspect.value, 0.7), 1)],
        "headlineSize": headline_size,
        "headlineStroke": round(headline_size * 0.12, 1),
        "headlinePosition": [width / 2, round(height * brand.headline.anchor.get(render.aspect.value, 0.15), 1)],
        "headline": headline_entry(result, brand, name, duration),
        "captions": caption_entries(result, brand, name, duration),
    }


def jsx_payload(manifest: RunManifest, brand: Brand) -> dict[str, object]:
    comps = [
        comp_entry(manifest, brand, result, render)
        for result in ordered_results(manifest.results)
        if is_reviewable(result)
        for render in ordered_renders(result)
    ]
    return {
        "runId": manifest.run_id,
        "projectFolder": f"Mitosis {manifest.run_id}",
        "caption": {
            "font": postscript_font(brand.subtitle.font),
            "fill": hex_to_rgb(brand.subtitle.primary),
            "stroke": hex_to_rgb(brand.subtitle.outline),
        },
        "headline": {
            "font": postscript_font(brand.headline.font),
            "fill": hex_to_rgb(brand.headline.color),
            "stroke": hex_to_rgb(brand.headline.box),
        },
        "comps": comps,
    }


def js_literal(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, indent=2)
    escaped = encoded.replace("/", "\\/").replace("<", "\\u003c").replace(">", "\\u003e")
    return escaped.replace("\n", "\n    ")


JSX_RUNTIME = r"""
    var groups = {};

    function addFolder(name, parent) {
        var folder = app.project.items.addFolder(name);
        if (parent) {
            folder.parentFolder = parent;
        }
        return folder;
    }

    function groupFolder(root, name) {
        if (!groups[name]) {
            groups[name] = addFolder(name, root);
        }
        return groups[name];
    }

    function importFootage(entry, folder) {
        var file = new File(entry.footage);
        if (!file.exists) {
            return null;
        }
        var options = new ImportOptions(file);
        if (options.canImportAs(ImportAsType.FOOTAGE)) {
            options.importAs = ImportAsType.FOOTAGE;
        }
        var footage = app.project.importFile(options);
        footage.name = entry.name;
        footage.parentFolder = folder;
        return footage;
    }

    function applyFont(textDocument, font) {
        try {
            textDocument.font = font;
        } catch (fontError) {
            $.writeln("Mitosis: font " + font + " is unavailable, keeping the default font");
        }
    }

    function styleText(layer, look, size, stroke, position) {
        var sourceText = layer.property("ADBE Text Properties").property("ADBE Text Document");
        var textDocument = sourceText.value;
        applyFont(textDocument, look.font);
        textDocument.fontSize = size;
        textDocument.applyFill = true;
        textDocument.fillColor = look.fill;
        textDocument.applyStroke = stroke > 0;
        if (stroke > 0) {
            textDocument.strokeColor = look.stroke;
            textDocument.strokeWidth = stroke;
            textDocument.strokeOverFill = false;
        }
        textDocument.justification = ParagraphJustification.CENTER_JUSTIFY;
        sourceText.setValue(textDocument);
        layer.property("ADBE Transform Group").property("ADBE Position").setValue(position);
    }

    function addTextLayer(comp, item, look, size, stroke, position) {
        var layer = comp.layers.addText(item.text);
        layer.name = item.name;
        styleText(layer, look, size, stroke, position);
        layer.inPoint = item.start;
        layer.outPoint = item.end;
        return layer;
    }

    function buildComp(entry, root, footageFolder, missing) {
        var comp = app.project.items.addComp(entry.name, entry.width, entry.height, 1, entry.duration, entry.fps);
        comp.parentFolder = groupFolder(root, entry.group);
        var footage = importFootage(entry, footageFolder);
        if (footage) {
            comp.layers.add(footage).name = entry.name;
        } else {
            missing.push(entry.footage);
        }
        for (var index = 0; index < entry.captions.length; index += 1) {
            addTextLayer(comp, entry.captions[index], data.caption, entry.captionSize, entry.captionStroke, entry.captionPosition);
        }
        if (entry.headline) {
            addTextLayer(comp, entry.headline, data.headline, entry.headlineSize, entry.headlineStroke, entry.headlinePosition);
        }
        if (entry.hookSeconds > 0 && comp.markerProperty) {
            comp.markerProperty.setValueAtTime(entry.hookSeconds, new MarkerValue("hook end"));
        }
        return comp;
    }

    function summary(built, missing) {
        var message = "Mitosis: built " + built + " comps in \"" + data.projectFolder + "\".";
        if (missing.length > 0) {
            message += "\nMissing footage (" + missing.length + "):\n" + missing.slice(0, 8).join("\n");
        }
        return message;
    }

    function run() {
        var root = addFolder(data.projectFolder, null);
        var footageFolder = addFolder("Footage", root);
        var missing = [];
        for (var index = 0; index < data.comps.length; index += 1) {
            buildComp(data.comps[index], root, footageFolder, missing);
        }
        alert(summary(data.comps.length, missing));
    }

    app.beginUndoGroup(data.projectFolder);
    try {
        run();
    } catch (error) {
        alert("Mitosis import stopped: " + error.toString() + (error.line ? " at line " + error.line : ""));
    } finally {
        app.endUndoGroup();
    }
"""


def export_ae_jsx(manifest: RunManifest, brand: Brand, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = jsx_payload(manifest, brand)
    script = "#target aftereffects\n(function () {\n    var data = " + js_literal(payload) + ";\n" + JSX_RUNTIME + "}());\n"
    out.write_text(script, encoding="utf-8")
    log.info("after effects script: %d comps -> %s", len(payload["comps"]), out)
    return out


def bundle(manifest: RunManifest, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(f"{out.name}.part")
    entries = collect_bundle_entries(manifest, out)
    report = html.render_report(manifest, html.MappedLinks({source: arcname for arcname, source in entries.items()}))
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.html", report)
        for arcname, source in entries.items():
            method = zipfile.ZIP_STORED if source.suffix.lower() in PRECOMPRESSED_SUFFIXES else zipfile.ZIP_DEFLATED
            archive.write(source, arcname, compress_type=method)
    partial.replace(out)
    log.info("bundle: %d files and a relinked report -> %s", len(entries), out)
    return out


def collect_bundle_entries(manifest: RunManifest, out: Path) -> dict[str, Path]:
    excluded = out.resolve()
    entries: dict[str, Path] = {}
    for arcname, source in bundle_candidates(manifest):
        if source is None:
            continue
        source = Path(source)
        if not source.is_file():
            log.debug("bundle: skipping missing %s", source)
            continue
        if source.resolve() == excluded or arcname in entries:
            continue
        entries[arcname] = source
    return entries


def bundle_candidates(manifest: RunManifest) -> Iterator[tuple[str, Path | None]]:
    root = Path(manifest.out_dir)
    for result in ordered_results(manifest.results):
        if not is_reviewable(result):
            continue
        folder = "passed" if is_launchable(result) else "flagged"
        for render in ordered_renders(result):
            yield f"renders/{folder}/{Path(render.path).name}", Path(render.path)
            if render.thumbnail:
                yield f"thumbnails/{Path(render.thumbnail).name}", Path(render.thumbnail)
    for reference in manifest.hero.references:
        yield f"hero/{Path(reference).name}", Path(reference)
    yield "manifest.json", root / "manifest.json"
    yield "ledger.jsonl", manifest.ledger or root / "ledger.jsonl"
    for path in export_files(manifest):
        yield f"exports/{path.name}", path


def export_files(manifest: RunManifest) -> list[Path]:
    suffixes = {".csv", ".jsx"}
    declared = [Path(path) for path in manifest.exports.values() if Path(path).suffix.lower() in suffixes]
    folder = Path(manifest.out_dir) / "exports"
    discovered = sorted(p for p in folder.glob("*") if p.suffix.lower() in suffixes) if folder.is_dir() else []
    unique: dict[Path, Path] = {}
    for path in declared + discovered:
        unique.setdefault(path.resolve(), path)
    return list(unique.values())
