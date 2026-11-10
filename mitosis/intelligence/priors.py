from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from mitosis.intelligence.hooks import ARCHETYPES
from mitosis.intelligence.policy import win_rule as policy_win_rule
from mitosis.models import BetaArm, Priors

log = logging.getLogger(__name__)

FACTORS = ("archetype", "voice", "location")
VARIANT_TOKEN = re.compile(r"^v\d{1,4}$", re.IGNORECASE)
HOOK_TOKEN = re.compile(r"^h\d{1,4}(?:-(?P<archetype>.+))?$", re.IGNORECASE)
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "ad_name": ("ad_name", "ad", "name", "creative_name"),
    "spend": ("spend", "amount_spent", "amount_spent_usd"),
    "impressions": ("impressions", "impr"),
    "purchases": ("purchases", "website_purchases", "results", "conversions"),
    "cpa": ("cpa", "cost_per_purchase", "cost_per_result", "cost_per_website_purchase"),
    "roas": ("roas", "purchase_roas", "purchase_roas_return_on_ad_spend", "website_purchase_roas", "return_on_ad_spend"),
    "win": ("win", "winner", "is_win", "won"),
}
TRUE_WORDS = {"1", "true", "yes", "y", "win", "winner", "won"}
FALSE_WORDS = {"0", "false", "no", "n", "loss", "lose", "lost"}


@dataclass
class AdKey:
    variant: str
    archetype: str | None
    voice: str
    location: str


@dataclass
class Tally:
    key: AdKey
    spend: float = 0.0
    revenue: float = 0.0
    purchases: float = 0.0
    impressions: float = 0.0
    has_spend: bool = False
    has_revenue: bool = False
    has_purchases: bool = False
    cpa_values: list[float] = field(default_factory=list)
    roas_values: list[float] = field(default_factory=list)
    votes: list[bool] = field(default_factory=list)


def load(path: Path) -> Priors:
    location = Path(path)
    if not location.exists():
        return Priors()
    try:
        return Priors.model_validate_json(location.read_text())
    except (ValidationError, ValueError) as exc:
        log.warning("priors file %s is unreadable (%s), starting fresh", location, exc)
        return Priors()


def save(priors: Priors, path: Path) -> Path:
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text(priors.model_dump_json(indent=2))
    return location


def canonical(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", header.strip().lower()).strip("_")


def column_map(headers: list[str]) -> dict[str, str]:
    normalized = {canonical(h): h for h in headers if h}
    mapping: dict[str, str] = {}
    for field_name, aliases in COLUMN_ALIASES.items():
        match = next((alias for alias in aliases if alias in normalized), None)
        if match:
            mapping[field_name] = normalized[match]
    return mapping


def number(raw: str | None) -> float | None:
    if raw is None:
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", raw.replace(",", ""))
    if not cleaned or cleaned in {"-", ".", "-."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def flag(raw: str | None) -> bool | None:
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in TRUE_WORDS:
        return True
    if value in FALSE_WORDS:
        return False
    return None


def archetype_key(raw: str | None) -> str | None:
    if not raw:
        return None
    candidate = re.sub(r"[\s-]+", "_", raw.strip().lower())
    if candidate in ARCHETYPES:
        return candidate
    dashed = raw.strip().lower()
    return next((key for key in ARCHETYPES if key.replace("_", "-") == dashed), candidate)


def parse_ad_name(name: str) -> AdKey | None:
    parts = [part for part in name.strip().split("_") if part]
    for index in range(1, len(parts) - 2):
        match = HOOK_TOKEN.match(parts[index])
        if match and VARIANT_TOKEN.match(parts[index - 1]):
            return AdKey(
                variant="_".join(parts[:index]),
                archetype=archetype_key(match.group("archetype")),
                voice=parts[index + 1].replace("-", "_"),
                location=parts[index + 2].replace("-", "_"),
            )
    return None


def decide(tally: Tally, rule: dict[str, float]) -> bool | None:
    roas_target = rule.get("roas_target")
    cpa_target = rule.get("cpa_target")
    if tally.votes:
        return sum(tally.votes) * 2 >= len(tally.votes)
    min_spend = rule.get("min_spend", 0.0)
    if tally.has_spend and tally.spend < min_spend:
        return None
    if roas_target is not None and (tally.has_revenue or tally.roas_values):
        roas = tally.revenue / tally.spend if tally.has_revenue and tally.spend > 0 else sum(tally.roas_values) / len(tally.roas_values)
        return roas >= roas_target
    if cpa_target is not None and tally.has_spend and tally.has_purchases:
        return tally.purchases > 0 and tally.spend / tally.purchases <= cpa_target
    if cpa_target is not None and tally.cpa_values:
        valid = [v for v in tally.cpa_values if v > 0]
        return bool(valid) and sum(valid) / len(valid) <= cpa_target
    return None


def accumulate(tally: Tally, row: dict[str, str], columns: dict[str, str]) -> None:
    def value(field_name: str) -> str | None:
        column = columns.get(field_name)
        return row.get(column) if column else None

    vote = flag(value("win"))
    if vote is not None:
        tally.votes.append(vote)
    spend = number(value("spend"))
    roas = number(value("roas"))
    purchases = number(value("purchases"))
    cpa = number(value("cpa"))
    impressions = number(value("impressions"))
    if spend is not None:
        tally.spend += spend
        tally.has_spend = True
        if roas is not None:
            tally.revenue += roas * spend
            tally.has_revenue = True
    if roas is not None:
        tally.roas_values.append(roas)
    if purchases is not None:
        tally.purchases += purchases
        tally.has_purchases = True
    if cpa is not None:
        tally.cpa_values.append(cpa)
    if impressions is not None:
        tally.impressions += impressions


def read_tallies(results_csv: Path) -> dict[str, Tally]:
    tallies: dict[str, Tally] = {}
    with Path(results_csv).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = column_map(list(reader.fieldnames or []))
        if "ad_name" not in columns:
            log.warning("results file %s has no ad name column", results_csv)
            return tallies
        for row in reader:
            key = parse_ad_name(row.get(columns["ad_name"]) or "")
            if key is None:
                continue
            tally = tallies.setdefault(key.variant, Tally(key=key))
            accumulate(tally, row, columns)
    return tallies


def update(priors: Priors, factor: str, level: str, won: bool) -> None:
    arms = priors.factors.setdefault(factor, {})
    arm = arms.setdefault(level, BetaArm())
    arm.alpha += 1.0 if won else 0.0
    arm.beta += 0.0 if won else 1.0
    arm.trials += 1
    arm.wins += 1 if won else 0


def learn(priors: Priors, results_csv: Path, win_rule: dict[str, float] | None = None) -> Priors:
    rule = win_rule if win_rule is not None else policy_win_rule()
    learned = priors.model_copy(deep=True)
    tallies = read_tallies(results_csv)
    decided = 0
    for tally in tallies.values():
        won = decide(tally, rule)
        if won is None:
            continue
        decided += 1
        levels = {"archetype": tally.key.archetype, "voice": tally.key.voice, "location": tally.key.location}
        for factor in FACTORS:
            if levels[factor]:
                update(learned, factor, levels[factor], won)
    learned.updated_at = datetime.now(UTC).isoformat(timespec="seconds")
    log.info("learned from %d of %d variants in %s", decided, len(tallies), results_csv)
    return learned
