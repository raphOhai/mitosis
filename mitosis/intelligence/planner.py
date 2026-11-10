from __future__ import annotations

import logging
import math
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from mitosis.intelligence import bandit
from mitosis.intelligence.text import clip_words, normalize
from mitosis.models import Aspect, CostEstimate, CreativeDNA, Hook, Location, Priors, VariantPlan, VariantSpec, Voice

log = logging.getLogger(__name__)

SEED_STRIDE = 1009
SECONDS_PER_VARIANT = 75.0
MIN_HOOK_WEIGHT = 0.05
NEW_PAIR_WEIGHT = 10.0
BODY_MODES = ("keep", "sts", "tts")
DESIGN = "pairwise covering design, bandit-weighted hooks"
ORIGINAL_LOCATION = Location(
    id="original",
    label="Original set",
    prompt="the same room and lighting as the reference frames",
    grade="neutral",
    original=True,
)


@dataclass
class Slot:
    hook: Hook
    priority: float
    voice: Voice | None = None
    location: Location | None = None


def control_hook(dna: CreativeDNA) -> Hook:
    return Hook(
        id="H00",
        archetype=dna.hook_archetype,
        text=dna.hook_text,
        on_screen_text=clip_words(dna.hook_text, 6),
        visual="",
        motion="",
        rationale="Original winning hook used as the control because no new hooks passed the gates.",
        score=0.5,
    )


def hook_weights(hooks: list[Hook], priors: Priors, rng: random.Random) -> dict[str, float]:
    return {
        hook.id: max(hook.score, MIN_HOOK_WEIGHT) * bandit.sample(rng, priors.arm("archetype", hook.archetype)) for hook in hooks
    }


def hook_uses(hooks: list[Hook], weights: dict[str, float], count: int, rng: random.Random) -> dict[str, int]:
    ranked = sorted(hooks, key=lambda h: -weights[h.id])
    if count <= len(hooks):
        return {hook.id: 1 for hook in ranked[:count]}
    extra = count - len(hooks)
    total = sum(weights.values()) or 1.0
    quotas = {hook.id: extra * weights[hook.id] / total for hook in hooks}
    uses = {hook.id: 1 + math.floor(quotas[hook.id]) for hook in hooks}
    jitter = {hook.id: rng.random() for hook in hooks}
    remainder = count - sum(uses.values())
    for hook in sorted(hooks, key=lambda h: (-(quotas[h.id] - math.floor(quotas[h.id])), jitter[h.id]))[:remainder]:
        uses[hook.id] += 1
    return uses


def schedule(hooks: list[Hook], weights: dict[str, float], uses: dict[str, int]) -> list[Slot]:
    slots = [
        Slot(hook=hook, priority=weights[hook.id] / (1 + occurrence))
        for hook in hooks
        for occurrence in range(uses.get(hook.id, 0))
    ]
    order = {hook.id: index for index, hook in enumerate(hooks)}
    return sorted(slots, key=lambda s: (-s.priority, order[s.hook.id]))


def assign(slots: list[Slot], voices: list[Voice], locations: list[Location], rng: random.Random) -> None:
    hook_voice: set[tuple[str, str]] = set()
    hook_location: set[tuple[str, str]] = set()
    voice_location: set[tuple[str, str]] = set()
    voice_uses: Counter[str] = Counter()
    location_uses: Counter[str] = Counter()
    voice_cap = math.ceil(len(slots) / len(voices))
    location_cap = math.ceil(len(slots) / len(locations))
    pairs = [(voice, location) for voice in voices for location in locations]
    for slot in slots:
        hook_id = slot.hook.id
        feasible = [(v, l) for v, l in pairs if voice_uses[v.id] < voice_cap and location_uses[l.id] < location_cap] or pairs
        values = [
            NEW_PAIR_WEIGHT
            * (
                ((hook_id, v.id) not in hook_voice)
                + ((hook_id, l.id) not in hook_location)
                + ((v.id, l.id) not in voice_location)
            )
            - voice_uses[v.id]
            - location_uses[l.id]
            + 0.01 * rng.random()
            for v, l in feasible
        ]
        voice, location = feasible[max(range(len(feasible)), key=values.__getitem__)]
        slot.voice, slot.location = voice, location
        hook_voice.add((hook_id, voice.id))
        hook_location.add((hook_id, location.id))
        voice_location.add((voice.id, location.id))
        voice_uses[voice.id] += 1
        location_uses[location.id] += 1


def changed_factors(spec_hook: Hook, voice: Voice, location: Location, dna: CreativeDNA, control_voice: Voice, control_location: Location) -> list[str]:
    changed: list[str] = []
    if normalize(spec_hook.text).lower() != normalize(dna.hook_text).lower():
        changed.append("hook")
    if voice.id != control_voice.id:
        changed.append("voice")
    if location.id != control_location.id:
        changed.append("location")
    return changed


def build_spec(
    index: int,
    slot: Slot,
    aspects: list[Aspect],
    seed: int,
    body_mode: str,
    dna: CreativeDNA,
    control_voice: Voice,
    control_location: Location,
) -> VariantSpec:
    return VariantSpec(
        id=f"V{index:02d}",
        index=index,
        hook=slot.hook,
        voice=slot.voice,
        location=slot.location,
        aspects=list(aspects),
        seed=seed + SEED_STRIDE * index,
        new_shot=not slot.location.original,
        changed=changed_factors(slot.hook, slot.voice, slot.location, dna, control_voice, control_location),
        body_mode=body_mode,
    )


def relocate(spec: VariantSpec, location: Location, dna: CreativeDNA, control_voice: Voice, control_location: Location) -> VariantSpec:
    return spec.model_copy(
        update={
            "location": location,
            "new_shot": not location.original,
            "changed": changed_factors(spec.hook, spec.voice, location, dna, control_voice, control_location),
        }
    )


def fit_budget(
    specs: list[VariantSpec],
    budget_usd: float,
    cost: Callable[[VariantSpec], float],
    original: Location,
    dna: CreativeDNA,
    control_voice: Voice,
    control_location: Location,
    notes: list[str],
) -> tuple[list[VariantSpec], list[float]]:
    costs = [cost(spec) for spec in specs]
    total = sum(costs)
    if total <= budget_usd:
        return specs, costs
    notes.append(f"estimated ${total:.2f} exceeds the ${budget_usd:.2f} budget")
    for index in sorted(range(len(specs)), key=lambda i: (-costs[i], -i)):
        if total <= budget_usd:
            break
        spec = specs[index]
        if not spec.new_shot:
            continue
        cheaper = relocate(spec, original, dna, control_voice, control_location)
        price = cost(cheaper)
        if price >= costs[index]:
            continue
        notes.append(f"{spec.id}: reuses the original hook shot instead of '{spec.location.label}' (saves ${costs[index] - price:.2f})")
        total -= costs[index] - price
        specs[index], costs[index] = cheaper, price
    while total > budget_usd and len(specs) > 1:
        dropped, price = specs.pop(), costs.pop()
        total -= price
        notes.append(f"{dropped.id}: dropped as lowest priority ({dropped.hook.id} x {dropped.voice.id} x {dropped.location.id}, ${price:.2f})")
    if total > budget_usd:
        notes.append(f"budget is below the cost of a single variant; kept {specs[0].id} as the minimum run")
    return specs, costs


def ratio(covered: int, possible: int) -> float:
    return round(covered / possible, 4) if possible else 1.0


def balance(counts: Counter[str], levels: list[str], runs: int) -> float:
    considered = levels if runs >= len(levels) else [level for level in levels if counts[level]]
    values = [counts[level] for level in considered]
    return round(min(values) / max(values), 4) if values and max(values) else 1.0


def coverage(specs: list[VariantSpec], hooks: list[str], voices: list[str], locations: list[str]) -> dict[str, float]:
    hook_voice = {(s.hook.id, s.voice.id) for s in specs}
    hook_location = {(s.hook.id, s.location.id) for s in specs}
    voice_location = {(s.voice.id, s.location.id) for s in specs}
    covered = len(hook_voice) + len(hook_location) + len(voice_location)
    possible = len(hooks) * len(voices) + len(hooks) * len(locations) + len(voices) * len(locations)
    attainable = min(possible, 3 * len(specs)) if specs else 0
    runs = len(specs)
    return {
        "hook_voice": ratio(len(hook_voice), len(hooks) * len(voices)),
        "hook_location": ratio(len(hook_location), len(hooks) * len(locations)),
        "voice_location": ratio(len(voice_location), len(voices) * len(locations)),
        "pairwise": ratio(covered, possible),
        "pairwise_attainable": ratio(covered, attainable),
        "hook_balance": balance(Counter(s.hook.id for s in specs), hooks, runs),
        "voice_balance": balance(Counter(s.voice.id for s in specs), voices, runs),
        "location_balance": balance(Counter(s.location.id for s in specs), locations, runs),
        "new_shot_ratio": ratio(sum(1 for s in specs if s.new_shot), runs),
    }


def estimate(specs: list[VariantSpec], costs: list[float], cost: Callable[[VariantSpec], float], original: Location, budget_usd: float) -> CostEstimate:
    usd = round(sum(costs), 4)
    shots = 0.0
    for spec, price in zip(specs, costs, strict=True):
        if spec.new_shot:
            shots += max(0.0, price - cost(spec.model_copy(update={"location": original, "new_shot": False})))
    count = len(specs)
    return CostEstimate(
        usd=usd,
        seconds=SECONDS_PER_VARIANT * count,
        per_variant_usd=round(usd / count, 4) if count else 0.0,
        breakdown={
            "new_hook_shots": round(shots, 4),
            "voice_and_copy": round(usd - shots, 4),
            "budget_headroom": round(budget_usd - usd, 4),
        },
    )


def plan(
    winner_id: str,
    dna: CreativeDNA,
    hooks: list[Hook],
    voices: list[Voice],
    locations: list[Location],
    aspects: list[Aspect],
    count: int,
    budget_usd: float,
    priors: Priors,
    cost: Callable[[VariantSpec], float],
    body_mode: str = "sts",
    seed: int = 7,
) -> VariantPlan:
    rng = random.Random(seed)
    notes: list[str] = []
    mode = body_mode if body_mode in BODY_MODES else "sts"
    if count <= 0 or not voices:
        notes.append("nothing to plan: no variants requested" if count <= 0 else "nothing to plan: no voices available")
        return VariantPlan(winner_id=winner_id, design=DESIGN, variants=[], notes=notes)
    if not hooks:
        hooks = [control_hook(dna)]
        notes.append("no generated hooks available; every variant reuses the original hook")
    places = list(locations) or [ORIGINAL_LOCATION]
    original = next((location for location in places if location.original), ORIGINAL_LOCATION)
    control_voice, control_location = voices[0], original
    weights = hook_weights(hooks, priors, rng)
    uses = hook_uses(hooks, weights, count, rng)
    if count >= len(hooks):
        notes.append(f"all {len(hooks)} hooks used at least once; extra slots weighted by score x Thompson-sampled archetype prior")
    else:
        notes.append(f"{count} of {len(hooks)} hooks selected by score x Thompson-sampled archetype prior")
    slots = schedule(hooks, weights, uses)
    assign(slots, voices, places, rng)
    specs = [build_spec(i, slot, aspects, seed, mode, dna, control_voice, control_location) for i, slot in enumerate(slots, start=1)]
    specs, costs = fit_budget(specs, budget_usd, cost, original, dna, control_voice, control_location, notes)
    hook_ids = list(dict.fromkeys(h.id for h in hooks))
    voice_ids = [v.id for v in voices]
    location_ids = list(dict.fromkeys([place.id for place in places] + [spec.location.id for spec in specs]))
    return VariantPlan(
        winner_id=winner_id,
        design=f"{DESIGN} ({len(specs)} variants x {len(aspects)} placements)",
        variants=specs,
        levels={"hook": hook_ids, "voice": voice_ids, "location": location_ids, "aspect": [a.value for a in aspects]},
        coverage=coverage(specs, hook_ids, voice_ids, location_ids),
        estimate=estimate(specs, costs, cost, original, budget_usd),
        notes=notes,
    )
