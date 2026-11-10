from __future__ import annotations

import math
import random

from mitosis.models import BetaArm, Priors


def sample(rng: random.Random, arm: BetaArm) -> float:
    return rng.betavariate(max(arm.alpha, 1e-3), max(arm.beta, 1e-3))


def draw(rng: random.Random, priors: Priors, factor: str, levels: list[str]) -> dict[str, float]:
    return {level: sample(rng, priors.arm(factor, level)) for level in levels}


def allocate(rng: random.Random, priors: Priors, factor: str, levels: list[str], total: int, cap_share: float = 0.5) -> dict[str, int]:
    counts = {level: 0 for level in levels}
    if total <= 0 or not levels:
        return counts
    if total >= len(levels):
        for level in levels:
            counts[level] = 1
    else:
        explore = max(1, math.ceil(total * 0.25))
        by_trials = sorted(levels, key=lambda level: (priors.arm(factor, level).trials, rng.random()))
        for level in by_trials[:explore]:
            counts[level] = 1
        samples = draw(rng, priors, factor, [level for level in levels if counts[level] == 0])
        for level in sorted(samples, key=lambda level: -samples[level])[: total - explore]:
            counts[level] = 1
        return counts
    cap = max(1, math.ceil(total * cap_share))
    for _ in range(total - len(levels)):
        samples = draw(rng, priors, factor, levels)
        open_levels = [level for level in levels if counts[level] < cap] or levels
        winner = max(open_levels, key=lambda level: samples[level])
        counts[winner] += 1
    return counts
