from __future__ import annotations

import csv
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from mitosis.config import PACKAGE_ROOT, Brand, read_yaml
from mitosis.intelligence import bandit, deconstruct, hooks, planner, policy, predict, priors, structure
from mitosis.intelligence.text import jaccard, word_count
from mitosis.models import (
    Aspect,
    BetaArm,
    CreativeDNA,
    Hook,
    Location,
    MediaInfo,
    PolicyFlag,
    Priors,
    Shot,
    VariantSpec,
    Voice,
    Winner,
    Word,
)
from mitosis.providers.base import GenResult, ProviderError

CONFIG = PACKAGE_ROOT / "config"
SCRIPT = (
    "I'll be honest, my wife and I had become roommates. "
    "We only talked about the kids and the bills. "
    "Then I started a two-minute check-in every night, and she noticed by day three. "
    "Try Betterly free."
)


class FakeLLM:
    def __init__(self, responses: dict[str, Any] | None = None, fail: bool = False) -> None:
        self.responses = responses or {}
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def json(self, system: str, prompt: str, schema: dict, images: list[Path] | None = None, max_tokens: int = 8000) -> GenResult:
        self.calls.append({"system": system, "prompt": prompt, "schema": schema, "images": images or []})
        if self.fail:
            raise ProviderError("all models failed for 'llm'")
        keys = set(schema.get("properties", {}))
        for key, payload in self.responses.items():
            if key in keys:
                return GenResult(data=payload(prompt) if callable(payload) else payload)
        return GenResult(data={})


@pytest.fixture(scope="module")
def brand() -> Brand:
    data = read_yaml(CONFIG / "brand.yaml")
    data["policy"] = read_yaml(CONFIG / "policy.yaml")
    return Brand(**data)


@pytest.fixture(scope="module")
def voices() -> list[Voice]:
    return [Voice(id=key, **value) for key, value in read_yaml(CONFIG / "voices.yaml")["voices"].items()]


@pytest.fixture(scope="module")
def locations() -> list[Location]:
    return [Location(id=key, **value) for key, value in read_yaml(CONFIG / "locations.yaml")["locations"].items()]


def timed_words(text: str, start: float = 0.1, word_seconds: float = 0.28, gap: float = 0.06, sentence_gap: float = 0.35) -> list[Word]:
    words: list[Word] = []
    cursor = start
    for token in text.split():
        words.append(Word(text=token, start=round(cursor, 3), end=round(cursor + word_seconds, 3)))
        cursor += word_seconds + (sentence_gap if token[-1] in ".!?" else gap)
    return words


def make_winner(tmp_path: Path, script: str = SCRIPT, talking_head: bool = True) -> Winner:
    words = timed_words(script)
    duration = round(words[-1].end + 0.5, 3)
    spans = [(0.0, 2.4), (2.4, 7.0), (7.0, duration)]
    hook_end, cta_start = structure.segment(words, spans, duration)
    info = MediaInfo(path=tmp_path / "winner.mp4", width=1080, height=1920, fps=30.0, duration=duration, has_audio=True)
    shots = [Shot(index=i, start=s, end=e, face=None) for i, (s, e) in enumerate(spans)]
    return Winner(
        id="w01",
        video=tmp_path / "winner.mp4",
        script=script,
        info=info,
        audio=tmp_path / "audio.wav",
        words=words,
        shots=shots,
        hook_end=hook_end,
        cta_start=cta_start,
        talking_head=talking_head,
    )


def make_dna(**overrides: Any) -> CreativeDNA:
    values: dict[str, Any] = {
        "angle": "Reconnection: from roommates back to partners",
        "audience": "US men 28-45 in long-term relationships",
        "promise": "Feel like partners again",
        "hook_archetype": "confession",
        "hook_text": "I'll be honest, my wife and I had become roommates.",
        "body_text": "We only talked about the kids and the bills.",
        "cta_text": "Try Betterly free.",
        "hero_description": "the same man in his mid-30s seen in the winning ad, exactly as in the reference frames",
        "setting": "an everyday home interior",
        "tone": "honest, warm",
        "pacing_wps": 2.7,
        "format": "talking_head",
    }
    values.update(overrides)
    return CreativeDNA(**values)


def test_sentences_split_on_terminal_punctuation() -> None:
    words = timed_words("We drifted. Nobody noticed! Did she? Then this")
    groups = structure.sentences(words)
    assert [len(g) for g in groups] == [2, 2, 2, 2]
    assert structure.sentences([]) == []


def test_segment_ends_hook_at_first_sentence_past_minimum() -> None:
    words = timed_words("Okay. I had become a roommate in my own marriage. Then something changed for us. Try Betterly free.")
    hook_end, _ = structure.segment(words, [], words[-1].end + 0.5)
    first_long = structure.sentences(words)[1]
    assert hook_end == pytest.approx(first_long[-1].end)
    assert hook_end >= 1.2


def test_segment_snaps_to_cut_but_never_inside_a_word() -> None:
    words = timed_words("We stopped talking at dinner. Then I changed one thing. Try it.")
    sentence_end = structure.sentences(words)[0][-1].end
    next_start = structure.sentences(words)[1][0].start
    cut_in_gap = round((sentence_end + next_start) / 2, 3)
    snapped, _ = structure.segment(words, [(0.0, cut_in_gap), (cut_in_gap, 9.0)], 9.0)
    assert snapped == pytest.approx(cut_in_gap)
    cut_in_word = round(next_start + 0.1, 3)
    unsnapped, _ = structure.segment(words, [(0.0, cut_in_word), (cut_in_word, 9.0)], 9.0)
    assert unsnapped == pytest.approx(sentence_end)


def test_segment_cuts_long_first_sentence_near_three_seconds() -> None:
    words = timed_words("I never thought the thing that would bring us back together was a tiny question I ask every single night before we sleep.")
    hook_end, _ = structure.segment(words, [], words[-1].end + 1.0)
    assert abs(hook_end - 3.0) < 0.4
    assert any(w.end == pytest.approx(hook_end) for w in words)


def test_segment_cta_rules() -> None:
    words = timed_words(SCRIPT)
    duration = words[-1].end + 0.5
    hook_end, cta_start = structure.segment(words, [], duration)
    assert cta_start == pytest.approx(structure.sentences(words)[-1][0].start)
    assert cta_start >= hook_end
    early = timed_words("We drifted apart for years and I hated it. Try it.")
    hook_end, cta_start = structure.segment(early, [], 20.0)
    assert cta_start == pytest.approx(max(hook_end, 16.0))


def test_segment_handles_empty_input() -> None:
    assert structure.segment([], [], 0.0) == (0.0, 0.0)
    hook_end, cta_start = structure.segment([], [(0.0, 2.9), (2.9, 10.0)], 10.0)
    assert hook_end == pytest.approx(2.9)
    assert cta_start == pytest.approx(6.0)


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        ("Are you lonely in your marriage?", "personal_attributes"),
        ("Struggling with ED?", "personal_attributes"),
        ("Meet other bankrupt dads near you.", "personal_attributes"),
        ("Is your sex life boring?", "personal_attributes"),
        ("Want better sex tonight?", "sexual_enhancement"),
        ("The orgasm trick she will love.", "sexual_explicit"),
        ("This is guaranteed to fix your marriage.", "guaranteed_outcomes"),
        ("Tag your wife if this is you.", "engagement_bait"),
        ("Tap the play button now.", "fake_buttons"),
        ("Better than therapy for couples.", "medical_claims"),
        ("This shit actually works.", "profanity"),
        ("What the f*ck happened to us.", "masked_words"),
        ("Your wife is bored of you.", "negative_self_perception"),
    ],
)
def test_lint_blocks_violations(brand: Brand, text: str, rule: str) -> None:
    flags = policy.lint(text, brand)
    assert rule in {f.rule for f in flags}
    assert policy.blocking(flags)


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        ("Honestly THIS ONE HABIT CHANGED EVERYTHING for us at home", "excessive_caps"),
        ("We talk every night now!!!", "excessive_punctuation"),
        ("We talk every night now \U0001F60D", "emoji"),
        ("Our sex life came back.", "sexual_suggestive"),
    ],
)
def test_lint_warns_on_quality_issues(brand: Brand, text: str, rule: str) -> None:
    flags = policy.lint(text, brand)
    assert rule in {f.rule for f in flags}
    assert all(f.severity == "warn" for f in flags if f.rule == rule)


@pytest.mark.parametrize(
    "text",
    [
        "I felt lonely in my own marriage for two years.",
        "Most couples hit a lonely stretch.",
        "Date nights didn't fix us. A two-minute check-in did.",
        "POV: you put the phone down and she actually noticed.",
        "My wife said: I miss you, even when you're right here.",
        "Most guys over 40 never talk about this.",
        "Try Betterly free.",
        "Every couple deserves more intimacy and connection.",
    ],
)
def test_lint_lets_safe_copy_through(brand: Brand, text: str) -> None:
    assert policy.lint(text, brand) == []


def test_caps_rule_ignores_short_headlines(brand: Brand) -> None:
    assert policy.lint("DATE NIGHTS DIDN'T FIX US", brand) == []
    assert policy.lint("POV: PHONE DOWN, EYES UP", brand) == []
    shouty = "I NEVER EXPECTED THIS to work for us at home"
    assert [f.rule for f in policy.lint(shouty, brand)] == ["excessive_caps"]
    assert policy.repair(shouty, policy.lint(shouty, brand), brand) == "I never expected this to work for us at home"


def test_repair_reframes_personal_attribute(brand: Brand) -> None:
    text = "Are you lonely in your marriage?"
    fixed = policy.repair(text, policy.lint(text, brand), brand)
    assert fixed == "Most couples hit a lonely stretch."
    assert not policy.blocking(policy.lint(fixed, brand))


def test_repair_substitutes_and_drops(brand: Brand) -> None:
    fixed = policy.repair("Want better sex? Try Betterly!!!", policy.lint("Want better sex? Try Betterly!!!", brand), brand)
    assert fixed == "Want a closer connection? Try Betterly!"
    assert policy.repair("Tag your wife if this is you.", [], brand) is None
    assert policy.repair("Try Betterly free.", [], brand) == "Try Betterly free."


def test_repair_uses_llm_rewrite_when_compliant(brand: Brand) -> None:
    llm = FakeLLM({"changes": {"text": "I felt lonely next to her for years.", "changes": "first person"}})
    text = "Are you lonely next to her?"
    assert policy.repair(text, policy.lint(text, brand), brand, llm) == "I felt lonely next to her for years."
    stubborn = FakeLLM({"changes": {"text": "Are you still lonely?", "changes": "none"}})
    assert policy.repair(text, policy.lint(text, brand), brand, stubborn) == "Most couples hit a lonely stretch."


def test_repair_refuses_unknown_blocking_flag_without_llm(brand: Brand) -> None:
    flag = PolicyFlag(rule="other", severity="block", message="implied relationship status")
    assert policy.repair("Sound familiar?", [flag], brand) is None


def test_review_merges_llm_flags_and_dedupes_by_rule(brand: Brand) -> None:
    reviews = {
        "reviews": [
            {
                "index": 0,
                "flags": [
                    {"rule": "personal_attributes", "severity": "block", "span": "lonely", "message": "dup", "suggestion": ""},
                    {"rule": "other", "severity": "warn", "span": "marriage", "message": "status", "suggestion": ""},
                ],
            }
        ]
    }
    flags = policy.review("Are you lonely in your marriage?", brand, FakeLLM({"reviews": reviews}))
    rules = [f.rule for f in flags]
    assert sorted(rules) == ["other", "personal_attributes"]
    assert policy.review("Try Betterly free.", brand, FakeLLM(fail=True)) == []


def test_policy_falls_back_to_default_file() -> None:
    bare = Brand(name="B", product="B", category="c", audience="a", cta="c", tone="t")
    assert policy.blocking(policy.lint("Are you depressed?", bare))
    assert policy.win_rule(bare)["roas_target"] > 0


def test_archetype_catalog_is_complete() -> None:
    expected = {"confession", "contrarian", "curiosity_gap", "pov", "stat_shock", "partner_quote", "mistake", "question_to_self", "us_vs_them", "tiny_habit"}
    assert expected <= set(hooks.ARCHETYPES)
    for name, archetype in hooks.ARCHETYPES.items():
        assert archetype.name == name
        assert archetype.templates and archetype.visuals and archetype.motions and archetype.on_screen


def test_generate_hooks_offline_is_compliant_diverse_and_deterministic(brand: Brand) -> None:
    dna = make_dna()
    accepted, rejected = hooks.generate_hooks(dna, 6, brand, Priors(), seed=11)
    assert [h.id for h in accepted] == [f"H{i:02d}" for i in range(1, 7)]
    assert len({h.text for h in accepted}) == 6
    assert len({h.archetype for h in accepted}) >= 5
    for index, hook in enumerate(accepted):
        assert word_count(hook.text) <= 14
        assert 1 <= word_count(hook.on_screen_text) <= 6
        assert hook.visual.startswith("The same man")
        assert hook.motion and hook.rationale
        assert 0.0 <= hook.score <= 1.0
        assert not policy.blocking(policy.lint(hook.text, brand) + policy.lint(hook.on_screen_text, brand))
        assert jaccard(hook.text, dna.hook_text) <= 0.6
        for other in accepted[index + 1 :]:
            assert jaccard(hook.text, other.text) <= 0.6
    for hook in rejected:
        assert hook.flags and policy.blocking(hook.flags)
    again, _ = hooks.generate_hooks(dna, 6, brand, Priors(), seed=11)
    assert [h.model_dump() for h in again] == [h.model_dump() for h in accepted]
    different, _ = hooks.generate_hooks(dna, 6, brand, Priors(), seed=12)
    assert [h.text for h in different] != [h.text for h in accepted]


def test_generate_hooks_respects_archetype_filter(brand: Brand) -> None:
    accepted, _ = hooks.generate_hooks(make_dna(), 4, brand, Priors(), archetypes=["pov", "question-to-self"], seed=3)
    assert accepted
    assert {h.archetype for h in accepted} <= {"pov", "question_to_self"}


def test_generate_hooks_adapts_to_female_hero(brand: Brand) -> None:
    dna = make_dna(hero_description="the same woman in her early 30s, exactly as in the reference frames", hook_text="My husband and I stopped talking.")
    accepted, _ = hooks.generate_hooks(dna, 4, brand, Priors(), seed=5)
    assert all(h.visual.startswith("The same woman") for h in accepted)


def test_generate_hooks_llm_path_blends_scores_and_guards_policy(brand: Brand) -> None:
    payload = {
        "hooks": [
            {"archetype": "confession", "text": "I thanked the barista more than my wife.", "on_screen_text": "the barista got more thanks", "visual": "He winces at the lens.", "motion": "slow push-in", "rationale": "Specific and self-aware.", "score": 0.95},
            {"archetype": "pov", "text": "Are you lonely in your marriage?", "on_screen_text": "lonely in your marriage?", "visual": "", "motion": "", "rationale": "Direct callout.", "score": 0.9},
            {"archetype": "mistake", "text": "The biggest mistake I made was thinking that being busy every single night was totally normal for us.", "on_screen_text": "my mistake", "visual": "x", "motion": "y", "rationale": "Too long.", "score": 0.8},
            {"archetype": "not_real", "text": "Ignored.", "on_screen_text": "x", "visual": "x", "motion": "x", "rationale": "x", "score": 0.1},
        ]
    }
    llm = FakeLLM({"hooks": payload, "reviews": {"reviews": []}, "changes": {"text": "I felt lonely in my own marriage.", "changes": "first person"}})
    accepted, rejected = hooks.generate_hooks(make_dna(), 3, brand, Priors(), llm=llm, seed=2)
    texts = [h.text for h in accepted]
    assert "I thanked the barista more than my wife." in texts
    assert "Are you lonely in your marriage?" not in texts
    assert any("hook_quality" in {f.rule for f in h.flags} for h in rejected)
    barista = next(h for h in accepted if h.text.startswith("I thanked"))
    assert barista.score > 0.6
    assert barista.visual == "He winces at the lens."
    assert any("hooks" in call["schema"]["properties"] for call in llm.calls)


def test_bandit_allocation_favors_strong_arms_and_explores(brand: Brand) -> None:
    levels = ["a", "b", "c", "d"]
    strong = Priors(factors={"archetype": {"a": BetaArm(alpha=60, beta=2, trials=60, wins=59), "b": BetaArm(alpha=2, beta=60, trials=60, wins=1)}})
    counts = bandit.allocate(random.Random(1), strong, "archetype", levels, 12)
    assert sum(counts.values()) == 12
    assert all(count >= 1 for count in counts.values())
    assert counts["a"] > counts["b"]
    few = bandit.allocate(random.Random(1), strong, "archetype", levels, 2)
    assert sum(few.values()) == 2
    assert few["c"] + few["d"] >= 1


def sample_hooks(count: int) -> list[Hook]:
    names = list(hooks.ARCHETYPES)
    return [
        Hook(id=f"H{i:02d}", archetype=names[i % len(names)], text=f"Hook number {i} about us.", on_screen_text="short line", visual="v", motion="m", score=0.5 + i / 50)
        for i in range(1, count + 1)
    ]


def flat_cost(spec: VariantSpec) -> float:
    return 2.0 if spec.new_shot else 0.5


def test_planner_builds_balanced_covering_plan(voices: list[Voice], locations: list[Location]) -> None:
    hook_list = sample_hooks(6)
    aspects = [Aspect.VERTICAL, Aspect.PORTRAIT, Aspect.SQUARE]
    result = planner.plan("w01", make_dna(), hook_list, voices, locations, aspects, 12, 100.0, Priors(), cost=flat_cost, body_mode="sts", seed=7)
    specs = result.variants
    assert [s.id for s in specs] == [f"V{i:02d}" for i in range(1, 13)]
    assert [s.seed for s in specs] == [7 + 1009 * s.index for s in specs]
    assert all(s.aspects == aspects and s.body_mode == "sts" for s in specs)
    assert {s.hook.id for s in specs} == {h.id for h in hook_list}
    voice_counts = [sum(1 for s in specs if s.voice.id == v.id) for v in voices]
    location_counts = [sum(1 for s in specs if s.location.id == l.id) for l in locations]
    assert max(voice_counts) - min(voice_counts) <= 1
    assert max(location_counts) - min(location_counts) <= 1
    for spec in specs:
        assert spec.new_shot == (not spec.location.original)
        assert "hook" in spec.changed
        assert ("voice" in spec.changed) == (spec.voice.id != voices[0].id)
        assert ("location" in spec.changed) == (not spec.location.original)
    assert result.design.startswith("pairwise covering design, bandit-weighted hooks")
    assert result.coverage["voice_location"] > 0.3
    assert result.coverage["pairwise_attainable"] > 0.8
    assert result.estimate.seconds == pytest.approx(75 * 12)
    assert result.estimate.usd == pytest.approx(sum(flat_cost(s) for s in specs))
    assert set(result.levels) == {"hook", "voice", "location", "aspect"}
    again = planner.plan("w01", make_dna(), hook_list, voices, locations, aspects, 12, 100.0, Priors(), cost=flat_cost, seed=7)
    assert [s.model_dump() for s in again.variants] == [s.model_dump() for s in specs]


def test_planner_selects_top_hooks_when_count_is_small(voices: list[Voice], locations: list[Location]) -> None:
    result = planner.plan("w01", make_dna(), sample_hooks(8), voices, locations, [Aspect.VERTICAL], 3, 100.0, Priors(), cost=flat_cost)
    assert len(result.variants) == 3
    assert len({s.hook.id for s in result.variants}) == 3


def test_planner_budget_switches_locations_then_drops(voices: list[Voice], locations: list[Location]) -> None:
    hook_list = sample_hooks(4)
    switched = planner.plan("w01", make_dna(), hook_list, voices, locations, [Aspect.VERTICAL], 6, 6.5, Priors(), cost=flat_cost)
    assert len(switched.variants) == 6
    assert switched.estimate.usd <= 6.5
    assert any("reuses the original hook shot" in note for note in switched.notes)
    unconstrained = planner.plan("w01", make_dna(), hook_list, voices, locations, [Aspect.VERTICAL], 6, 100.0, Priors(), cost=flat_cost)
    new_shot_ids = [s.id for s in unconstrained.variants if s.new_shot]
    switched_ids = [note.split(":")[0] for note in switched.notes if "reuses the original hook shot" in note]
    assert switched_ids == sorted(new_shot_ids, reverse=True)[: len(switched_ids)]
    dropped = planner.plan("w01", make_dna(), hook_list, voices, locations, [Aspect.VERTICAL], 6, 1.6, Priors(), cost=flat_cost)
    assert len(dropped.variants) == 3
    assert dropped.estimate.usd <= 1.6
    assert all(not s.new_shot for s in dropped.variants)
    assert any("dropped as lowest priority" in note for note in dropped.notes)


def test_planner_falls_back_to_original_hook(voices: list[Voice], locations: list[Location]) -> None:
    result = planner.plan("w01", make_dna(), [], voices, locations, [Aspect.SQUARE], 2, 50.0, Priors(), cost=flat_cost, body_mode="keep")
    assert all(s.hook.id == "H00" and "hook" not in s.changed for s in result.variants)
    assert result.variants[0].body_mode == "keep"


def test_priors_round_trip(tmp_path: Path) -> None:
    assert priors.load(tmp_path / "missing.json") == Priors()
    stored = Priors(factors={"voice": {"warm_30s": BetaArm(alpha=3, beta=2, trials=3, wins=2)}}, updated_at="2026-11-01T00:00:00+00:00")
    path = priors.save(stored, tmp_path / "nested" / "dir" / "priors.json")
    assert path.exists()
    assert priors.load(path) == stored
    (tmp_path / "broken.json").write_text("{not json")
    assert priors.load(tmp_path / "broken.json") == Priors()


def write_results(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_priors_learn_from_meta_export(tmp_path: Path) -> None:
    rows = []
    for aspect in ("9x16", "4x5", "1x1"):
        rows.append([f"Betterly_w01_V01_H02-curiosity-gap_warm-30s_living-room-evening_{aspect}", "40.00", "5000", "3", "2.4"])
        rows.append([f"Betterly_w01_V02_H05-tiny-habit_deep-40s_original_{aspect}", "40.00", "5000", "1", "0.6"])
    rows.append(["Unrelated campaign ad", "100", "1", "1", "9"])
    path = write_results(tmp_path / "results.csv", ["Ad Name", "Amount spent (USD)", "Impressions", "Purchases", "Purchase ROAS (return on ad spend)"], rows)
    start = Priors(factors={"voice": {"warm_30s": BetaArm(alpha=2, beta=1, trials=1, wins=1)}})
    learned = priors.learn(start, path)
    assert learned.updated_at
    assert start.factors["voice"]["warm_30s"].trials == 1
    assert learned.arm("archetype", "curiosity_gap") == BetaArm(alpha=2, beta=1, trials=1, wins=1)
    assert learned.arm("archetype", "tiny_habit") == BetaArm(alpha=1, beta=2, trials=1, wins=0)
    assert learned.arm("voice", "warm_30s") == BetaArm(alpha=3, beta=1, trials=2, wins=2)
    assert learned.arm("location", "living_room_evening").wins == 1
    assert learned.arm("location", "original").trials == 1


def test_priors_learn_uses_win_column_cpa_and_min_spend(tmp_path: Path) -> None:
    flagged = write_results(
        tmp_path / "flagged.csv",
        ["ad_name", "win"],
        [["Betterly_w01_V01_H01-pov_casual-20s_parked-car_9x16", "yes"], ["Betterly_w01_V02_H02-mistake_calm-coach_kitchen-morning_9x16", "0"]],
    )
    learned = priors.learn(Priors(), flagged)
    assert learned.arm("archetype", "pov").wins == 1
    assert learned.arm("archetype", "mistake").beta == 2
    by_cpa = write_results(
        tmp_path / "cpa.csv",
        ["Ad name", "Spend", "Cost per purchase"],
        [["Betterly_w01_V03_H03-stat-shock_warm-30s_home-office_9x16", "60", "30"], ["Betterly_w01_V04_H04-scenario_warm-30s_home-office_9x16", "10", "5"]],
    )
    learned = priors.learn(Priors(), by_cpa, win_rule={"cpa_target": 40.0, "min_spend": 25.0})
    assert learned.arm("archetype", "stat_shock").wins == 1
    assert learned.arm("archetype", "scenario").trials == 0


def test_deconstruct_offline(tmp_path: Path, brand: Brand) -> None:
    winner = make_winner(tmp_path)
    dna = deconstruct.deconstruct(winner, brand, llm=None, frames=[])
    assert dna.hook_text.startswith("I'll be honest")
    assert dna.cta_text == "Try Betterly free."
    assert "two-minute check-in" in dna.body_text
    assert dna.hook_archetype == "confession"
    assert dna.angle.startswith("Reconnection")
    assert "mid-30s" in dna.hero_description and "exactly as in the reference frames" in dna.hero_description
    assert dna.format == "talking_head"
    assert 2.0 < dna.pacing_wps < 3.5
    assert dna.beats[0].startswith("hook:") and dna.beats[-1].startswith("cta:")
    assert "married" in dna.audience
    assert all(getattr(dna, name) for name in ("angle", "audience", "promise", "setting", "tone"))


def test_deconstruct_script_only_and_voiceover(tmp_path: Path, brand: Brand) -> None:
    winner = make_winner(tmp_path, talking_head=False).model_copy(update={"words": [], "hook_end": 0.0, "cta_start": 0.0})
    dna = deconstruct.deconstruct(winner, brand)
    assert dna.hook_text == "I'll be honest, my wife and I had become roommates."
    assert dna.cta_text == "Try Betterly free."
    assert dna.format == "voiceover"
    assert dna.pacing_wps == pytest.approx(2.6)


def test_deconstruct_llm_path_and_failure(tmp_path: Path, brand: Brand) -> None:
    winner = make_winner(tmp_path)
    frame = tmp_path / "frame.jpg"
    import cv2
    import numpy as np

    cv2.imwrite(str(frame), np.full((64, 36, 3), 120, dtype=np.uint8))
    response = {
        "angle": "Roommate syndrome: quiet distance in a busy marriage",
        "audience": "Married men 30-40 with young kids",
        "promise": "Feel close again in minutes a day",
        "hook_archetype": "confession",
        "hook_text": "ignored",
        "body_text": "ignored",
        "cta_text": "ignored",
        "beats": ["hook: admission", "turn: check-in", "cta: try free"],
        "hero_description": "a man in his mid-30s with a short beard and grey hoodie, exactly as in the reference frames",
        "setting": "living room couch, evening lamp light",
        "tone": "confessional, warm",
        "format": "voiceover",
    }
    llm = FakeLLM({"hero_description": response})
    dna = deconstruct.deconstruct(winner, brand, llm=llm, frames=[frame])
    assert dna.angle == response["angle"]
    assert dna.hero_description == response["hero_description"]
    assert dna.hook_text.startswith("I'll be honest")
    assert dna.format == "talking_head"
    assert dna.beats == response["beats"]
    assert llm.calls[0]["images"] == [frame]
    assert "[00." in llm.calls[0]["prompt"]
    schema = llm.calls[0]["schema"]
    assert schema["additionalProperties"] is False and set(schema["required"]) == set(schema["properties"])
    fallback = deconstruct.deconstruct(winner, brand, llm=FakeLLM(fail=True), frames=[frame])
    assert fallback.angle.startswith("Reconnection")


def make_spec(hook_text: str = "Date nights didn't fix us. A six-second kiss did.", screen: str = "date nights didn't fix us", archetype: str = "contrarian", score: float = 0.8) -> VariantSpec:
    hook = Hook(id="H01", archetype=archetype, text=hook_text, on_screen_text=screen, visual="v", motion="m", score=score)
    voice = Voice(id="warm_30s", label="Warm")
    location = Location(id="kitchen_morning", label="Kitchen", prompt="kitchen")
    return VariantSpec(id="V01", index=1, hook=hook, voice=voice, location=location, aspects=[Aspect.VERTICAL], seed=1016)


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    path = tmp_path_factory.mktemp("video") / "tiny.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=180x320:rate=30:duration=2", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def test_prelaunch_score_reads_video(tiny_video: Path) -> None:
    words = timed_words("Date nights didn't fix us. A six-second kiss did. Here is the rest of it.", start=0.05)
    result = predict.prelaunch_score(tiny_video, make_spec(), words, 2.8, Priors())
    assert 0.0 <= result.score <= 100.0
    assert result.features["video_readable"] == 1.0
    assert result.features["motion_energy"] > 0.0
    assert 0.0 < result.features["brightness"] < 1.0
    assert result.reasons and result.reasons[0].startswith("+ ")


def test_prelaunch_score_survives_unreadable_video(tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    words = timed_words("Date nights didn't fix us.", start=0.05)
    result = predict.prelaunch_score(broken, make_spec(), words, 2.4, Priors())
    assert result.features["video_readable"] == 0.0
    assert "motion_energy" not in result.features
    assert any("could not be read" in reason for reason in result.reasons)
    empty = predict.prelaunch_score(tmp_path / "missing.mp4", make_spec(), [], 2.0, Priors())
    assert 0.0 <= empty.score <= 100.0


def test_prelaunch_score_rewards_strong_hooks_and_priors(tiny_video: Path) -> None:
    good_words = timed_words("Date nights didn't fix us. A six-second kiss did.", start=0.05)
    slow_words = timed_words("So um today I want to talk about a thing that happened to me and my wife at home recently okay", start=1.3, word_seconds=0.12, gap=0.02)
    good = predict.prelaunch_score(tiny_video, make_spec(), good_words, 2.8, Priors())
    weak = predict.prelaunch_score(tiny_video, make_spec(screen="", score=0.3), slow_words, 4.4, Priors())
    assert good.score > weak.score
    assert any(reason.startswith("- ") for reason in weak.reasons)
    winning = Priors(factors={"archetype": {"contrarian": BetaArm(alpha=30, beta=5, trials=33, wins=29)}})
    boosted = predict.prelaunch_score(tiny_video, make_spec(), good_words, 2.8, winning)
    assert boosted.score > good.score
    assert boosted.features["prior_archetype"] == pytest.approx(30 / 35, abs=1e-3)


def test_body_and_partner_shaming_are_blocked(brand: Brand) -> None:
    weight = policy.lint("Lose 20 pounds and she'll want you again", brand)
    assert {"body_and_weight", "negative_self_perception"} <= {flag.rule for flag in weight}
    assert policy.repair("Lose 20 pounds and she'll want you again", weight, brand) is None
    shaming = "Struggling with ED? Your wife deserves better"
    fixed = policy.repair(shaming, policy.lint(shaming, brand), brand)
    assert fixed is not None
    assert "deserves better" not in fixed
    assert not policy.blocking(policy.lint(fixed, brand))
