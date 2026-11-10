from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from mitosis.config import PACKAGE_ROOT, Brand, read_yaml
from mitosis.intelligence.llmio import array, ask_json, strict_object, string
from mitosis.intelligence.text import normalize, split_sentences, tidy
from mitosis.models import PolicyFlag
from mitosis.providers.base import LanguageModel

log = logging.getLogger(__name__)

SEVERITY_RANK = {"warn": 1, "block": 2}
EMOJI = re.compile(
    r"[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\u2190-\u21FF\u2300-\u23FF\u25A0-\u25FF\uFE0F\u200D]"
)
IGNORED_JOINERS = {"\ufe0f", "\u200d"}
TOKEN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)*")
CAPS_WORD = re.compile(r"\b[A-Z][A-Z']*[A-Z]\b|\b[A-Z]\b")


@dataclass(frozen=True)
class Group:
    name: str
    label: str
    matcher: re.Pattern[str]
    reframe: str


@dataclass(frozen=True)
class Rule:
    id: str
    severity: str
    kind: str
    category: str
    explanation: str
    suggestion: str
    repair: str
    matcher: re.Pattern[str] | None = None
    substitutions: tuple[tuple[re.Pattern[str], str], ...] = ()
    referents: frozenset[str] = frozenset()
    adjacent: frozenset[str] = frozenset()
    openers: re.Pattern[str] | None = None
    window: int = 6
    groups: tuple[Group, ...] = ()
    min_text_words: int = 0
    min_word_length: int = 3
    max_caps_words: int = 1
    allow: frozenset[str] = frozenset()
    max_emoji: int = 0


@dataclass(frozen=True)
class Hit:
    rule: str
    span: str
    group: str = ""


def default_policy_path() -> Path:
    return Path(os.environ.get("MITOSIS_HOME") or PACKAGE_ROOT) / "config" / "policy.yaml"


@lru_cache(maxsize=4)
def load_policy_file(path: str) -> dict[str, Any]:
    location = Path(path)
    if not location.exists():
        log.warning("policy file %s not found, policy checks disabled", location)
        return {}
    return read_yaml(location)


def policy_data(brand: Brand | None = None) -> dict[str, Any]:
    if brand is not None and brand.policy:
        return brand.policy
    return load_policy_file(str(default_policy_path()))


def win_rule(brand: Brand | None = None) -> dict[str, float]:
    rule = policy_data(brand).get("win_rule", {}) or {}
    return {key: float(value) for key, value in rule.items() if isinstance(value, (int, float))}


def phrase_pattern(phrase: str) -> str:
    parts = [re.escape(piece) for piece in re.split(r"[\s-]+", phrase.strip()) if piece]
    return r"(?<![\w'])" + r"[\s-]+".join(parts) + r"(?![\w'])"


def compile_matcher(terms: list[str], patterns: list[str]) -> re.Pattern[str] | None:
    alternatives = [phrase_pattern(str(term)) for term in sorted(terms, key=lambda t: -len(str(t)))]
    alternatives += [f"(?:{pattern})" for pattern in patterns]
    if not alternatives:
        return None
    return re.compile("|".join(alternatives), re.IGNORECASE)


def compile_rule(raw: dict[str, Any]) -> Rule:
    groups = tuple(
        Group(
            name=name,
            label=str(spec.get("label", name.replace("_", " "))),
            matcher=compile_matcher(list(spec.get("terms", [])), list(spec.get("patterns", []))) or re.compile(r"(?!)"),
            reframe=str(spec.get("reframe", "")),
        )
        for name, spec in (raw.get("groups") or {}).items()
    )
    openers = list(raw.get("openers", []))
    return Rule(
        id=str(raw["id"]),
        severity="block" if raw.get("severity") == "block" else "warn",
        kind=str(raw.get("kind", "lexicon")),
        category=str(raw.get("category", "")),
        explanation=str(raw.get("explanation", "")),
        suggestion=str(raw.get("suggestion", "")),
        repair=str(raw.get("repair", "drop_sentence")),
        matcher=compile_matcher(list(raw.get("terms", [])), list(raw.get("patterns", []))),
        substitutions=tuple(
            (re.compile(item["match"], re.IGNORECASE), str(item.get("replace", ""))) for item in raw.get("substitutions", [])
        ),
        referents=frozenset(str(r).lower() for r in raw.get("referents", [])),
        adjacent=frozenset(str(r).lower() for r in raw.get("adjacent_referents", [])),
        openers=re.compile(r"^\s*(?:" + "|".join(phrase_pattern(o) for o in openers) + ")", re.IGNORECASE) if openers else None,
        window=int(raw.get("window", 6)),
        groups=groups,
        min_text_words=int(raw.get("min_text_words", 0)),
        min_word_length=int(raw.get("min_word_length", 3)),
        max_caps_words=int(raw.get("max_caps_words", 1)),
        allow=frozenset(str(a).upper() for a in raw.get("allow", [])),
        max_emoji=int(raw.get("max_emoji", 0)),
    )


@lru_cache(maxsize=8)
def compile_rules(serialized: str) -> tuple[Rule, ...]:
    return tuple(compile_rule(raw) for raw in json.loads(serialized) if isinstance(raw, dict) and raw.get("id"))


def ruleset(brand: Brand | None = None) -> tuple[Rule, ...]:
    return compile_rules(json.dumps(policy_data(brand).get("rules", []), sort_keys=True))


def rule_index(brand: Brand | None = None) -> dict[str, Rule]:
    return {rule.id: rule for rule in ruleset(brand)}


def proximity_hits(rule: Rule, sentence: str) -> list[Hit]:
    spans = [(m.start(), m.end(), m.group(0).lower()) for m in TOKEN.finditer(sentence)]
    if not spans:
        return []
    referent_positions = [i for i, (_, _, word) in enumerate(spans) if word in rule.referents]
    if rule.openers is not None and rule.openers.search(sentence):
        referent_positions.append(0)
    hits: list[Hit] = []
    for group in rule.groups:
        for match in group.matcher.finditer(sentence):
            covered = [i for i, (start, end, _) in enumerate(spans) if start < match.end() and end > match.start()]
            if not covered:
                continue
            first, last = covered[0], covered[-1]
            near = any(min(abs(p - first), abs(p - last)) <= rule.window or first <= p <= last for p in referent_positions)
            adjacent = first > 0 and spans[first - 1][2] in rule.adjacent
            if near or adjacent:
                hits.append(Hit(rule=rule.id, span=match.group(0), group=group.name))
    return hits


def caps_hits(rule: Rule, text: str) -> list[Hit]:
    if len(text.split()) < rule.min_text_words:
        return []
    shouted = [
        word
        for word in CAPS_WORD.findall(text)
        if len(word.replace("'", "")) >= rule.min_word_length and word.replace("'", "") not in rule.allow
    ]
    if len(shouted) > rule.max_caps_words:
        return [Hit(rule=rule.id, span=" ".join(shouted[:6]))]
    return []


def emoji_hits(rule: Rule, text: str) -> list[Hit]:
    found = [char for char in EMOJI.findall(text) if char not in IGNORED_JOINERS]
    if len(found) > rule.max_emoji:
        return [Hit(rule=rule.id, span="".join(found[:6]))]
    return []


def detect(text: str, rules: tuple[Rule, ...]) -> list[Hit]:
    text = normalize(text)
    hits: list[Hit] = []
    for rule in rules:
        if rule.kind == "proximity":
            for sentence in split_sentences(text):
                hits.extend(proximity_hits(rule, sentence))
        elif rule.kind == "caps":
            hits.extend(caps_hits(rule, text))
        elif rule.kind == "emoji":
            hits.extend(emoji_hits(rule, text))
        elif rule.matcher is not None:
            hits.extend(Hit(rule=rule.id, span=m.group(0)) for m in rule.matcher.finditer(text))
    return hits


def flag_for(rule: Rule, spans: list[str], groups: list[str]) -> PolicyFlag:
    labels = [g.label for g in rule.groups if g.name in groups]
    prefix = f"Implies the viewer's {', '.join(labels)}. " if labels else ""
    return PolicyFlag(
        rule=rule.id,
        severity=rule.severity,
        span=", ".join(spans),
        message=f"{prefix}{rule.explanation}".strip(),
        suggestion=rule.suggestion,
    )


def lint(text: str, brand: Brand) -> list[PolicyFlag]:
    rules = ruleset(brand)
    by_rule: dict[str, tuple[list[str], list[str]]] = {}
    for hit in detect(text, rules):
        spans, groups = by_rule.setdefault(hit.rule, ([], []))
        if hit.span.lower() not in (s.lower() for s in spans):
            spans.append(hit.span)
        if hit.group and hit.group not in groups:
            groups.append(hit.group)
    index = {rule.id: rule for rule in rules}
    return [flag_for(index[rule_id], spans, groups) for rule_id, (spans, groups) in by_rule.items()]


def blocking(flags: list[PolicyFlag]) -> bool:
    return any(flag.severity == "block" for flag in flags)


def merge(primary: list[PolicyFlag], secondary: list[PolicyFlag]) -> list[PolicyFlag]:
    merged: dict[str, PolicyFlag] = {}
    for flag in [*primary, *secondary]:
        current = merged.get(flag.rule)
        if current is None:
            merged[flag.rule] = flag
        elif SEVERITY_RANK[flag.severity] > SEVERITY_RANK[current.severity]:
            merged[flag.rule] = current.model_copy(update={"severity": flag.severity})
    return list(merged.values())


REVIEW_SYSTEM = (
    "You are a senior Meta ads policy reviewer for a US relationship and intimacy coaching app whose ads target adults 18+. "
    "Apply the Meta Advertising Standards as enforced in late 2026: personal attributes (never assert or imply that the viewer or "
    "their family has a health condition, sexual practice or orientation, financial status, age, faith or similar trait), adult "
    "sexual content and sexually explicit language, sexual and reproductive health that must not focus on pleasure or enhancement, "
    "negative self-perception and insecurity exploitation, before/after and unrealistic or guaranteed outcomes, sensational and "
    "clickbait language, engagement bait, non-functional buttons, unsubstantiated or medical claims, and profanity. "
    "Judge the literal text the way a reviewer would, flag only real risks, and prefer the provided rule ids. "
    "Use rule id 'other' for a risk that fits none of them."
)


def rule_catalog(brand: Brand) -> str:
    return "\n".join(f"- {rule.id} ({rule.severity}): {rule.explanation}" for rule in ruleset(brand))


def flag_schema(rule_ids: list[str]) -> dict[str, Any]:
    return strict_object(
        {
            "rule": string([*rule_ids, "other"]),
            "severity": string(["block", "warn"]),
            "span": string(),
            "message": string(),
            "suggestion": string(),
        }
    )


def parse_flags(items: Any) -> list[PolicyFlag]:
    flags: list[PolicyFlag] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not item.get("rule"):
            continue
        flags.append(
            PolicyFlag(
                rule=str(item["rule"]),
                severity="block" if item.get("severity") == "block" else "warn",
                span=str(item.get("span", "")),
                message=str(item.get("message", "")) or "Flagged by policy review.",
                suggestion=str(item.get("suggestion", "")),
            )
        )
    return flags


def review_many(texts: list[str], brand: Brand, llm: LanguageModel | None = None) -> list[list[PolicyFlag]]:
    linted = [lint(text, brand) for text in texts]
    if llm is None or not texts:
        return linted
    rule_ids = [rule.id for rule in ruleset(brand)]
    schema = strict_object(
        {"reviews": array(strict_object({"index": {"type": "integer"}, "flags": array(flag_schema(rule_ids))}))}
    )
    numbered = "\n".join(f"{i}: {text}" for i, text in enumerate(texts))
    prompt = (
        f"Brand: {brand.name}, {brand.category}. Audience: {brand.audience}. Tone: {brand.tone}.\n"
        f"Rules:\n{rule_catalog(brand)}\n\n"
        f"Review each numbered ad text independently and return one review per index (empty flags when compliant).\n{numbered}"
    )
    data = ask_json(llm, REVIEW_SYSTEM, prompt, schema, max_tokens=3000)
    if data is None:
        return linted
    extra: dict[int, list[PolicyFlag]] = {}
    for review in data.get("reviews", []) if isinstance(data.get("reviews"), list) else []:
        if isinstance(review, dict) and isinstance(review.get("index"), int):
            extra[review["index"]] = parse_flags(review.get("flags"))
    return [merge(flags, extra.get(i, [])) for i, flags in enumerate(linted)]


def review(text: str, brand: Brand, llm: LanguageModel | None = None) -> list[PolicyFlag]:
    return review_many([text], brand, llm)[0]


REPAIR_SYSTEM = (
    "You are a direct-response copywriter who fixes Meta ad policy problems without losing the hook's intent, energy or length. "
    "Rewrite in first person or at the group level instead of telling the viewer what they are, keep it non-explicit, remove "
    "guarantees, hype and profanity, and keep roughly the same word count."
)


def llm_rewrite(text: str, flags: list[PolicyFlag], brand: Brand, llm: LanguageModel) -> str | None:
    issues = "\n".join(f"- {f.rule} ({f.severity}) on '{f.span}': {f.message} Fix: {f.suggestion}" for f in flags)
    prompt = (
        f"Brand: {brand.name} ({brand.category}). Tone: {brand.tone}.\n"
        f"Original text: {text}\nPolicy issues:\n{issues}\n"
        "Return the compliant rewrite and a one-line note on what changed."
    )
    data = ask_json(llm, REPAIR_SYSTEM, prompt, strict_object({"text": string(), "changes": string()}), max_tokens=800)
    rewritten = tidy(str(data.get("text", ""))) if data else ""
    return rewritten or None


def substitute(sentence: str, rule: Rule) -> str | None:
    updated = sentence
    for pattern, replacement in rule.substitutions:
        updated = pattern.sub(replacement, updated)
    if rule.kind != "proximity" and rule.matcher is not None and rule.matcher.search(normalize(updated)):
        return None
    return updated


def normalize_caps(sentence: str, rule: Rule) -> str:
    def soften(match: re.Match[str]) -> str:
        word = match.group(0)
        if len(word.replace("'", "")) < rule.min_word_length or word.replace("'", "") in rule.allow:
            return word
        return word.capitalize() if match.start() == 0 else word.lower()

    return CAPS_WORD.sub(soften, sentence)


def normalize_punctuation(sentence: str) -> str:
    sentence = re.sub(r"\.{4,}", "...", sentence)
    sentence = re.sub(r"([!?])(?:\s*[!?])+", r"\1", sentence)
    while sentence.count("!") > 1:
        sentence = sentence.replace("!", ".", 1)
    return sentence


def reframe_for(rule: Rule, hit: Hit, used_reframes: set[str]) -> str | None:
    group = next((g for g in rule.groups if g.name == hit.group), None)
    reframe = group.reframe if group else ""
    if not reframe or reframe in used_reframes:
        return None
    used_reframes.add(reframe)
    return reframe


def rewrite_spans(sentence: str, rule: Rule) -> str | None:
    if rule.repair == "substitute":
        return substitute(sentence, rule)
    if rule.repair == "normalize_caps":
        return normalize_caps(sentence, rule)
    if rule.repair == "normalize_punctuation":
        return normalize_punctuation(sentence)
    if rule.repair == "strip_emoji":
        return EMOJI.sub("", sentence)
    return sentence


def fix_sentence(sentence: str, rules: tuple[Rule, ...], used_reframes: set[str]) -> str | None:
    index = {rule.id: rule for rule in rules}
    hits = detect(sentence, rules)
    for hit in hits:
        rule = index[hit.rule]
        if rule.repair == "reframe":
            return reframe_for(rule, hit, used_reframes)
    updated: str | None = sentence
    for rule_id in dict.fromkeys(hit.rule for hit in hits):
        updated = rewrite_spans(updated, index[rule_id])
        if updated is None:
            return None
    if any(index[hit.rule].repair == "drop_sentence" for hit in detect(updated, rules)):
        return None
    return updated


def rule_based_rewrite(text: str, rules: tuple[Rule, ...]) -> str | None:
    current = normalize(text)
    for _ in range(3):
        if not detect(current, rules):
            break
        used: set[str] = set()
        fixed = [fix_sentence(sentence, rules, used) for sentence in split_sentences(current)]
        current = tidy(" ".join(sentence for sentence in fixed if sentence))
        if not current:
            return None
    if not text.strip().endswith((".", "!", "?")) and len(split_sentences(current)) == 1:
        current = current.rstrip(".")
    return current or None


def repair(text: str, flags: list[PolicyFlag], brand: Brand, llm: LanguageModel | None = None) -> str | None:
    original = normalize(text)
    current_flags = lint(original, brand)
    known = {rule.id for rule in ruleset(brand)}
    unresolvable = [f for f in flags if f.severity == "block" and f.rule not in known]
    if not current_flags and not unresolvable:
        return original
    if llm is not None:
        rewritten = llm_rewrite(original, merge(current_flags, flags), brand, llm)
        if rewritten and not blocking(lint(rewritten, brand)):
            return rewritten
    if unresolvable:
        return None
    rewritten = rule_based_rewrite(original, ruleset(brand))
    if rewritten is None or blocking(lint(rewritten, brand)):
        return None
    return rewritten
