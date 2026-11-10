from __future__ import annotations

import logging
import random
import re
import string as formatter_lib
from dataclasses import dataclass, field

from pydantic import Field

from mitosis.config import Brand
from mitosis.intelligence import bandit, policy
from mitosis.intelligence.copybank import Angle, angle_for, partner_word
from mitosis.intelligence.llmio import array, ask_json, strict_object, string
from mitosis.intelligence.text import (
    article,
    capitalize_first,
    jaccard,
    normalize,
    speech_seconds,
    tidy,
    tokens,
    total_syllables,
    word_count,
)
from mitosis.models import CreativeDNA, Hook, Model, PolicyFlag, Priors
from mitosis.providers.base import LanguageModel

log = logging.getLogger(__name__)

MAX_SPOKEN_WORDS = 14
MAX_SPOKEN_SECONDS = 3.5
MAX_SCREEN_WORDS = 6
DUPLICATE_JACCARD = 0.6
MMR_LAMBDA = 0.72
SAME_ARCHETYPE_PENALTY = 0.35
LLM_WEIGHT = 0.6


class Archetype(Model):
    name: str
    label: str
    description: str
    templates: list[str]
    visuals: list[str]
    motions: list[str]
    on_screen: list[str]
    cues: list[str] = Field(default_factory=list)
    why: str = ""


ARCHETYPES: dict[str, Archetype] = {
    "confession": Archetype(
        name="confession",
        label="Confession",
        description="First-person admission of something most men never say out loud about their relationship.",
        templates=[
            "I'll be honest: {problem}.",
            "Real talk: for {years}, {problem}.",
            "I didn't want to admit {problem}.",
            "{Problem}, and I didn't even notice.",
        ],
        visuals=[
            "looks down, exhales, then lifts {his} eyes to the lens as if admitting something",
            "rubs {his} jaw, half-laughs at {him}self, then looks straight into the phone camera",
        ],
        motions=["slow push-in from medium to close-up on {his} face", "handheld selfie framing with a gentle breathing drift"],
        on_screen=["I didn't see it coming", "nobody warned me about this", "my honest {bond} confession", "real talk about my {bond}"],
        cues=[r"^\s*(?:i'll be honest|honestly|real talk|i'll admit|i have to admit|confession|i never (?:told|said))", r"\bi didn't want to admit\b"],
        why="a first-person admission earns trust before the viewer decides to scroll",
    ),
    "contrarian": Archetype(
        name="contrarian",
        label="Contrarian",
        description="Challenges the default relationship advice (date nights, grand gestures) and teases what actually worked.",
        templates=[
            "{belief} {A_habit} did.",
            "Skip the grand gesture. Try {a_habit} instead.",
            "Unpopular opinion: {belief_lower}",
            "The thing that helped us most wasn't romantic at all.",
        ],
        visuals=[
            "shakes {his} head with a half-smile, then points at the lens on the first word",
            "raises one hand like saying stop, then leans into the camera",
        ],
        motions=["quick snap zoom-in on the first word, then steady", "static framing with a subtle push-in on the reveal"],
        on_screen=["{belief_screen}", "not what I expected", "this worked, date nights didn't"],
        cues=[r"^\s*(?:stop|forget|don't|unpopular opinion|everyone says|hot take)\b", r"\b(?:didn't|wasn't|isn't) (?:fix|the answer|help)\w*"],
        why="challenging a common belief is a pattern interrupt that signals new information",
    ),
    "curiosity_gap": Archetype(
        name="curiosity_gap",
        label="Curiosity gap",
        description="Opens a specific loop (a result, a reaction, a detail) that only the rest of the video closes.",
        templates=[
            "What my {partner} said after {days} of this.",
            "I tried one small thing for a week. {She} noticed by day three.",
            "{She} noticed on day three. I never told {her}.",
            "Here's the tiny thing that changed our evenings.",
        ],
        visuals=[
            "leans in close to the phone camera and lowers {his} voice like sharing a secret",
            "glances over {his} shoulder, then back to the lens with a small knowing smile",
        ],
        motions=["slow creeping push-in to an intimate close-up", "slight handheld lean-in matching {his} movement"],
        on_screen=["what {she} said on day {days_n}", "{she} noticed by day 3", "the tiny thing that worked"],
        cues=[r"\b(?:this one|the one thing|here's what|here's the|what happened|nobody talks about|the reason|what (?:my \w+|she) said|noticed)\b"],
        why="an open loop anchored in a concrete detail makes the viewer stay for the payoff",
    ),
    "pov": Archetype(
        name="pov",
        label="POV",
        description="Native POV framing that drops the viewer into a small, recognizable relationship moment.",
        templates=[
            "POV: you put the phone down and {she} actually noticed.",
            "POV: day {day} of the {habit}.",
            "POV: you stopped saying fine and asked a real question.",
        ],
        visuals=[
            "holds the phone at arm's length selfie-style and raises {his} eyebrows at the lens",
            "sets something down off-frame, then turns to the camera with a slow nod",
        ],
        motions=["arm's-length selfie framing with natural handheld sway", "slow orbit a few degrees to the right while {he} talks"],
        on_screen=["POV: phone down, eyes up", "POV: day {day}", "POV: you actually listened"],
        cues=[r"^\s*pov\b"],
        why="POV framing reads as native content rather than an ad",
    ),
    "stat_shock": Archetype(
        name="stat_shock",
        label="Stat shock (personal number)",
        description="Leads with a specific personal number from the creator's own relationship, never an unverifiable statistic.",
        templates=["{number}"],
        visuals=[
            "counts on {his} fingers, then looks into the lens deadpan",
            "holds up fingers for the number, lets {his} hand drop and raises an eyebrow",
        ],
        motions=["locked-off frame with a sharp punch-in on the number", "static medium shot with a tiny push-in on the pause"],
        on_screen=["{number_screen}"],
        cues=[
            r"^\s*(?:\d+|one|two|three|five|six|seven|nine|ten|thirty)\b[^.?!]{0,30}\b(?:days?|minutes?|seconds?|times|nights?|weeks?)\b",
            r"\b(?:we went|i counted|we tracked)\b",
        ],
        why="a specific personal number is concrete and instantly scannable without making a statistical claim",
    ),
    "partner_quote": Archetype(
        name="partner_quote",
        label="Partner quote",
        description="Opens with a line the partner actually said, so the problem is heard from her side.",
        templates=[
            "My {partner} said: {quote}",
            "{quote} That's what my {partner} told me.",
            "{quote} Hearing that from my {partner} hurt.",
        ],
        visuals=[
            "glances off-frame as if someone just spoke, then turns back to the lens, serious",
            "pauses mid-breath, eyes down, then looks into the camera and repeats the line",
        ],
        motions=["slow push-in during the quote, holding on {his} reaction", "handheld close-up with a gentle drift toward {his} eyes"],
        on_screen=["what {she} told me hurt", "{she} said this to me", "the sentence that woke me up"],
        cues=[r"\b(?:my (?:wife|girlfriend|partner|husband) (?:said|told me|asked)|(?:she|he) (?:said|told me|asked))\b", r"^\s*[\"']"],
        why="a partner's exact words are emotionally specific and show the stakes without accusing the viewer",
    ),
    "mistake": Archetype(
        name="mistake",
        label="Mistake",
        description="Owns a specific mistake the creator made as a partner and hints at the fix.",
        templates=[
            "The mistake I made for {years} as a {role}.",
            "My biggest mistake as a {role}? {Kept}.",
            "I thought {myth}. I was wrong.",
            "Don't make my mistake: {kept}.",
        ],
        visuals=[
            "winces slightly and rubs the back of {his} neck, then looks straight into the lens",
            "lets out a short laugh at {him}self and shakes {his} head before speaking",
        ],
        motions=["static medium framing that slowly tightens to a close-up", "subtle handheld push-in on the admission"],
        on_screen=["my biggest {role} mistake", "don't make my mistake", "I was so wrong"],
        cues=[r"\b(?:mistake|i was wrong|i got (?:it|this) wrong|biggest regret|i thought)\b"],
        why="owning a mistake makes the advice feel earned rather than preached",
    ),
    "question_to_self": Archetype(
        name="question_to_self",
        label="Question to self",
        description="The creator asks himself the uncomfortable question, so the viewer answers it privately.",
        templates=[
            "Why did I stop {stopped}?",
            "Honest question: when did we last {shared}?",
            "What happened to the guy who used to {used_to}?",
        ],
        visuals=[
            "stares off-frame thinking, then turns to the camera mid-question",
            "rests {his} chin on {his} hand, then looks up into the lens as {he} asks",
        ],
        motions=["slow dolly-in as {he} turns toward the lens", "static frame with a gentle focus pull onto {his} eyes"],
        on_screen=["when did I stop trying?", "honest question for myself", "what happened to that guy?"],
        cues=[r"^\s*(?:honest question:\s*)?(?:why|when|how|what) (?:did|do|have|was|happened)\b[^?]*\b(?:i|we|me|us)\b"],
        why="a self-directed question invites reflection without the ad naming the viewer's situation",
    ),
    "us_vs_them": Archetype(
        name="us_vs_them",
        label="Us vs them",
        description="Contrasts what close couples do with what drifting couples do.",
        templates=[
            "Couples who last {us}. The rest {them}.",
            "Some couples {them}. We {us}.",
            "Most couples {them}. Close ones {us}.",
        ],
        visuals=[
            "gestures to one side, then the other, as {he} contrasts two kinds of couples",
            "counts a point on each hand, then brings {his} hands together",
        ],
        motions=["slight lateral slide following {his} gesture", "static medium shot with a punch-in on the second line"],
        on_screen=["couples who last vs drift", "what close couples do daily", "two kinds of couples"],
        cues=[r"\bcouples who\b", r"\b(?:most (?:guys|men|couples)|some couples|happy couples)\b"],
        why="a clean contrast lets the viewer sort himself into the better group without being labeled",
    ),
    "tiny_habit": Archetype(
        name="tiny_habit",
        label="Tiny habit",
        description="Leads with one small, specific, doable action and the change it made.",
        templates=[
            "One {habit}. That's what changed things for us.",
            "Try this tonight: {a_habit}.",
            "Just {a_habit}. Now {outcome}.",
            "The {habit} that changed our evenings.",
        ],
        visuals=[
            "holds up one finger to the lens with a small confident nod",
            "mimes a quick demonstration with {his} hands, then smiles at the camera",
        ],
        motions=["gentle push-in on the raised finger, then settle on {his} face", "handheld framing with a soft focus pull from hand to face"],
        on_screen=["the {habit} habit", "try this tonight", "one tiny habit"],
        cues=[r"\b(?:habit|ritual|tiny|small thing|one thing)\b", r"\b(?:a day|every day|daily|every night|each night|tonight)\b"],
        why="a tiny, specific action feels doable and makes the promise believable",
    ),
    "scenario": Archetype(
        name="scenario",
        label="Scenario drop-in",
        description="Starts mid-scene in a familiar evening or dinner moment before any product appears.",
        templates=["{scene}", "{clock}. Same couch, two phones, zero words."],
        visuals=[
            "sits back scrolling {his} phone, then looks up into the camera as {he} starts talking",
            "sets {his} phone face down, exhales, and turns to the lens",
        ],
        motions=["slow push-in from a resting medium shot to a close-up", "observational static frame, then a subtle handheld drift"],
        on_screen=["sound familiar?", "this was us every night", "every night at {clock}"],
        cues=[r"^\s*(?:it's|it was|sunday|friday|every night|dinner's|kids|another|bedtime)\b", r"\b\d{1,2}(?::\d\d)?\s?(?:pm|am)\b"],
        why="dropping into a familiar mid-scene moment hooks recognition before any pitch",
    ),
}

STRONG_OPENERS = {
    "i", "i'll", "i've", "my", "we", "we'd", "she", "he", "stop", "nobody", "why", "when", "what", "pov", "most", "couples",
    "date", "one", "real", "honest", "unpopular", "every", "sunday", "friday", "dinner's", "kids", "same", "six", "nine",
    "five", "thirty", "try", "don't", "some", "it's", "here's", "the",
}
WEAK_OPENERS = {"so", "hey", "hi", "um", "uh", "today", "okay", "ok", "guys", "are", "do", "have", "if", "in", "this", "is"}
CONCRETE = {
    "phone", "phones", "dinner", "couch", "kitchen", "kiss", "text", "texts", "minute", "minutes", "seconds", "days", "day",
    "night", "nights", "sunday", "friday", "kids", "coffee", "walk", "door", "tv", "question", "check-in", "wife", "girlfriend",
    "partner", "husband", "bedtime", "week", "month", "hug", "thank-you", "barista", "laugh", "talk", "words", "fine", "fight",
    "roommates", "apology", "compliment",
}
ABSTRACT = {"journey", "transform", "unlock", "elevate", "amazing", "incredible", "ultimate", "solution", "potential", "empower", "revolutionary", "secret", "level", "best"}
OPEN_LOOP = re.compile(r"\b(?:this|here's|what|why|the thing|one thing|nobody|noticed|didn't|wasn't|stop|instead|until)\b|[:?]", re.IGNORECASE)
CONTRAST = re.compile(r"\b(?:but|didn't|wasn't|instead|not|stop|never|vs)\b", re.IGNORECASE)
STALE = ("i was skeptical", "this changed my life", "game changer", "obsessed", "you need this", "run don't walk", "hear me out", "let me tell you", "hey guys", "so i tried")
OBJECT_FOLLOWERS = {"", "for", "like", "laugh", "feel", "to", "on", "out", "up", "and", "with", "again", "just", "about", "first", "back", "over", "smile", "know", "a", "the"}


@dataclass
class Draft:
    archetype: str
    text: str
    on_screen: str
    visual: str
    motion: str
    rationale: str = ""
    llm_score: float | None = None
    flags: list[PolicyFlag] = field(default_factory=list)
    score: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class Persona:
    he: str
    his: str
    him: str
    subject: str


def resolve_archetypes(names: list[str] | None) -> list[str]:
    if not names:
        return list(ARCHETYPES)
    resolved = [key for key in (normalize_name(name) for name in names) if key in ARCHETYPES]
    if not resolved:
        log.warning("none of the requested archetypes %s exist, using all", names)
        return list(ARCHETYPES)
    return list(dict.fromkeys(resolved))


def normalize_name(name: str) -> str:
    return re.sub(r"[\s-]+", "_", name.strip().lower())


def hero_persona(description: str) -> Persona:
    lowered = description.lower()
    if re.search(r"\b(?:woman|female|she)\b", lowered):
        return Persona(he="she", his="her", him="her", subject="The same woman from the reference frames")
    if re.search(r"\b(?:man|male|guy|he)\b", lowered) or not lowered:
        return Persona(he="he", his="his", him="him", subject="The same man from the reference frames")
    return Persona(he="they", his="their", him="them", subject="The same person from the reference frames")


def swap_partner_pronouns(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        word, following = match.group(1), (match.group(2) or "").strip().lower()
        lowered = word.lower()
        if lowered == "she":
            swapped = "he"
        elif lowered == "hers":
            swapped = "his"
        else:
            swapped = "him" if following in OBJECT_FOLLOWERS or not following[:1].isalpha() else "his"
        swapped = swapped.capitalize() if word[0].isupper() else swapped
        return swapped + (match.group(2) or "")

    return re.sub(r"\b(she|her|hers)\b(\s+[\w']+)?", replace, text, flags=re.IGNORECASE)


@dataclass
class Slots:
    values: dict[str, list[str]]
    numbers: list[tuple[str, str]]
    beliefs: list[str]
    habits: list[str]
    days: list[tuple[str, str]]

    def choose(self, rng: random.Random) -> dict[str, str]:
        fill = {name: rng.choice(options) for name, options in self.values.items() if options}
        habit = rng.choice(self.habits)
        belief = rng.choice(self.beliefs)
        number, number_screen = rng.choice(self.numbers)
        days, days_n = rng.choice(self.days)
        fill.update(
            habit=habit,
            a_habit=f"{article(habit)} {habit}",
            A_habit=capitalize_first(f"{article(habit)} {habit}"),
            belief=belief,
            belief_lower=belief[:1].lower() + belief[1:],
            number=number,
            number_screen=number_screen,
            days=days,
            days_n=days_n,
        )
        if word_count(belief) <= MAX_SCREEN_WORDS:
            fill["belief_screen"] = belief.rstrip(".")
        fill["Problem"] = capitalize_first(fill.get("problem", ""))
        fill["Kept"] = capitalize_first(fill.get("kept", ""))
        fill["She"] = capitalize_first(fill.get("she", "she"))
        return fill


def build_slots(dna: CreativeDNA, angle: Angle, rng: random.Random) -> Slots:
    corpus = " ".join([dna.hook_text, dna.body_text, dna.cta_text, dna.audience, dna.promise])
    partner = partner_word(corpus)
    male_partner = partner in {"husband", "boyfriend"}
    role = {"wife": "husband", "girlfriend": "boyfriend", "husband": "wife", "boyfriend": "girlfriend"}.get(partner, "partner")

    def adapt(lines: list[str]) -> list[str]:
        return [swap_partner_pronouns(line) if male_partner else line for line in lines]

    numbers = [
        (adapt([line.spoken.format(n=n, partner=partner)])[0], line.screen.format(n=n, partner=partner))
        for line in angle.numbers
        for n in line.values
    ]
    rng.shuffle(numbers)
    return Slots(
        values={
            "problem": adapt(angle.problems),
            "stopped": adapt(angle.stopped),
            "kept": adapt(angle.kept),
            "used_to": adapt(angle.used_to),
            "shared": adapt(angle.shared),
            "myth": adapt(angle.myths),
            "outcome": adapt(angle.outcomes),
            "quote": angle.quotes,
            "scene": adapt(angle.scenes),
            "us": angle.us,
            "them": angle.them,
            "partner": [partner],
            "role": [role],
            "bond": ["marriage" if partner in {"wife", "husband"} else "relationship"],
            "years": ["two years", "three years", "five years", "six years"],
            "day": ["4", "9", "12", "19"],
            "clock": ["10pm", "9:30"],
            "she": ["he" if male_partner else "she"],
            "her": ["him" if male_partner else "her"],
        },
        numbers=numbers,
        beliefs=adapt(angle.beliefs),
        habits=angle.habits,
        days=[("28 days", "28"), ("two weeks", "14"), ("ten days", "10")],
    )


def template_fields(template: str) -> list[str]:
    return [name for _, name, _, _ in formatter_lib.Formatter().parse(template) if name]


def render(template: str, fill: dict[str, str]) -> str | None:
    names = template_fields(template)
    if any(not fill.get(name) for name in names):
        return None
    return template.format_map({name: fill[name] for name in names})


def visual_for(archetype: Archetype, persona: Persona, rng: random.Random) -> tuple[str, str]:
    pronouns = {"he": persona.he, "his": persona.his, "him": persona.him}
    action = rng.choice(archetype.visuals).format_map(pronouns)
    motion = rng.choice(archetype.motions).format_map(pronouns)
    return f"{persona.subject} {action}, alone in frame", capitalize_first(motion)


def screen_for(archetype: Archetype, template: str, fill: dict[str, str], rng: random.Random) -> str:
    used = set(template_fields(template))
    options = [option for option in archetype.on_screen if render(option, fill)]
    linked = [option for option in options if used & set(template_fields(option))]
    generic = [option for option in options if not template_fields(option)]
    pool = linked or generic or options
    rendered = render(rng.choice(pool), fill) if pool else ""
    return headline(rendered or "")


def headline(text: str) -> str:
    words = normalize(text).rstrip(".").split()
    return capitalize_first(" ".join(words[:MAX_SCREEN_WORDS]).rstrip(",;:-"))


def template_drafts(archetype: Archetype, quota: int, slots: Slots, persona: Persona, rng: random.Random, attempts: int = 6) -> list[Draft]:
    drafts: list[Draft] = []
    seen: set[str] = set()
    templates = list(archetype.templates)
    for _ in range(max(quota * attempts, attempts)):
        if len(drafts) >= quota:
            break
        template = rng.choice(templates)
        fill = slots.choose(rng)
        spoken = render(template, fill)
        if not spoken:
            continue
        spoken = tidy(spoken)
        key = spoken.lower()
        if key in seen:
            continue
        seen.add(key)
        visual, motion = visual_for(archetype, persona, rng)
        drafts.append(Draft(archetype=archetype.name, text=spoken, on_screen=screen_for(archetype, template, fill, rng), visual=visual, motion=motion))
    return drafts


HOOK_SYSTEM = (
    "You are a performance-creative strategist who writes the first three seconds of UGC-style Meta video ads. "
    "Hooks must be spoken by the same on-camera creator as the winning ad, sound like a real man talking to his phone, "
    "and be specific, concrete and native rather than ad-like. Avoid stale openers such as 'I was skeptical', 'this changed my life', "
    "'hear me out' and value lists. Stay compliant with Meta Advertising Standards: never assert or imply the viewer's personal "
    "attributes (health, sexual practices, relationship trouble, age, finances) with 'you/your', keep everything non-explicit, "
    "no guarantees or timeframe promises, no shaming, no clickbait, no engagement bait, no profanity, no emoji, no ALL CAPS."
)


def llm_drafts(
    dna: CreativeDNA,
    brand: Brand,
    allocation: dict[str, int],
    persona: Persona,
    llm: LanguageModel,
    max_words: int,
) -> list[Draft]:
    wanted = {name: count for name, count in allocation.items() if count > 0}
    if not wanted:
        return []
    archetype_lines = "\n".join(f"- {name} x{count}: {ARCHETYPES[name].description}" for name, count in wanted.items())
    item = strict_object(
        {
            "archetype": string(list(wanted)),
            "text": string(),
            "on_screen_text": string(),
            "visual": string(),
            "motion": string(),
            "rationale": string(),
            "score": {"type": "number"},
        }
    )
    prompt = (
        f"Brand: {brand.name} - {brand.category}. Audience: {brand.audience}. Tone: {brand.tone}. CTA: {brand.cta}.\n"
        f"Winning ad angle: {dna.angle}. Promise: {dna.promise}. Audience insight: {dna.audience}.\n"
        f"Original hook (do not paraphrase it): {dna.hook_text}\nBody: {dna.body_text}\nCTA: {dna.cta_text}\n"
        f"Hero on screen: {dna.hero_description}\nPacing: {dna.pacing_wps:.1f} words per second.\n\n"
        f"Write exactly {sum(wanted.values())} new hooks with this archetype allocation:\n{archetype_lines}\n\n"
        f"Rules: spoken text at most {max_words} words and under {MAX_SPOKEN_SECONDS} seconds out loud; on_screen_text at most "
        f"{MAX_SCREEN_WORDS} words that complements, not repeats, the line; visual describes only what {persona.subject.lower()} does in "
        "the first 2-3 seconds (no other faces, no text, no specific location because the location changes per variant); motion is one "
        "camera move suited to image-to-video; rationale is one sentence; score is your 0-1 estimate of thumb-stop strength "
        "(specificity, curiosity, emotional truth, clarity in under 3 seconds)."
    )
    data = ask_json(llm, HOOK_SYSTEM, prompt, strict_object({"hooks": array(item)}), max_tokens=6000)
    drafts: list[Draft] = []
    for raw in (data or {}).get("hooks", []) if isinstance((data or {}).get("hooks"), list) else []:
        if not isinstance(raw, dict):
            continue
        name = normalize_name(str(raw.get("archetype", "")))
        text = tidy(str(raw.get("text", "")))
        if name not in ARCHETYPES or not text:
            continue
        archetype = ARCHETYPES[name]
        fallback_visual, fallback_motion = visual_for(archetype, persona, random.Random(text))
        screen = headline(str(raw.get("on_screen_text", "")))
        drafts.append(
            Draft(
                archetype=name,
                text=text,
                on_screen=screen or headline(text),
                visual=str(raw.get("visual") or "").strip() or fallback_visual,
                motion=str(raw.get("motion") or "").strip() or fallback_motion,
                rationale=str(raw.get("rationale") or "").strip(),
                llm_score=normalize_score(raw.get("score")),
            )
        )
    return drafts


def normalize_score(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    score = float(value)
    if score > 1.0:
        score = score / 10.0 if score <= 10.0 else score / 100.0
    return max(0.0, min(1.0, score))


def heuristic_score(text: str, on_screen: str, pacing_wps: float, warnings: int) -> float:
    words = tokens(text)
    count = len(words)
    if not count:
        return 0.0
    seconds = speech_seconds(text, pacing_wps)
    length = max(0.0, 1.0 - abs(count - 8) / 8)
    duration = 1.0 if seconds <= 2.8 else max(0.0, 1.0 - (seconds - 2.8) / 1.2)
    has_number = any(any(ch.isdigit() for ch in w) for w in words) or bool(set(words) & {"two", "three", "five", "six", "nine", "ten", "thirty", "one"})
    concrete_hits = sum(1 for w in words if w in CONCRETE)
    specificity = min(1.0, 0.4 * has_number + 0.2 * concrete_hits + 0.2 * bool(set(words) & {"wife", "girlfriend", "partner", "husband", "she", "her"}))
    curiosity = min(1.0, 0.35 * bool(OPEN_LOOP.search(text)) + 0.3 * text.strip().endswith("?") + 0.35 * bool(CONTRAST.search(text)))
    abstract_hits = sum(1 for w in words if w in ABSTRACT)
    concreteness = max(0.0, min(1.0, 0.5 + 0.15 * concrete_hits - 0.25 * abstract_hits))
    first = words[0]
    opener = 1.0 if first in STRONG_OPENERS or first.isdigit() else 0.3 if first in WEAK_OPENERS else 0.65
    readability = max(0.0, min(1.0, 1.0 - max(0.0, total_syllables(text) / count - 1.3)))
    screen_words = word_count(on_screen)
    screen = 1.0 if 2 <= screen_words <= MAX_SCREEN_WORDS else 0.5
    stale = sum(0.15 for phrase in STALE if phrase in text.lower())
    score = (
        0.13 * length
        + 0.14 * duration
        + 0.17 * specificity
        + 0.17 * curiosity
        + 0.11 * concreteness
        + 0.1 * opener
        + 0.1 * readability
        + 0.08 * screen
        - stale
        - 0.06 * warnings
    )
    return round(max(0.0, min(1.0, score)), 3)


def fits(text: str, on_screen: str, pacing_wps: float) -> str | None:
    if word_count(text) > MAX_SPOKEN_WORDS:
        return f"spoken hook has {word_count(text)} words (max {MAX_SPOKEN_WORDS})"
    seconds = speech_seconds(text, pacing_wps)
    if seconds > MAX_SPOKEN_SECONDS:
        return f"spoken hook needs about {seconds:.1f}s (max {MAX_SPOKEN_SECONDS}s)"
    if word_count(on_screen) > MAX_SCREEN_WORDS or not on_screen.strip():
        return f"on-screen text must be 1-{MAX_SCREEN_WORDS} words"
    return None


def guard(draft: Draft, brand: Brand, llm: LanguageModel | None) -> bool:
    text_flags = policy.lint(draft.text, brand)
    if text_flags:
        fixed = policy.repair(draft.text, text_flags, brand, llm)
        if fixed is None:
            draft.flags = text_flags
            return False
        if fixed != draft.text:
            draft.notes.append(f"rewritten for policy ({', '.join(f.rule for f in text_flags)}) from: {draft.text}")
            draft.text = fixed
    screen_flags = policy.lint(draft.on_screen, brand)
    if screen_flags:
        fixed_screen = policy.repair(draft.on_screen, screen_flags, brand, llm)
        draft.on_screen = headline(fixed_screen) if fixed_screen else headline(draft.text)
    draft.flags = policy.merge(policy.lint(draft.text, brand), policy.lint(draft.on_screen, brand))
    return not policy.blocking(draft.flags)


def quality_flag(reason: str) -> PolicyFlag:
    return PolicyFlag(rule="hook_quality", severity="block", span="", message=reason, suggestion="Shorten the line so it lands inside the first 3 seconds.")


def rationale_for(draft: Draft, dna: CreativeDNA, priors: Priors) -> str:
    archetype = ARCHETYPES[draft.archetype]
    arm = priors.arm("archetype", draft.archetype)
    evidence = (
        f"prior win rate {arm.mean:.0%} over {arm.trials} tests"
        if arm.trials
        else "no live results yet, kept in the mix for exploration"
    )
    base = draft.rationale or f"{archetype.label}: {archetype.why}"
    angle = dna.angle.split(":")[0].strip().lower() or "core"
    parts = [base.rstrip("."), f"keeps the winner's {angle} angle", evidence]
    note = f" Note: {draft.notes[-1].rstrip('.')}." if draft.notes else ""
    return "; ".join(parts) + "." + note


def dedupe(drafts: list[Draft], original: str) -> list[Draft]:
    kept: list[Draft] = []
    for draft in sorted(drafts, key=lambda d: -d.score):
        if original and jaccard(draft.text, original) > DUPLICATE_JACCARD:
            continue
        if any(jaccard(draft.text, other.text) > DUPLICATE_JACCARD for other in kept):
            continue
        kept.append(draft)
    return kept


def similarity(left: Draft, right: Draft) -> float:
    return min(1.0, jaccard(left.text, right.text) + (SAME_ARCHETYPE_PENALTY if left.archetype == right.archetype else 0.0))


def marginal_relevance(draft: Draft, chosen: list[Draft]) -> float:
    redundancy = max((similarity(draft, other) for other in chosen), default=0.0)
    return MMR_LAMBDA * draft.score - (1 - MMR_LAMBDA) * redundancy


def select(pool: list[Draft], count: int) -> list[Draft]:
    chosen: list[Draft] = []
    remaining = list(pool)
    while remaining and len(chosen) < count:
        best = max(remaining, key=lambda draft: marginal_relevance(draft, chosen))
        chosen.append(best)
        remaining.remove(best)
    return chosen


def unique_by_text(drafts: list[Draft]) -> list[Draft]:
    return list({draft.text.lower(): draft for draft in reversed(drafts)}.values())[::-1]


def to_hook(draft: Draft, hook_id: str, dna: CreativeDNA, priors: Priors) -> Hook:
    return Hook(
        id=hook_id,
        archetype=draft.archetype,
        text=draft.text,
        on_screen_text=draft.on_screen,
        visual=draft.visual,
        motion=draft.motion,
        rationale=rationale_for(draft, dna, priors),
        score=round(draft.score, 3),
        flags=draft.flags,
    )


def score_draft(draft: Draft, pacing_wps: float) -> float:
    warnings = sum(1 for flag in draft.flags if flag.severity == "warn")
    heuristic = heuristic_score(draft.text, draft.on_screen, pacing_wps, warnings)
    if draft.llm_score is None:
        return heuristic
    return round(LLM_WEIGHT * draft.llm_score + (1 - LLM_WEIGHT) * heuristic, 3)


def gather(
    dna: CreativeDNA,
    brand: Brand,
    allocation: dict[str, int],
    llm: LanguageModel | None,
    persona: Persona,
    slots: Slots,
    rng: random.Random,
) -> list[Draft]:
    drafts = llm_drafts(dna, brand, allocation, persona, llm, MAX_SPOKEN_WORDS) if llm is not None else []
    produced = {name: sum(1 for d in drafts if d.archetype == name) for name in allocation}
    for name, quota in allocation.items():
        missing = quota - produced[name]
        if missing > 0:
            drafts.extend(template_drafts(ARCHETYPES[name], missing, slots, persona, rng))
    return drafts


def evaluate(drafts: list[Draft], dna: CreativeDNA, brand: Brand, llm: LanguageModel | None) -> tuple[list[Draft], list[Draft]]:
    accepted: list[Draft] = []
    rejected: list[Draft] = []
    for draft in drafts:
        if not guard(draft, brand, llm):
            rejected.append(draft)
            continue
        problem = fits(draft.text, draft.on_screen, dna.pacing_wps)
        if problem:
            draft.flags = [*draft.flags, quality_flag(problem)]
            rejected.append(draft)
            continue
        draft.score = score_draft(draft, dna.pacing_wps)
        accepted.append(draft)
    return accepted, rejected


def llm_screen(pool: list[Draft], count: int, brand: Brand, llm: LanguageModel | None) -> tuple[list[Draft], list[Draft]]:
    if llm is None or not pool:
        return pool, []
    shortlist = sorted(pool, key=lambda d: -d.score)[: max(count * 2, count)]
    reviews = policy.review_many([f"{d.text} | on-screen: {d.on_screen}" for d in shortlist], brand, llm)
    blocked: list[Draft] = []
    for draft, flags in zip(shortlist, reviews, strict=True):
        if policy.blocking(flags):
            draft.flags = policy.merge(draft.flags, flags)
            blocked.append(draft)
    blocked_ids = {id(draft) for draft in blocked}
    return [d for d in pool if id(d) not in blocked_ids], blocked


def generate_hooks(
    dna: CreativeDNA,
    count: int,
    brand: Brand,
    priors: Priors,
    llm: LanguageModel | None = None,
    archetypes: list[str] | None = None,
    seed: int = 7,
) -> tuple[list[Hook], list[Hook]]:
    if count <= 0:
        return [], []
    rng = random.Random(seed)
    names = resolve_archetypes(archetypes)
    angle = angle_for(f"{dna.angle} {dna.hook_text} {dna.body_text}")
    persona = hero_persona(dna.hero_description)
    slots = build_slots(dna, angle, rng)
    allocation = bandit.allocate(rng, priors, "archetype", names, 3 * count)
    pool: list[Draft] = []
    rejected: list[Draft] = []
    round_allocation = allocation
    for attempt in range(3):
        drafts = gather(dna, brand, round_allocation, llm if attempt == 0 else None, persona, slots, rng)
        accepted, blocked = evaluate(drafts, dna, brand, llm)
        rejected.extend(blocked)
        pool = dedupe(pool + accepted, dna.hook_text)
        if len(pool) >= count:
            break
        round_allocation = {name: max(1, allocation.get(name, 0)) + attempt + 1 for name in names}
    pool, screened_out = llm_screen(pool, count, brand, llm)
    rejected.extend(screened_out)
    chosen = select(pool, count)
    hooks = [to_hook(draft, f"H{index:02d}", dna, priors) for index, draft in enumerate(chosen, start=1)]
    blocked_hooks = [to_hook(draft, f"R{index:02d}", dna, priors) for index, draft in enumerate(unique_by_text(rejected), start=1)]
    log.info("hooks: %d accepted from %d candidates, %d rejected", len(hooks), len(pool) + len(rejected), len(blocked_hooks))
    return hooks, blocked_hooks
