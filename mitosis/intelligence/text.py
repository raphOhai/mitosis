from __future__ import annotations

import re
import unicodedata

WORD = re.compile(r"[A-Za-z0-9#%]+(?:'[A-Za-z]+)*")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
TERMINAL = re.compile(r"[.!?][\"')\]]*$")
VOWEL_RUN = re.compile(r"[aeiouy]+")
CLAUSE_BREAK = re.compile(r"[,;:\u2013\u2014-]$")
QUOTES = {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00a0": " "}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    for source, target in QUOTES.items():
        text = text.replace(source, target)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD.finditer(normalize(text))]


def word_count(text: str) -> int:
    return len([part for part in normalize(text).split() if any(ch.isalnum() for ch in part)])


def jaccard(left: str, right: str) -> float:
    a, b = set(tokens(left)), set(tokens(right))
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_SPLIT.split(normalize(text)) if part.strip()]


def ends_sentence(word: str) -> bool:
    return bool(TERMINAL.search(word.strip()))


def ends_clause(word: str) -> bool:
    return bool(CLAUSE_BREAK.search(word.strip()))


def syllables(word: str) -> int:
    word = word.lower().strip("'")
    if not word:
        return 0
    if word.isdigit():
        return max(1, round(len(word) * 1.6))
    runs = VOWEL_RUN.findall(word)
    count = len(runs)
    if word.endswith("e") and not word.endswith(("le", "ee", "ye")) and count > 1:
        count -= 1
    return max(1, count)


def total_syllables(text: str) -> int:
    return sum(syllables(token) for token in tokens(text))


def speech_seconds(text: str, words_per_second: float = 2.6) -> float:
    rate = 4.8 * max(0.85, min(1.4, words_per_second / 2.6))
    pauses = len(re.findall(r"[,;:.!?]\s+\S", normalize(text)))
    return total_syllables(text) / rate + 0.15 * pauses


def capitalize_first(text: str) -> str:
    for index, char in enumerate(text):
        if char.isspace() or char in "\"'([":
            continue
        return text[:index] + char.upper() + text[index + 1 :] if char.isalpha() else text
    return text


def tidy(text: str) -> str:
    text = normalize(text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:])(?=[.!?])", "", text)
    text = re.sub(r"([,;:]){2,}", r"\1", text)
    text = re.sub(r"^[\s,;:.!?-]+", "", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return " ".join(capitalize_first(sentence) for sentence in split_sentences(text))


def article(phrase: str) -> str:
    lowered = phrase.lower()
    if lowered.startswith(("one", "uni", "use", "usu", "eu")):
        return "a"
    if lowered.startswith(("hour", "honest", "honor")) or lowered[:1] in "aeiou":
        return "an"
    return "a"


def clip_words(text: str, limit: int) -> str:
    words = normalize(text).split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).rstrip(",;:-")
