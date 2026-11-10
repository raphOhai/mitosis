from __future__ import annotations

import re

from pydantic import Field

from mitosis.intelligence.text import tokens
from mitosis.models import Model


class NumberLine(Model):
    spoken: str
    screen: str
    values: list[int]


class Angle(Model):
    key: str
    label: str
    keywords: list[str]
    promise: str
    problems: list[str]
    stopped: list[str]
    kept: list[str]
    used_to: list[str]
    shared: list[str]
    myths: list[str]
    habits: list[str]
    outcomes: list[str]
    quotes: list[str]
    beliefs: list[str]
    scenes: list[str]
    numbers: list[NumberLine]
    us: list[str]
    them: list[str]
    beats: list[str] = Field(default_factory=list)


ANGLES: dict[str, Angle] = {
    "reconnection": Angle(
        key="reconnection",
        label="Reconnection: from roommates back to partners",
        keywords=["roommate", "roommates", "distance", "drift", "drifted", "disconnected", "apart", "lonely", "closer", "connection", "connected", "reconnect", "strangers", "parallel", "team"],
        promise="feel like partners again instead of roommates, with a few guided minutes a day",
        problems=["we'd become roommates", "we were living parallel lives", "we only talked about logistics", "we felt like polite strangers", "we stopped feeling like a team"],
        stopped=["asking about her day", "sitting next to her on purpose", "texting her just because", "looking up when she walked in"],
        kept=["checking my phone mid-conversation", "saying we were fine", "waiting for her to make the first move"],
        used_to=["sit next to her on purpose", "text her for no reason", "make her laugh every day"],
        shared=["laugh at dinner", "go on a walk, just us", "talk about anything but schedules"],
        myths=["being busy was normal", "drifting apart was just marriage", "we'd reconnect on vacation"],
        habits=["two-minute check-in", "six-second kiss", "high-low question", "phone-free dinner"],
        outcomes=["we talk at dinner again", "she sits closer", "it feels like us again", "we laugh again"],
        quotes=["I miss you, even when you're right here.", "When did we stop being us?", "It feels like we're roommates."],
        beliefs=["Date nights didn't fix us.", "A vacation didn't fix us.", "More time together didn't fix us."],
        scenes=["Sunday night, same couch, two phones, zero words.", "Dinner's done, she's on her phone, I'm on mine.", "Kids asleep, TV on, and we're both scrolling."],
        numbers=[
            NumberLine(spoken="We went {n} days without a real conversation.", screen="{n} days, zero real talks", values=[17, 23, 41, 63]),
            NumberLine(spoken="{n} days. That's how long since we really talked.", screen="{n} days without really talking", values=[19, 31, 44]),
        ],
        us=["check in daily", "talk every night", "put phones down"],
        them=["wait for vacations", "wait for date night", "assume they're fine"],
        beats=["names the roommate feeling", "admits the drift was gradual", "introduces a small daily ritual", "shows the first sign of closeness returning"],
    ),
    "communication": Angle(
        key="communication",
        label="Communication: talking about more than logistics",
        keywords=["talk", "talking", "talked", "conversation", "conversations", "listen", "listening", "heard", "communicate", "communication", "question", "questions", "words", "ask", "asked"],
        promise="have real conversations again with one guided prompt a day",
        problems=["every conversation was about logistics", "I answered everything with fine", "I was listening just to reply", "we only talked about the kids"],
        stopped=["asking her real questions", "putting my phone face down", "listening without fixing", "asking a second question"],
        kept=["fixing instead of listening", "answering with one word", "listening just to reply"],
        used_to=["ask her real questions", "listen without fixing", "talk with her for hours"],
        shared=["talk for an hour straight", "talk about anything but the kids", "ask each other something new"],
        myths=["fine meant fine", "talking about the schedule counted", "she wanted me to fix it"],
        habits=["better dinner question", "ten-minute walk-and-talk", "two-question check-in", "phone-down hour"],
        outcomes=["she tells me things first", "dinner lasts an hour", "we talk before bed"],
        quotes=["You listen to your phone more than me.", "You never ask how I'm really doing.", "I feel like I'm talking to a wall."],
        beliefs=["Talking more didn't fix us.", "Communication tips didn't fix us.", "Fixing her problems didn't help."],
        scenes=["She asked how my day was. I said fine. Again.", "Another dinner where we only talked about schedules."],
        numbers=[
            NumberLine(spoken="We tracked it: {n} minutes of real talk a week.", screen="{n} minutes of real talk", values=[9, 12, 14]),
            NumberLine(spoken="{n} minutes a week. That was our real conversation.", screen="{n} minutes a week", values=[8, 11, 15]),
        ],
        us=["ask real questions", "listen first", "walk and talk"],
        them=["talk schedules", "fix instead of listen", "answer with fine"],
        beats=["shows the one-word-answer pattern", "admits listening to reply", "introduces one better question", "shows a longer, warmer conversation"],
    ),
    "spark": Angle(
        key="spark",
        label="Spark: bringing back flirting and closeness",
        keywords=["spark", "flirt", "flirting", "passion", "attraction", "desire", "romance", "romantic", "chemistry", "intimacy", "intimate", "touch", "kiss", "affection", "wanted", "date"],
        promise="bring back the flirting and closeness with small daily moves that feel natural",
        problems=["we stopped flirting", "affection became a goodnight peck", "romance turned into routine", "I stopped making her feel wanted"],
        stopped=["flirting with her", "kissing her like I meant it", "planning little surprises", "complimenting her out loud"],
        kept=["waiting for the right mood", "saving romance for anniversaries", "treating her like a roommate"],
        used_to=["flirt with her", "plan little surprises", "kiss her like I meant it"],
        shared=["flirt over text", "go on a real date", "kiss for longer than a second"],
        myths=["the spark just fades", "romance needed a big gesture", "flirting was for dating"],
        habits=["six-second kiss", "daily compliment", "two-minute flirt text", "real hello hug"],
        outcomes=["she flirts back", "she texts me first", "it feels like dating again"],
        quotes=["When did we stop flirting?", "I miss how you used to look at me.", "We feel more like co-workers."],
        beliefs=["Grand gestures didn't bring the spark back.", "Expensive dates didn't fix it.", "Waiting for the mood didn't work."],
        scenes=["Friday night, kids down, and we're both asleep by nine.", "Goodnight peck, lights off, same as every night."],
        numbers=[
            NumberLine(spoken="Six seconds. That's how long our goodbye kiss is now.", screen="the six-second rule", values=[6]),
            NumberLine(spoken="{n} days since I last flirted with my own {partner}.", screen="{n} days without flirting", values=[30, 45, 90]),
        ],
        us=["flirt every day", "kiss for six seconds", "plan tiny surprises"],
        them=["wait for the mood", "save it for anniversaries", "let it fade"],
        beats=["names the goodnight-peck routine", "admits romance became routine", "introduces a tiny flirting habit", "shows her flirting back"],
    ),
    "appreciation": Angle(
        key="appreciation",
        label="Appreciation: noticing and saying it out loud",
        keywords=["appreciate", "appreciation", "appreciated", "notice", "noticed", "noticing", "thank", "thanks", "grateful", "gratitude", "invisible", "unseen", "compliment", "praise", "effort", "seen"],
        promise="make your partner feel seen with one small, specific habit a day",
        problems=["I stopped noticing what she does", "I took her effort for granted", "I thanked the barista more than her"],
        stopped=["thanking her for small things", "noticing the little stuff", "saying it out loud"],
        kept=["taking her for granted", "noticing only what went wrong", "saving thanks for birthdays"],
        used_to=["notice the little things", "say thank you out loud", "brag about her to friends"],
        shared=["say thank you for nothing special", "notice each other", "celebrate a normal Tuesday"],
        myths=["she knew I was grateful", "thanks was implied", "gifts said it for me"],
        habits=["specific thank-you", "thirty-second appreciation text", "I-noticed moment", "out-loud compliment"],
        outcomes=["she lights up", "she notices me back", "the house feels lighter"],
        quotes=["Do you even see what I do?", "It would be nice to hear thank you.", "I feel invisible around here."],
        beliefs=["Flowers didn't make her feel seen.", "Big gifts didn't fix it.", "Saying love you didn't cover it."],
        scenes=["She cleaned the whole kitchen. I didn't say a word.", "She handled bedtime again, and I kept scrolling."],
        numbers=[
            NumberLine(spoken="Thirty seconds a day. That's all I changed.", screen="thirty seconds a day", values=[30]),
            NumberLine(spoken="I counted {n} thank-yous to strangers and zero to her.", screen="{n} thank-yous, zero to her", values=[7, 11, 14]),
        ],
        us=["say thank you daily", "notice small things", "say it out loud"],
        them=["save it for birthdays", "assume she knows", "notice only mistakes"],
        beats=["shows unnoticed effort", "admits taking her for granted", "introduces a specific thank-you", "shows her lighting up"],
    ),
    "conflict": Angle(
        key="conflict",
        label="Conflict: fighting less and repairing faster",
        keywords=["fight", "fights", "fighting", "argue", "arguing", "argument", "arguments", "conflict", "yelling", "defensive", "tension", "resent", "resentment", "silent", "sorry", "apologize", "angry"],
        promise="turn the same old fight into a five-minute repair",
        problems=["we kept having the same fight", "every talk turned into a fight", "I got defensive every time", "we went to bed angry"],
        stopped=["letting the small stuff go", "apologizing first", "hearing her out"],
        kept=["getting defensive", "keeping score", "going quiet for days"],
        used_to=["let the small stuff go", "apologize first", "laugh off a bad day"],
        shared=["make up in minutes", "fight without yelling", "laugh after an argument"],
        myths=["winning the argument mattered", "silence would cool things down", "good couples never fight"],
        habits=["five-minute repair talk", "one-sentence apology", "ten-minute cool-down walk", "softer first sentence"],
        outcomes=["fights end in minutes", "we laugh after arguments", "we never go to bed angry"],
        quotes=["You always have to win.", "Can we not do this again?", "I don't want to fight. I want you to listen."],
        beliefs=["Winning arguments didn't help us.", "Avoiding fights didn't fix us.", "Being right didn't fix anything."],
        scenes=["Same argument, third time this week, different room.", "Door closed, silent treatment, day two."],
        numbers=[
            NumberLine(spoken="Same fight, {n} times this month. I counted.", screen="same fight, {n} times", values=[4, 6, 9]),
            NumberLine(spoken="{n} minutes. That's how long our fights last now.", screen="{n}-minute fights now", values=[5, 10]),
        ],
        us=["repair in minutes", "start soft", "take a timeout"],
        them=["fight to win", "keep score", "go silent for days"],
        beats=["shows the repeating fight", "admits defensiveness", "introduces a short repair ritual", "shows a faster make-up"],
    ),
    "busy_parents": Angle(
        key="busy_parents",
        label="Busy parents: staying a couple, not just co-parents",
        keywords=["kids", "kid", "baby", "toddler", "parents", "parenting", "busy", "tired", "exhausted", "schedule", "schedules", "work", "routine", "bedtime", "co-parents", "mom", "dad"],
        promise="stay a couple, not just co-parents, in five minutes a day",
        problems=["we became co-parents, not partners", "every talk was about the kids", "we were too tired for us"],
        stopped=["being her boyfriend after the kids came", "planning anything just for us", "asking about her, not the kids"],
        kept=["saying we'd have time later", "putting us last", "choosing my phone at night"],
        used_to=["plan nights just for us", "ask about her, not the kids", "be her boyfriend, not just dad"],
        shared=["talk about something besides the kids", "have a night just for us", "sit together without a screen"],
        myths=["we'd have time when the kids got older", "being good parents was enough", "tired was just our life now"],
        habits=["five-minute couch check-in", "Sunday planning date", "after-bedtime cuddle", "kid-free question"],
        outcomes=["we feel like a team", "we're a couple again", "we look forward to nights"],
        quotes=["We only talk about the kids.", "I miss being your girlfriend, not just mom.", "When did we stop being a couple?"],
        beliefs=["Waiting for the kids to grow up didn't work.", "More babysitters didn't fix us.", "Date night once a month didn't fix us."],
        scenes=["Kids finally asleep, and we're both too tired to talk.", "Bedtime routine done. Two phones out. Again."],
        numbers=[
            NumberLine(spoken="Five minutes after bedtime. That's our whole system.", screen="our five-minute system", values=[5]),
            NumberLine(spoken="{n} nights in a row of only talking about the kids.", screen="{n} nights of kid talk", values=[14, 21, 30]),
        ],
        us=["take five minutes nightly", "plan one night a week", "talk about us first"],
        them=["wait for the kids to grow", "only talk logistics", "put themselves last"],
        beats=["shows the exhausted after-bedtime routine", "admits they became co-parents", "introduces a five-minute ritual", "shows them as a team again"],
    ),
}

DEFAULT_ANGLE = "reconnection"


def classify_angle(text: str) -> str:
    words = tokens(text)
    counts = {key: sum(words.count(keyword) for keyword in angle.keywords) for key, angle in ANGLES.items()}
    best = max(counts.items(), key=lambda item: (item[1], item[0] == DEFAULT_ANGLE))
    return best[0] if best[1] > 0 else DEFAULT_ANGLE


def angle_for(label_or_text: str) -> Angle:
    lowered = label_or_text.lower()
    for key, angle in ANGLES.items():
        if lowered.startswith(key.replace("_", " ")) or lowered.startswith(angle.label.lower().split(":")[0]):
            return angle
    return ANGLES[classify_angle(label_or_text)]


def partner_word(text: str) -> str:
    lowered = text.lower()
    for word, pattern in (("wife", r"\b(?:wife|married|marriage|husband)\b"), ("girlfriend", r"\bgirlfriend\b"), ("boyfriend", r"\bboyfriend\b")):
        if re.search(pattern, lowered):
            return "husband" if word == "wife" and re.search(r"\bmy husband\b", lowered) else word
    return "partner"
