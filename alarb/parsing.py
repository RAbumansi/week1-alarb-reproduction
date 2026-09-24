"""Pull structured answers out of raw model output.

The paper's templates close their sections with a backslash, [\\VERDICT], which
is unusual enough that models routinely emit [/VERDICT] instead, or drop the
closing tag altogether. Reasoning models add <think> blocks the templates never
asked for. Parsing therefore accepts all of these rather than scoring a correct
answer as wrong over a punctuation mark.

A failure to parse is reported as such, never silently converted into a wrong
answer: the rate itself says something about how well a model follows the
paper's output format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_THINKING = re.compile(r"<(think|thinking)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_OPEN_TAG = "[{name}]"
_CLOSE_TAGS = ("[\\{name}]", "[/{name}]")
_MCQ_LETTER = re.compile(r"\b([ABCD])\b")


@dataclass(frozen=True)
class ParsedVerdict:
    verdict: str | None
    reasoning: str | None

    @property
    def ok(self) -> bool:
        return bool(self.verdict)


def strip_thinking(text: str) -> str:
    return _THINKING.sub("", text).strip()


def _trim(value: str) -> str:
    value = value.strip()
    # Templates show the payload as "Your verdict here", quotes included, so
    # models frequently wrap the real answer the same way.
    if len(value) >= 2 and value[0] in "\"'“”" and value[-1] in "\"'“”":
        value = value[1:-1].strip()
    return value


def extract_section(text: str, name: str) -> str | None:
    """Content of [NAME]...[\\NAME], tolerating [/NAME], a missing close, or neither."""
    opening = _OPEN_TAG.format(name=name)
    start = text.find(opening)
    if start == -1:
        return None
    start += len(opening)

    ends = [
        position
        for position in (text.find(tag.format(name=name), start) for tag in _CLOSE_TAGS)
        if position != -1
    ]
    # No closing tag: stop at whatever section starts next, else run to the end.
    if not ends:
        following = re.search(r"\n\s*\[[\\/]?[A-Z]+\]", text[start:])
        ends = [start + following.start()] if following else [len(text)]
    section = _trim(text[start : min(ends)])
    return section or None


def parse_verdict(text: str) -> ParsedVerdict:
    cleaned = strip_thinking(text)
    verdict = extract_section(cleaned, "VERDICT")
    reasoning = extract_section(cleaned, "REASONING")
    if verdict is None and reasoning is None and cleaned:
        # Some models answer with a bare verdict and no tags at all. Only treat
        # the whole output that way when it is short enough to be one.
        collapsed = _trim(cleaned)
        if len(collapsed.split()) <= 60:
            verdict = collapsed
    return ParsedVerdict(verdict=verdict, reasoning=reasoning)


def parse_mcq_answer(text: str) -> str | None:
    cleaned = strip_thinking(text)
    section = extract_section(cleaned, "ANSWER")
    for candidate in (section, cleaned):
        if not candidate:
            continue
        match = _MCQ_LETTER.search(candidate)
        if match:
            return match.group(1)
    return None


def parse_judgement(text: str) -> str | None:
    """The judge's label, from [EVALUATION] or from the raw text as a fallback."""
    cleaned = strip_thinking(text)
    section = extract_section(cleaned, "EVALUATION") or cleaned
    upper = section.upper()
    # "PARTIALLY CORRECT" contains "CORRECT", so it has to be tested first.
    for label in ("PARTIALLY CORRECT", "INCORRECT", "CORRECT"):
        if label in upper:
            return label
    return None
