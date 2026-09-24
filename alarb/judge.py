"""Grade generated verdicts against the court's own.

The paper grades with GPT-4o as an LLM-as-a-judge using the prompt in Appendix
B.1, labelling each prediction CORRECT, PARTIALLY CORRECT or INCORRECT.
AnthropicJudge runs that prompt unchanged against a Claude model instead, which
is the substitution forced by the available credentials.

HeuristicJudge is not that metric and must not be reported as it. It compares
wording, not meaning, and exists so the pipeline can be exercised without an API
key and so the LLM judge can be measured against something -- if the two agree
almost everywhere, the LLM judge is adding less than it appears to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from alarb import prompts
from alarb.parsing import parse_judgement
from alarb.providers import Completion, Model


LABELS = ("CORRECT", "PARTIALLY CORRECT", "INCORRECT")
UNPARSEABLE = "UNPARSEABLE"

_STOPWORDS = {"في", "من", "على", "الى", "إلى", "عن", "مع", "ان", "أن", "التي", "الذي", "و", "ما"}


@dataclass(frozen=True)
class Judgement:
    label: str
    judge: str
    raw: str = ""


class Judge:
    name: str

    def judge(self, reference: str, prediction: str) -> Judgement:
        raise NotImplementedError


class AnthropicJudge(Judge):
    def __init__(self, model: Model) -> None:
        self.name = f"anthropic:{model.name}"
        self._model = model

    def judge(self, reference: str, prediction: str) -> Judgement:
        prompt = prompts.JUDGE.format(judge_verdict=reference, predicted_verdict=prediction)
        completion: Completion = self._model.complete(prompt)
        label = parse_judgement(completion.text)
        return Judgement(
            label=label or UNPARSEABLE, judge=self.name, raw=completion.text
        )


def _tokens(text: str) -> set[str]:
    normalized = re.sub(r"[أإآٱ]", "ا", re.sub(r"[^\w\s]", " ", text))
    return {
        token
        for token in normalized.split()
        if len(token) > 2 and token not in _STOPWORDS
    }


class HeuristicJudge(Judge):
    """Word overlap with the reference verdict. A plumbing check, not the metric."""

    def __init__(self, correct_at: float = 0.6, partial_at: float = 0.3) -> None:
        self.name = "heuristic:token-overlap"
        self._correct_at = correct_at
        self._partial_at = partial_at

    def judge(self, reference: str, prediction: str) -> Judgement:
        expected = _tokens(reference)
        if not expected:
            return Judgement(label=UNPARSEABLE, judge=self.name)
        overlap = len(expected & _tokens(prediction)) / len(expected)
        if overlap >= self._correct_at:
            label = "CORRECT"
        elif overlap >= self._partial_at:
            label = "PARTIALLY CORRECT"
        else:
            label = "INCORRECT"
        return Judgement(label=label, judge=self.name, raw=f"overlap={overlap:.3f}")


def get_judge(name: str, model: Model | None = None) -> Judge:
    if name == "heuristic":
        return HeuristicJudge()
    if model is None:
        raise ValueError(f"Judge {name!r} needs a model to call.")
    return AnthropicJudge(model)
