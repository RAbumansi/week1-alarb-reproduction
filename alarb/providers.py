"""Models behind a single interface, with every call cached on disk.

Caching is content-addressed on model, prompt and decoding parameters, so an
interrupted run resumes for nothing and re-scoring never re-pays for inference.

The two offline models are reference points rather than substitutes for a real
one. ConstantModel answers every case with the same generic verdict, which
measures how much credit a content-free answer collects from the judge --
59.1% of cases in this dataset end in an order against the defendant, so a
model must clearly beat that to have shown anything. RandomChoiceModel is the
25% floor for the four-way multiple-choice tasks.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

from alarb.tasks import CHOICE_LETTERS


CACHE_ROOT = Path(".cache/llm")
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0

# A generic ruling for the majority outcome, in the register the dataset uses.
MAJORITY_CLASS_VERDICT = "إلزام المدعى عليه بسداد المبلغ المطالب به للمدعي."


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cached: bool = False


def estimate_tokens(text: str) -> int:
    """Rough pre-flight count. Arabic runs near three characters per token."""
    return max(1, round(len(text) / 3.0))


class CompletionCache:
    def __init__(self, root: Path = CACHE_ROOT) -> None:
        self.root = root

    def key(self, model: str, prompt: str, params: dict) -> str:
        payload = json.dumps(
            {"model": model, "prompt": prompt, "params": params},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Completion | None:
        path = self._path(key)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return Completion(**payload, cached=True)

    def put(self, key: str, completion: Completion) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "text": completion.text,
                    "model": completion.model,
                    "input_tokens": completion.input_tokens,
                    "output_tokens": completion.output_tokens,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


class Model:
    name: str

    def complete(self, prompt: str) -> Completion:
        raise NotImplementedError


class AnthropicModel(Model):
    def __init__(
        self,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        cache: CompletionCache | None = None,
    ) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set, so no model can be called. Export a "
                "key, or use --model constant / --model random for the offline "
                "reference points."
            )
        try:
            import anthropic
        except ImportError as error:
            raise RuntimeError("pip install anthropic") from error

        self.name = model
        self.params = {"max_tokens": max_tokens, "temperature": temperature}
        self.cache = cache or CompletionCache()
        self._client = anthropic.Anthropic()

    def complete(self, prompt: str) -> Completion:
        key = self.cache.key(self.name, prompt, self.params)
        hit = self.cache.get(key)
        if hit is not None:
            return hit

        response = self._client.messages.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            **self.params,
        )
        completion = Completion(
            text="".join(block.text for block in response.content if block.type == "text"),
            model=self.name,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        self.cache.put(key, completion)
        return completion


class ConstantModel(Model):
    """Answers every verdict task with the same generic ruling."""

    def __init__(self, verdict: str = MAJORITY_CLASS_VERDICT) -> None:
        self.name = "baseline:constant"
        self._verdict = verdict

    def complete(self, prompt: str) -> Completion:
        text = f"[VERDICT]\n{self._verdict}\n[\\VERDICT]"
        return Completion(
            text=text,
            model=self.name,
            input_tokens=estimate_tokens(prompt),
            output_tokens=estimate_tokens(text),
        )


class RandomChoiceModel(Model):
    """Picks a multiple-choice letter at random; the 25% floor."""

    def __init__(self, seed: int = 42) -> None:
        self.name = "baseline:random"
        self._seed = seed

    def complete(self, prompt: str) -> Completion:
        digest = hashlib.sha256(f"{self._seed}:{prompt}".encode("utf-8")).hexdigest()
        letter = CHOICE_LETTERS[int(digest, 16) % len(CHOICE_LETTERS)]
        text = f"[ANSWER]\n{letter}\n[\\ANSWER]"
        return Completion(
            text=text,
            model=self.name,
            input_tokens=estimate_tokens(prompt),
            output_tokens=estimate_tokens(text),
        )


def get_model(
    name: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Model:
    if name == "constant":
        return ConstantModel()
    if name == "random":
        return RandomChoiceModel()
    return AnthropicModel(name, max_tokens=max_tokens, temperature=temperature)
