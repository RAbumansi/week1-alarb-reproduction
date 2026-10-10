"""Record what produced a run, so incomparable runs cannot be compared silently.

A run's scores depend on more than its model and decoding parameters. Rebuild
the article corpus, edit a prompt, or repin the dataset revision, and the
numbers shift for reasons nothing in the output would show. Each run therefore
stores a fingerprint of its inputs, and the report refuses to present runs with
different fingerprints as one table without saying so.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from alarb import prompts
from alarb.corpus import CORPUS_PATH


SPLIT_MANIFEST = Path("artifacts/split_manifest.json")

TASK_TEMPLATES = {
    "verdict_facts": "VERDICT_FROM_FACTS",
    "verdict_laws": "VERDICT_FROM_FACTS_AND_LAWS",
    "verdict_reasoning": "VERDICT_FROM_FACTS_AND_REASONING",
    "argument_completion": "ARGUMENT_COMPLETION",
    "mcq_same_statute": "ARTICLE_MCQ",
    "mcq_semantic": "ARTICLE_MCQ",
}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _file_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return _digest(path.read_text(encoding="utf-8"))


def prompt_digest(task: str) -> str | None:
    name = TASK_TEMPLATES.get(task)
    if name is None:
        return None
    return _digest(getattr(prompts, name))


def dataset_revision() -> str | None:
    if not SPLIT_MANIFEST.is_file():
        return None
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    return manifest.get("source", {}).get("revision")


def harness_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def current(task: str) -> dict:
    return {
        "dataset_revision": dataset_revision(),
        "corpus_digest": _file_digest(CORPUS_PATH),
        "prompt_digest": prompt_digest(task),
        "harness_commit": harness_commit(),
    }


def comparable(runs: list[dict]) -> tuple[bool, list[str], list[str]]:
    """Whether these runs share the inputs that make their scores mean the same
    thing. Returns (ok, differing keys, run ids with no fingerprint).

    A run with no fingerprint predates provenance tracking; that is reported
    separately from a genuine mismatch, because the remedy differs -- one needs
    re-running to be comparable, the other means an input actually changed.

    The harness commit is excluded: it changes constantly and does not by itself
    alter what a task asks of a model.
    """
    unfingerprinted = [run["run_id"] for run in runs if not run.get("provenance")]
    fingerprinted = [run for run in runs if run.get("provenance")]

    differing = []
    for key in ("dataset_revision", "corpus_digest"):
        seen = {json.dumps(run["provenance"].get(key), sort_keys=True) for run in fingerprinted}
        if len(seen) > 1:
            differing.append(key)

    by_task: dict[str, set[str]] = {}
    for run in fingerprinted:
        by_task.setdefault(run["task"], set()).add(
            json.dumps(run["provenance"].get("prompt_digest"))
        )
    if any(len(digests) > 1 for digests in by_task.values()):
        differing.append("prompt_digest")

    return not differing and not unfingerprinted, differing, unfingerprinted
