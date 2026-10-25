"""Run a task, judge the output, score the result.

Inference and judging are separate passes writing separate files. Judging reads
predictions.jsonl rather than calling the model again, so a judge can be swapped
or a parser bug fixed without re-paying for generation -- which matters, because
generation is the expensive half.

A run directory is named after what produced it rather than when, so re-running
the same task and model resumes into the same place. With the completion cache
behind it, a resumed run costs nothing for work already done.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from alarb import mcq, outcomes, parsing, provenance, tasks
from alarb.data import Case, load_split
from alarb.judge import LABELS, UNPARSEABLE, Judge
from alarb.providers import Model


RUNS_DIR = Path("runs")
DEFAULT_CONCURRENCY = 4


@dataclass
class RunConfig:
    task: str
    model: str
    split: str = "test"
    limit: int | None = None
    seed: int = 42
    omitted_steps: int = tasks.DEFAULT_OMITTED_STEPS
    language: str = tasks.DEFAULT_LANGUAGE
    max_tokens: int = 1024
    temperature: float = 0.0
    notes: dict = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        slug = self.model.replace("/", "-").replace(":", "-")
        parts = [self.task, slug, self.split]
        if self.limit:
            parts.append(f"n{self.limit}")
        if self.task == "argument_completion":
            parts.append(f"omit{self.omitted_steps}")
        return "__".join(parts)

    @property
    def directory(self) -> Path:
        return RUNS_DIR / self.run_id


def stratified_sample(cases: list[Case], limit: int, seed: int) -> list[Case]:
    """A deterministic sample that keeps the outcome mix of the full split."""
    if limit >= len(cases):
        return cases
    grouped: dict[str, list[Case]] = {}
    for case in cases:
        grouped.setdefault(outcomes.classify_verdict(case.verdict), []).append(case)

    picked: list[Case] = []
    for label in sorted(grouped):
        bucket = sorted(grouped[label], key=lambda case: case.case_id)
        share = round(limit * len(bucket) / len(cases))
        take = min(len(bucket), max(1, share))
        picked.extend(random.Random(f"{seed}:{label}").sample(bucket, take))

    picked.sort(key=lambda case: case.case_id)
    if len(picked) > limit:
        picked = random.Random(seed).sample(picked, limit)
        picked.sort(key=lambda case: case.case_id)
    return picked


def build_items(config: RunConfig) -> list[tasks.TaskItem]:
    cases = load_split(config.split)
    if config.limit:
        cases = stratified_sample(cases, config.limit, config.seed)

    if config.task in tasks.VERDICT_TASKS:
        return tasks.build_verdict_items(
            config.task, cases, config.omitted_steps, config.language
        )

    path = mcq.TASK_DIR / f"{config.task}.{config.split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Run: python -m alarb.cli mcq")
    wanted = {case.case_id for case in cases}
    by_id = {case.case_id: case for case in cases}
    return [
        tasks.build_article_mcq(
            item.task, item.case_id, by_id[item.case_id].case_facts,
            item.choices, item.answer, item.metadata,
        )
        for item in mcq.load_mcqs(path)
        if item.case_id in wanted
    ]


def completed_case_ids(path: Path) -> set[str]:
    """Cases already written to a predictions file, so a run can pick up where it
    stopped. Any trailing partial line is ignored."""
    if not path.is_file():
        return set()
    done = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                done.add(json.loads(line)["case_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _predict(item: tasks.TaskItem, model: Model, is_mcq: bool) -> dict:
    completion = model.complete(item.prompt)
    if is_mcq:
        answer = parsing.parse_mcq_answer(completion.text)
        parsed, ok = {"answer": answer}, answer is not None
    else:
        verdict = parsing.parse_verdict(completion.text)
        parsed = {"verdict": verdict.verdict, "reasoning": verdict.reasoning}
        ok = verdict.ok
    return {
        "case_id": item.case_id,
        "task": item.task,
        "reference": item.reference,
        "parsed": parsed,
        "parsed_ok": ok,
        "raw": completion.text,
        "model": completion.model,
        "cached": completion.cached,
        "input_tokens": completion.input_tokens,
        "output_tokens": completion.output_tokens,
        "metadata": item.metadata,
    }


def run_inference(
    config: RunConfig,
    model: Model,
    concurrency: int = DEFAULT_CONCURRENCY,
    resume: bool = True,
    progress_every: int = 25,
) -> Path:
    items = build_items(config)
    directory = config.directory
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(
            asdict(config)
            | {"run_id": config.run_id, "provenance": provenance.current(config.task)},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    predictions = directory / "predictions.jsonl"
    already = completed_case_ids(predictions) if resume else set()
    pending = [item for item in items if item.case_id not in already]
    if already:
        print(f"  resuming: {len(already)} done, {len(pending)} to go")
    if not pending:
        print(f"  {len(already)} items, nothing left to do")
        return predictions

    is_mcq = config.task in tasks.MCQ_TASKS
    totals = Counter()
    # Results are written in input order so the file stays diffable, and each is
    # flushed as it arrives so an interrupted run keeps everything up to that point.
    with predictions.open("a" if already else "w", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = pool.map(lambda item: _predict(item, model, is_mcq), pending)
            for index, record in enumerate(results, start=1):
                totals["items"] += 1
                totals["unparseable"] += 0 if record["parsed_ok"] else 1
                totals["cached"] += 1 if record["cached"] else 0
                totals["input_tokens"] += record["input_tokens"]
                totals["output_tokens"] += record["output_tokens"]
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                if progress_every and index % progress_every == 0:
                    print(f"  {index}/{len(pending)} items", flush=True)

    print(
        f"  {totals['items']} items, {totals['cached']} from cache, "
        f"{totals['unparseable']} unparseable, "
        f"{totals['input_tokens']:,} in / {totals['output_tokens']:,} out tokens"
    )
    return predictions


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _judge_one(record: dict, judge: Judge) -> dict:
    prediction = record["parsed"]["verdict"]
    if not record["parsed_ok"] or not prediction:
        return {"case_id": record["case_id"], "label": UNPARSEABLE, "judge": judge.name, "raw": ""}
    judged = judge.judge(record["reference"], prediction)
    return {
        "case_id": record["case_id"],
        "label": judged.label,
        "judge": judged.judge,
        "raw": judged.raw,
    }


def run_judging(
    config: RunConfig,
    judge: Judge,
    concurrency: int = DEFAULT_CONCURRENCY,
    resume: bool = True,
    progress_every: int = 25,
) -> Path:
    if config.task in tasks.MCQ_TASKS:
        raise ValueError(
            "Multiple-choice tasks are scored by comparing letters; they need no judge."
        )
    directory = config.directory
    records = read_jsonl(directory / "predictions.jsonl")
    judgments = directory / "judgments.jsonl"

    already = completed_case_ids(judgments) if resume else set()
    pending = [record for record in records if record["case_id"] not in already]
    if already:
        print(f"  resuming: {len(already)} judged, {len(pending)} to go")
    if not pending:
        print(f"  {len(already)} judged, nothing left to do")
        return judgments

    with judgments.open("a" if already else "w", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = pool.map(lambda record: _judge_one(record, judge), pending)
            for index, result in enumerate(results, start=1):
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                if progress_every and index % progress_every == 0:
                    print(f"  {index}/{len(pending)} judged", flush=True)
    return judgments


def score(config: RunConfig) -> dict:
    directory = config.directory
    predictions = read_jsonl(directory / "predictions.jsonl")
    total = len(predictions)
    unparseable = sum(1 for record in predictions if not record["parsed_ok"])

    config_path = directory / "config.json"
    stored = (
        json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    )
    metrics: dict = {
        "run_id": config.run_id,
        "task": config.task,
        "model": config.model,
        "split": config.split,
        "items": total,
        "unparseable": {"count": unparseable, "share": round(unparseable / total, 4)},
        "provenance": stored.get("provenance", {}),
    }

    if config.task in tasks.MCQ_TASKS:
        correct = sum(
            1
            for record in predictions
            if record["parsed_ok"] and record["parsed"]["answer"] == record["reference"]
        )
        metrics["accuracy"] = round(correct / total, 4)
        metrics["correct"] = correct
        metrics["random_floor"] = round(1 / len(tasks.CHOICE_LETTERS), 4)
    else:
        judgments_path = directory / "judgments.jsonl"
        if not judgments_path.is_file():
            raise FileNotFoundError(
                f"{judgments_path} is missing. Run the judge before scoring."
            )
        judgments = read_jsonl(judgments_path)
        counts = Counter(record["label"] for record in judgments)
        metrics["judge"] = judgments[0]["judge"] if judgments else None
        metrics["labels"] = {
            label: {
                "count": counts.get(label, 0),
                "share": round(counts.get(label, 0) / total, 4),
            }
            for label in (*LABELS, UNPARSEABLE)
        }

    (directory / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def collect_runs(runs_dir: Path = RUNS_DIR) -> list[dict]:
    """Every scored run on disk, ordered by task then model."""
    found = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(runs_dir.glob("*/metrics.json"))
    ]
    order = {task: index for index, task in enumerate(tasks.ALL_TASKS)}
    return sorted(found, key=lambda run: (order.get(run["task"], 99), run["model"]))
