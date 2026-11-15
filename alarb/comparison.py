"""Compare verdict runs case by case, on the cases they all share.

Conditions that differ only in what the model was given (no articles, retrieved
articles, the court's own articles) are scored on the same cases with the same
judge, so a difference in the table comes from the input and not from a
different sample. With samples this small the paired view matters more than
the averages: how many cases got better, how many got worse, and whether that
split is distinguishable from chance.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from alarb import runner
from alarb.judge import LABELS, UNPARSEABLE

RESULTS_PATH = Path("artifacts/verdict_comparison.json")
GRADE = {"CORRECT": 2, "PARTIALLY CORRECT": 1, "INCORRECT": 0, UNPARSEABLE: 0}


def describe(task: str, retriever: str | None) -> str:
    if task == "verdict_facts":
        return "Facts only (paper task 1)"
    if task == "verdict_laws":
        return "Facts + court-cited articles (paper task 2, ceiling)"
    if task == "verdict_rag" and retriever == "popularity":
        return "Facts + most-cited articles (control)"
    if task == "verdict_rag":
        return f"Facts + retrieved articles ({retriever})"
    return task


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(centre - half, 4), round(centre + half, 4))


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial test that improvements and regressions are equally likely."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return round(min(1.0, 2 * tail), 4)


def _load(run_id: str) -> dict:
    directory = runner.RUNS_DIR / run_id
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    predictions = {r["case_id"]: r for r in runner.read_jsonl(directory / "predictions.jsonl")}
    judgments_path = directory / "judgments.jsonl"
    if not judgments_path.is_file():
        raise FileNotFoundError(f"{run_id} has no judgments yet; run the judge first.")
    judgments = {r["case_id"]: r for r in runner.read_jsonl(judgments_path)}
    if set(judgments) != set(predictions):
        raise ValueError(f"{run_id}: judgments do not cover exactly the predictions.")
    return {"run_id": run_id, "config": config, "predictions": predictions, "judgments": judgments}


def compare_runs(run_ids: list[str], output: Path | None = RESULTS_PATH) -> dict:
    """The first run is the baseline every other run is paired against."""
    if len(run_ids) < 2:
        raise ValueError("Compare at least two runs.")
    runs = [_load(run_id) for run_id in run_ids]
    judges = {j["judge"] for run in runs for j in run["judgments"].values()}
    if len(judges) != 1:
        raise ValueError(f"Runs were graded by different judges ({sorted(judges)}); regrade with one.")
    common = sorted(set.intersection(*(set(run["predictions"]) for run in runs)))
    if not common:
        raise ValueError("The runs share no cases.")

    baseline = runs[0]
    rows = []
    for run in runs:
        labels = [run["judgments"][cid]["label"] for cid in common]
        records = [run["predictions"][cid] for cid in common]
        good = sum(label in ("CORRECT", "PARTIALLY CORRECT") for label in labels)
        row = {
            "run_id": run["run_id"],
            "condition": describe(run["config"]["task"], run["config"].get("retriever")),
            "cases": len(common),
            "labels": {label: round(labels.count(label) / len(common), 4) for label in (*LABELS, UNPARSEABLE)},
            "correct_or_partial": round(good / len(common), 4),
            "correct_or_partial_95ci": wilson(good, len(common)),
            "mean_grade": round(sum(GRADE[label] for label in labels) / len(common), 4),
            "outcome_agreement": runner.outcome_agreement(records),
            "mean_input_tokens": round(sum(r["input_tokens"] for r in records) / len(records)),
        }
        if run is not baseline:
            deltas = [
                GRADE[run["judgments"][cid]["label"]] - GRADE[baseline["judgments"][cid]["label"]]
                for cid in common
            ]
            wins = sum(d > 0 for d in deltas)
            losses = sum(d < 0 for d in deltas)
            row["vs_baseline"] = {
                "better": wins,
                "worse": losses,
                "same": len(deltas) - wins - losses,
                "sign_test_p": sign_test(wins, losses),
            }
        rows.append(row)

    result = {
        "judge": judges.pop(),
        "baseline": baseline["run_id"],
        "shared_cases": len(common),
        "grade_scale": "CORRECT 2, PARTIALLY CORRECT 1, INCORRECT or unparseable 0",
        "rows": rows,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
