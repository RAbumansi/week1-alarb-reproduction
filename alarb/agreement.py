"""Check the judge against a human, because the judge is the metric.

Every verdict score in this project is whatever the judge said, so an
unvalidated judge makes every number unverifiable. This samples a run's
judgements into a spreadsheet, waits for a person to label the same items
blind, and reports how far the two agree.

The sample is stratified across the judge's own labels. A random draw from a
run where the judge called 90% of answers INCORRECT would be almost entirely
INCORRECT items and would say nothing about whether its CORRECT calls can be
trusted -- which is the half that matters.

Cohen's kappa is reported alongside raw agreement because raw agreement is
flattering when one label dominates: a judge that answers INCORRECT every time
scores 90% against a 90% INCORRECT truth while having learned nothing.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

from alarb.judge import LABELS, UNPARSEABLE


REVIEW_DIR = Path("artifacts/judge_review")
HUMAN_COLUMN = "human_label"
VALID_LABELS = (*LABELS, UNPARSEABLE)

# Landis & Koch bands, the convention this is usually read against.
KAPPA_BANDS = (
    (0.81, "almost perfect"),
    (0.61, "substantial"),
    (0.41, "moderate"),
    (0.21, "fair"),
    (0.0, "slight"),
)


def interpret_kappa(kappa: float) -> str:
    if kappa < 0:
        return "worse than chance"
    for threshold, label in KAPPA_BANDS:
        if kappa >= threshold:
            return label
    return "slight"


def sample_for_review(
    predictions: list[dict], judgments: list[dict], size: int, seed: int = 42
) -> list[dict]:
    by_case = {record["case_id"]: record for record in predictions}
    grouped: dict[str, list[dict]] = {}
    for judgement in judgments:
        record = by_case.get(judgement["case_id"])
        if record is None:
            continue
        grouped.setdefault(judgement["label"], []).append(
            {
                "case_id": judgement["case_id"],
                "judge_label": judgement["label"],
                HUMAN_COLUMN: "",
                "gold_verdict": record["reference"],
                "predicted_verdict": record["parsed"].get("verdict") or "",
            }
        )

    labels = sorted(grouped)
    if not labels:
        return []
    per_label = max(1, size // len(labels))
    picked: list[dict] = []
    remaining: list[dict] = []
    for label in labels:
        bucket = sorted(grouped[label], key=lambda row: row["case_id"])
        shuffled = random.Random(f"{seed}:{label}").sample(bucket, len(bucket))
        picked.extend(shuffled[:per_label])
        remaining.extend(shuffled[per_label:])

    # Small or lopsided label sets leave the quota unfilled; top up from whatever
    # is left so asking for 30 yields 30 whenever 30 exist.
    if len(picked) < size and remaining:
        shortfall = size - len(picked)
        picked.extend(random.Random(seed).sample(remaining, min(shortfall, len(remaining))))

    # Shuffle so the reviewer cannot infer the judge's label from row order.
    random.Random(seed).shuffle(picked)
    return picked[:size]


def write_review_sheet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case_id", "judge_label", HUMAN_COLUMN, "gold_verdict", "predicted_verdict"]
    # utf-8-sig so Excel opens the Arabic correctly rather than as mojibake.
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_review_sheet(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def score_agreement(rows: list[dict]) -> dict:
    labelled, invalid = [], []
    for row in rows:
        human = (row.get(HUMAN_COLUMN) or "").strip().upper()
        if not human:
            continue
        if human not in VALID_LABELS:
            invalid.append({"case_id": row["case_id"], HUMAN_COLUMN: human})
            continue
        labelled.append((row["judge_label"], human))

    if not labelled:
        raise ValueError(
            f"No rows have a {HUMAN_COLUMN}. Fill that column with one of "
            f"{', '.join(VALID_LABELS)} and run this again."
        )

    judge_labels = [pair[0] for pair in labelled]
    human_labels = [pair[1] for pair in labelled]
    agreed = sum(1 for a, b in labelled if a == b)

    # Kappa is undefined when judge and human between them used only one label --
    # which happens easily on a small sample, or on a run where the judge gave
    # everything the same grade. Report that rather than emitting a NaN that
    # would serialise to invalid JSON.
    from sklearn.metrics import cohen_kappa_score

    if len(set(judge_labels)) < 2 and len(set(human_labels)) < 2:
        kappa = None
    else:
        raw = cohen_kappa_score(judge_labels, human_labels, labels=list(VALID_LABELS))
        kappa = None if raw != raw else round(float(raw), 4)

    confusion = Counter(labelled)
    disagreements = [
        {
            "case_id": row["case_id"],
            "judge": row["judge_label"],
            "human": (row.get(HUMAN_COLUMN) or "").strip().upper(),
            "gold_verdict": row.get("gold_verdict", ""),
            "predicted_verdict": row.get("predicted_verdict", ""),
        }
        for row in rows
        if (row.get(HUMAN_COLUMN) or "").strip().upper() in VALID_LABELS
        and row["judge_label"] != (row.get(HUMAN_COLUMN) or "").strip().upper()
    ]

    return {
        "reviewed": len(labelled),
        "unlabelled": sum(1 for row in rows if not (row.get(HUMAN_COLUMN) or "").strip()),
        "invalid_labels": invalid,
        "raw_agreement": round(agreed / len(labelled), 4),
        "cohens_kappa": kappa,
        "kappa_reading": (
            interpret_kappa(kappa)
            if kappa is not None
            else "undefined -- too few distinct labels in this sample to correct for chance"
        ),
        "judge_distribution": dict(Counter(judge_labels).most_common()),
        "human_distribution": dict(Counter(human_labels).most_common()),
        "confusion": {f"judge={a} human={b}": count for (a, b), count in confusion.most_common()},
        "disagreements": disagreements,
    }


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
