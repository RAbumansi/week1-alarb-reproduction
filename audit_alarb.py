#!/usr/bin/env python3
"""Verify the public ALARB dataset and compare it with the parent paper.

This is a thin verification wrapper around the official Hugging Face loader.
It does not reproduce unpublished model inference or fine-tuning code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from datasets import DatasetDict, concatenate_datasets, load_dataset


DATASET_ID = "THIQAH-RD/ALARB"
DATASET_REVISION = "e64bfdc867146294a65434c5ca16c2c4c5288ca2"
EXPECTED_FEATURES = {
    "case_facts",
    "court_reasoning",
    "applicable_laws",
    "verdict",
}
PAPER_CLAIMS = {
    "total_rows": 13344,
    "case_facts": {"words_min": 31, "words_max": 398, "words_avg": 181, "steps_min": 3, "steps_max": 11, "steps_avg": 8},
    "court_reasoning": {"words_min": 18, "words_max": 296, "words_avg": 129, "steps_min": 1, "steps_max": 11, "steps_avg": 6},
    "applicable_laws": {"words_min": 0, "words_max": 977, "words_avg": 186, "steps_min": 0, "steps_max": 15, "steps_avg": 3},
    "verdict": {"words_min": 5, "words_max": 26, "words_avg": 13},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-parquet-dir",
        type=Path,
        help="Read train/test parquet files from this directory instead of Hugging Face.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/verification.json"),
        help="Where to write the JSON audit report.",
    )
    return parser.parse_args()


def load_alarb(local_parquet_dir: Path | None) -> tuple[DatasetDict, dict[str, str]]:
    if local_parquet_dir is None:
        return load_dataset(DATASET_ID, revision=DATASET_REVISION), {
            "kind": "huggingface",
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
        }

    files = {
        "train": str(local_parquet_dir / "train-00000-of-00001.parquet"),
        "test": str(local_parquet_dir / "test-00000-of-00001.parquet"),
    }
    missing = [path for path in files.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing parquet files: {missing}")
    return load_dataset("parquet", data_files=files), {"kind": "local_parquet", "files": files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_hash(row: dict[str, Any]) -> str:
    stable = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def facts_hash(row: dict[str, Any]) -> str:
    stable = "\n".join(row["case_facts"]).strip()
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def list_field_stats(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    steps = [len(row[field]) for row in rows]
    words = [sum(len(item.split()) for item in row[field]) for row in rows]
    return {
        "steps_min": min(steps),
        "steps_max": max(steps),
        "steps_avg": round(mean(steps), 4),
        "words_min": min(words),
        "words_max": max(words),
        "words_avg": round(mean(words), 4),
        "empty_rows": sum(value == 0 for value in steps),
    }


def verdict_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    words = [len(row["verdict"].split()) for row in rows]
    return {
        "words_min": min(words),
        "words_max": max(words),
        "words_avg": round(mean(words), 4),
        "empty_rows": sum(not row["verdict"].strip() for row in rows),
    }


def law_prefixes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prefixes: Counter[str] = Counter()
    document_articles: set[tuple[str, str]] = set()
    total = 0
    for row in rows:
        for law in row["applicable_laws"]:
            parts = law.split(":", 2)
            document = parts[0]
            article = parts[1] if len(parts) > 1 else ""
            prefixes[document] += 1
            document_articles.add((document, article))
            total += 1
    return {
        "total_references": total,
        "raw_document_prefix_count": len(prefixes),
        "raw_document_article_key_count": len(document_articles),
        "references_by_raw_document_prefix": dict(prefixes.most_common()),
    }


def main() -> None:
    args = parse_args()
    dataset, source = load_alarb(args.local_parquet_dir)
    split_rows = {split: dataset[split].to_list() for split in dataset}
    rows = concatenate_datasets([dataset["train"], dataset["test"]]).to_list()

    feature_names = set(dataset["train"].features)
    if feature_names != EXPECTED_FEATURES:
        raise ValueError(f"Unexpected schema: {sorted(feature_names)}")

    duplicate_counts = {}
    for split, values in split_rows.items():
        hashes = [record_hash(row) for row in values]
        duplicate_counts[split] = len(hashes) - len(set(hashes))

    train_facts = {facts_hash(row) for row in split_rows["train"]}
    test_facts = {facts_hash(row) for row in split_rows["test"]}

    report: dict[str, Any] = {
        "status": "PASS",
        "source": source,
        "rows": {
            "train": len(dataset["train"]),
            "test": len(dataset["test"]),
            "total": len(rows),
        },
        "features": {name: str(spec) for name, spec in dataset["train"].features.items()},
        "observed_statistics": {
            "case_facts": list_field_stats(rows, "case_facts"),
            "court_reasoning": list_field_stats(rows, "court_reasoning"),
            "applicable_laws": list_field_stats(rows, "applicable_laws"),
            "verdict": verdict_stats(rows),
        },
        "integrity": {
            "exact_duplicates_within_split": duplicate_counts,
            "case_facts_overlap_between_train_and_test": len(train_facts & test_facts),
        },
        "law_prefix_audit": law_prefixes(rows),
        "paper_claims": PAPER_CLAIMS,
        "paper_public_dataset_row_difference": len(rows) - PAPER_CLAIMS["total_rows"],
    }

    if args.local_parquet_dir:
        report["local_file_sha256"] = {
            path.name: sha256_file(path)
            for path in sorted(args.local_parquet_dir.glob("*.parquet"))
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("ALARB public-data verification: PASS")
    print(f"Rows: train={len(dataset['train'])}, test={len(dataset['test'])}, total={len(rows)}")
    print(f"Features: {', '.join(sorted(feature_names))}")
    print(f"Exact duplicate rows: train={duplicate_counts['train']}, test={duplicate_counts['test']}")
    print(f"Train/test case-facts overlap: {len(train_facts & test_facts)}")
    print(f"Paper/public-dataset row difference: {report['paper_public_dataset_row_difference']}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
