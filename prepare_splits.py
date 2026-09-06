#!/usr/bin/env python3
"""Create course-ready ALARB train, test, and sealed validation splits.

The official ALARB test split is preserved as the final validation holdout. Only
the official training split is shuffled and divided into development train/test
sets. No validation examples are printed or used for development statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


DATASET_ID = "THIQAH-RD/ALARB"
REVISION = "e64bfdc867146294a65434c5ca16c2c4c5288ca2"
SEED = 42
TRAIN_RATIO = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/split_manifest.json")
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing split files. The default protects the sealed holdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    development_dir = args.output_root / "development"
    holdout_dir = args.output_root / "validation_holdout"
    train_path = development_dir / "train.parquet"
    test_path = development_dir / "test.parquet"
    validation_path = holdout_dir / "validation.parquet"
    paths = (train_path, test_path, validation_path)

    existing = [str(path) for path in paths if path.exists()]
    if existing and not args.force:
        if len(existing) == len(paths) and args.manifest.exists():
            print("ALARB three-way splits already exist: PASS")
            print("The sealed validation holdout was not opened or recreated.")
            print(f"Manifest: {args.manifest}")
            return
        raise SystemExit(
            "Only some split files exist, so nothing was changed. Use --force only "
            "when intentionally resetting every split.\n"
            + "Existing: "
            + ", ".join(existing)
        )
    if args.force:
        for path in paths:
            if path.exists():
                path.chmod(0o600)
                path.unlink()

    # The official test set becomes the final holdout. This is a one-time data
    # preparation step; development code must only load data/development/.
    official_train = load_dataset(DATASET_ID, revision=REVISION, split="train")
    official_validation = load_dataset(DATASET_ID, revision=REVISION, split="test")

    total_rows = len(official_train) + len(official_validation)
    train_rows = round(total_rows * TRAIN_RATIO)
    test_rows = len(official_train) - train_rows

    shuffled = official_train.shuffle(seed=args.seed)
    course_train = shuffled.select(range(train_rows))
    course_test = shuffled.select(range(train_rows, len(shuffled)))

    development_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)
    course_train.to_parquet(train_path)
    course_test.to_parquet(test_path)
    official_validation.to_parquet(validation_path)
    validation_path.chmod(0o400)

    counts = {
        "train": len(course_train),
        "test": len(course_test),
        "validation": len(official_validation),
    }
    manifest = {
        "status": "PASS",
        "source": {
            "dataset_id": DATASET_ID,
            "revision": REVISION,
            "official_train_rows": len(official_train),
            "official_test_rows": len(official_validation),
        },
        "strategy": (
            "Preserve the official test split as the sealed validation holdout; "
            "shuffle and divide only the official train split."
        ),
        "seed": args.seed,
        "requested_ratios": {"train": 0.70, "test": 0.20, "validation": 0.10},
        "rows": counts,
        "actual_ratios": {
            name: round(count / total_rows, 6) for name, count in counts.items()
        },
        "paths": {
            "train": str(train_path),
            "test": str(test_path),
            "validation": str(validation_path),
        },
        "validation_policy": (
            "Do not load validation.parquet during training, debugging, model "
            "selection, prompt tuning, or test evaluation. Open it once only for "
            "the final locked evaluation."
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("ALARB three-way split preparation: PASS")
    print(
        f"Rows: train={counts['train']}, test={counts['test']}, "
        f"validation={counts['validation']}, total={total_rows}"
    )
    print(f"Development data: {development_dir}")
    print(f"SEALED validation holdout: {validation_path}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
