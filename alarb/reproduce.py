"""Rerun CPU retrieval and independently score the archived Qwen3 pilot."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from alarb import agentic, colab_demo, corpus, data, dataset_bundle


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = "train-00000-of-00001.parquet"
TRAIN_ROWS = 9_339
TEST_ROWS = 2_673
SEED = 42
PILOT_SIZE = 20


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verified_training_file(local_file: Path | None, cache_dir: Path, source: dict) -> Path:
    spec = source["files"][TRAIN_FILE]
    if local_file is not None:
        dataset_bundle.verify_file(local_file, spec)
        return local_file

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / TRAIN_FILE
    if path.is_file():
        dataset_bundle.verify_file(path, spec)
        return path

    url = f"https://huggingface.co/datasets/{source['id']}/resolve/{source['git_commit']}/data/{TRAIN_FILE}"
    print("Downloading the publisher training file (about 19 MB)", flush=True)
    with tempfile.TemporaryDirectory(prefix="download-", dir=cache_dir) as folder:
        downloaded = Path(folder) / TRAIN_FILE
        with urllib.request.urlopen(url, timeout=90) as response, downloaded.open("wb") as output:
            shutil.copyfileobj(response, output)
        dataset_bundle.verify_file(downloaded, spec)
        downloaded.replace(path)
    return path


def prepare_splits(path: Path, directory: Path) -> tuple[list[data.Case], list[data.Case]]:
    table = pq.read_table(path)
    if table.num_rows != TRAIN_ROWS + TEST_ROWS:
        raise ValueError("The publisher training file has an unexpected number of rows.")
    # Hugging Face Dataset.shuffle(seed=42) uses this NumPy permutation.
    order = np.random.default_rng(SEED).permutation(table.num_rows)
    table = table.take(pa.array(order))
    directory.mkdir(parents=True)
    training = table.slice(0, TRAIN_ROWS)
    testing = table.slice(TRAIN_ROWS)
    pq.write_table(training, directory / "train.parquet")
    pq.write_table(testing, directory / "test.parquet")
    return colab_demo.cases_from_rows(training.to_pylist()), colab_demo.cases_from_rows(testing.to_pylist())


def rescore_pilot(training: list[data.Case], testing: list[data.Case]) -> dict:
    evidence = ROOT / "artifacts" / "colab_pilot_20"
    original = read_json(evidence / "original_config.json")
    expected = read_json(evidence / "pilot_summary.json")
    with (evidence / "pilot_predictions.jsonl").open(encoding="utf-8") as stream:
        predictions = [json.loads(line) for line in stream]
    config = colab_demo.DemoConfig(**original["config"])

    # Regenerate questions from the pinned dataset and verify the archived case
    # IDs and answer keys before consulting the recorded predictions.
    articles, _ = corpus.build_corpus(training + testing)
    items, preparation = colab_demo.prepare_items(testing, articles, config)
    if preparation["sample_case_ids"] != original["preparation"]["sample_case_ids"]:
        raise ValueError("The regenerated question order differs from the original run.")
    if [(record["case_id"], record["reference"]) for record in predictions] != [
        (item.case_id, item.reference) for item in items[:PILOT_SIZE]
    ]:
        raise ValueError("The archived pilot records do not match the regenerated questions.")

    scored = colab_demo.score_records(predictions, PILOT_SIZE, config)
    metrics = ("completed", "correct", "accuracy", "unparseable", "truncated")
    if any(scored[key] != expected[key] for key in metrics):
        raise ValueError("The recomputed pilot score differs from the published evidence.")
    return {
        "mode": "scored archived predictions",
        "new_model_inference": False,
        "model": config.model_id,
        "eligible_questions": preparation["eligible_questions"],
        "corpus_articles": len(articles.articles),
        **{key: scored[key] for key in metrics},
        "paper_accuracy": scored["paper_accuracy"],
        "difference_percentage_points": round(scored["difference_percentage_points"], 2),
    }


def rerun_retrieval(workspace: Path) -> dict:
    expected = read_json(ROOT / "artifacts" / "agentic_smoke_summary.json")
    old_directory = Path.cwd()
    try:
        os.chdir(workspace)
        result = agentic.run_experiment(agentic.SmokeModel(), Path("retrieval"), limit=50,
                                       seed=SEED, top_k=5, max_repairs=1)
    finally:
        os.chdir(old_directory)

    keys = ("items", "corpus_articles", "cases_with_gold_articles", "matched_training_corpus_articles",
            "retrieval_macro_recall_at_k", "verification_status_counts", "corpus_sha256",
            "training_index_sha256")
    differences = [key for key in keys if result[key] != expected[key]]
    if differences:
        raise ValueError("The retrieval run differs from the published reference: " + ", ".join(differences))
    return {
        "mode": "live CPU retrieval with an abstention fixture for drafting",
        **{key: result[key] for key in keys},
    }


def reproduce(source_file: Path | None, output: Path) -> None:
    source = read_json(ROOT / "SOURCE_LOCK.json")["official_dataset"]
    file = verified_training_file(source_file, ROOT / ".cache" / "reproduction", source)
    print("Publisher training Parquet SHA256 verified", flush=True)
    with tempfile.TemporaryDirectory(prefix="alarb-", dir=file.parent if source_file is None else None) as directory:
        workspace = Path(directory)
        training, testing = prepare_splits(file, workspace / data.DEVELOPMENT_DIR)
        print(f"Prepared {len(training):,} training and {len(testing):,} test cases", flush=True)
        print("Running the 50-case retrieval experiment on CPU", flush=True)
        retrieval = rerun_retrieval(workspace)
        print("Rebuilding the pilot questions and scoring archived predictions", flush=True)
        pilot = rescore_pilot(training, testing)

    requirements = (ROOT / "requirements-reproduction.txt").read_text(encoding="utf-8").splitlines()
    packages = [line.split("==", 1)[0] for line in requirements if "==" in line]
    report = {
        "status": "PASS",
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "architecture": platform.machine(),
                        "packages": {name: importlib.metadata.version(name) for name in packages}},
        "dataset_revision": source["git_commit"],
        "publisher_training_sha256": source["files"][TRAIN_FILE]["sha256"],
        "training_cases": len(training),
        "test_cases": len(testing),
        "validation_accessed": False,
        "retrieval": retrieval,
        "pilot": pilot,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="alarb-results-", dir=output.parent) as directory:
        temporary_output = Path(directory) / output.name
        temporary_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary_output.replace(output)

    print(f"Live CPU retrieval: {retrieval['items']} cases, {retrieval['corpus_articles']} articles, "
          f"macro recall@5 = {retrieval['retrieval_macro_recall_at_k']:.4%}")
    print(f"Archived Qwen3-8B predictions: {pilot['correct']}/{pilot['completed']} correct "
          f"({pilot['accuracy']:.2%}); paper reference {pilot['paper_accuracy']:.2%}")
    print("PASS: Both sets of metrics match their published project references.")
    print(f"Results: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-parquet", type=Path)
    args = parser.parse_args()
    source = args.train_parquet.resolve() if args.train_parquet is not None else None
    try:
        reproduce(source, args.output.resolve())
    except (OSError, ValueError) as error:
        print(f"ALARB reproduction failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
