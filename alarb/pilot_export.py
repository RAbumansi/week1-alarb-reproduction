"""Export the fixed first N real predictions as an explicitly labeled pilot.

The original run and its configuration are preserved. This never marks the
planned larger experiment complete and never chooses examples by correctness.
"""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from dataclasses import replace
from pathlib import Path

from alarb.colab_demo import DemoConfig, score_records


def export_pilot(source: Path, output: Path, count: int = 20) -> dict:
    if count < 1:
        raise ValueError("Pilot count must be positive.")
    if source.resolve() == output.resolve() or output.exists():
        raise FileExistsError("Use a new pilot directory; the source run must be preserved.")
    original_config = json.loads((source / "config.json").read_text())
    original_summary = json.loads((source / "summary.json").read_text())
    if original_config.get("backend", {}).get("backend") != "transformers" or original_summary.get("execution_backend") != "transformers":
        raise ValueError("Only real-model records may be exported as pilot evidence.")
    config = DemoConfig(**original_config["config"])
    if count > config.sample_size:
        raise ValueError("Pilot cannot exceed the originally planned sample.")
    records = [json.loads(line) for line in (source / "predictions.jsonl").read_text().splitlines() if line.strip()]
    if len({r["case_id"] for r in records}) != len(records):
        raise ValueError("Duplicate source predictions.")
    if len(records) < count:
        raise ValueError(f"Need {count} completed predictions, found {len(records)}.")
    chosen = records[:count]
    expected = original_config.get("preparation", {}).get("sample_case_ids", [])[:count]
    if [r["case_id"] for r in chosen] != expected:
        raise ValueError("Pilot must be the fixed first questions of the original frozen sample.")
    summary = score_records(chosen, count, replace(config, sample_size=count))
    summary.update({
        "status": "PILOT_COMPLETE", "execution_backend": "transformers",
        "original_planned_questions": config.sample_size,
        "observed_source_predictions": len(records),
        "original_experiment_complete": original_summary["status"] == "COMPLETE",
        "source_code_commit": original_config.get("code_commit"),
        "source_signature": original_config["signature"],
        "selection": "Fixed first questions of original frozen sample; no filtering by correctness",
        "reason_for_pilot": "User requested a shorter pilot because of free-GPU runtime",
        "similarity_to_paper_established": False,
    })
    output.mkdir(parents=True)
    (output / "pilot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "original_config.json").write_text(json.dumps(original_config, indent=2) + "\n")
    (output / "original_summary.json").write_text(json.dumps(original_summary, indent=2) + "\n")
    (output / "pilot_predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in chosen))
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["experiment", "questions", "correct", "accuracy", "scope"])
        writer.writerow(["Parent paper Qwen3-8B Table 5", 1159, "not reconstructed from rounded percentage", 0.846, "original protocol"])
        writer.writerow(["Our real-model pilot", count, summary["correct"], summary["accuracy"], "first questions of frozen development sample; NF4"])
    (output / "README.txt").write_text(
        "This is an explicitly reduced pilot, not a completed 100-question experiment.\n"
        "The larger original run and its configuration are preserved.\n"
        "All first-N questions are retained, including wrong, unparseable and truncated outputs.\n"
        "Numeric proximity to the paper does not establish equivalent evaluation conditions.\n"
        "No raw case facts, personal names or generated reasoning are included.\n")
    archive_path = output / "ALARB_Pilot_Evidence.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("pilot_summary.json", "original_config.json", "original_summary.json",
                     "pilot_predictions.jsonl", "comparison.csv", "README.txt"):
            archive.write(output / name, name)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    summary = export_pilot(args.source, args.output, args.count)
    print(json.dumps(summary, indent=2))
    print(f"Evidence: {args.output / 'ALARB_Pilot_Evidence.zip'}")


if __name__ == "__main__":
    main()
