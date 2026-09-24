#!/usr/bin/env python3
"""Command line entry point for the ALARB benchmark harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alarb import corpus as corpus_module
from alarb import data, embeddings, mcq, outcomes, tasks


OUTCOME_PATH = Path("artifacts/outcome_distribution.json")


def command_corpus(args: argparse.Namespace) -> None:
    corpus = corpus_module.write_corpus(args.output)
    grouped = corpus.by_document()
    eligible = [document for document, items in grouped.items() if len(items) >= 4]
    print("ALARB article corpus: PASS")
    print(f"Articles: {len(corpus.articles)} across {len(grouped)} documents")
    print(f"Duplicate-text keys merged: {len(corpus.aliases)}")
    print(f"Documents supporting same-statute distractors: {len(eligible)}/{len(grouped)}")
    print(f"Corpus: {args.output}")


def command_outcomes(args: argparse.Namespace) -> None:
    distribution = outcomes.outcome_distribution(data.load_development())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(distribution, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("ALARB verdict outcome distribution: PASS")
    for label, values in distribution["paper_categories"].items():
        print(f"  {label:18} {values['share']:7.1%}  (n={values['count']})")
    majority = distribution["majority_class"]
    print(
        f"Majority-class reference: always predicting {majority['label']} "
        f"matches {majority['share']:.1%} of cases."
    )
    print(f"Report: {args.output}")


def command_mcq(args: argparse.Namespace) -> None:
    corpus = corpus_module.load_corpus()
    cases = data.load_split(args.split)
    summary = {}

    for task in tasks.MCQ_TASKS:
        items, skipped = mcq.build_mcqs(
            task,
            cases,
            corpus,
            seed=args.seed,
            embedder_backend=args.embedder,
            embedder_model=args.embedder_model,
        )
        path = mcq.TASK_DIR / f"{task}.{args.split}.jsonl"
        mcq.write_mcqs(items, path)
        summary[task] = {"items": len(items), "skipped": skipped, "path": str(path)}
        print(f"  {task:18} {len(items):5} items  ->  {path}")

    manifest = {
        "source_split": f"data/development/{args.split}.parquet",
        "cases": len(cases),
        "seed": args.seed,
        "choices_per_item": mcq.CHOICES_PER_ITEM,
        "embedder": args.embedder,
        "embedder_model": args.embedder_model if args.embedder == "sentence-transformers" else None,
        "corpus_articles": len(corpus.articles),
        "distractor_rule": (
            "Distractors exclude every article cited by the case, not only the "
            "article being asked about, so no item has a second correct answer."
        ),
        "paper_reference": {"items_per_variant": 1159, "source_cases": 1329},
        "tasks": summary,
    }
    MANIFEST_PATH = mcq.MANIFEST_PATH
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Manifest: {MANIFEST_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m alarb.cli", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    corpus_parser = subcommands.add_parser(
        "corpus", help="Rebuild the statute article corpus from the development cases."
    )
    corpus_parser.add_argument("--output", type=Path, default=corpus_module.CORPUS_PATH)
    corpus_parser.set_defaults(handler=command_corpus)

    outcomes_parser = subcommands.add_parser(
        "outcomes", help="Report the verdict outcome breakdown and majority-class reference."
    )
    outcomes_parser.add_argument("--output", type=Path, default=OUTCOME_PATH)
    outcomes_parser.set_defaults(handler=command_outcomes)

    mcq_parser = subcommands.add_parser(
        "mcq", help="Build both article-identification MCQ variants."
    )
    mcq_parser.add_argument("--split", choices=data.DEVELOPMENT_SPLITS, default="test")
    mcq_parser.add_argument("--seed", type=int, default=mcq.SEED)
    mcq_parser.add_argument(
        "--embedder", choices=embeddings.BACKENDS, default="sentence-transformers"
    )
    mcq_parser.add_argument(
        "--embedder-model", default=embeddings.DEFAULT_SENTENCE_TRANSFORMER
    )
    mcq_parser.set_defaults(handler=command_mcq)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
