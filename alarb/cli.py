#!/usr/bin/env python3
"""Command line entry point for the ALARB benchmark harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alarb import corpus as corpus_module
from alarb import data, outcomes


OUTCOME_PATH = Path("artifacts/outcome_distribution.json")


def command_corpus(args: argparse.Namespace) -> None:
    articles = corpus_module.write_corpus(args.output)
    grouped = corpus_module.articles_by_document(articles)
    eligible = [document for document, items in grouped.items() if len(items) >= 4]
    print("ALARB article corpus: PASS")
    print(f"Articles: {len(articles)} across {len(grouped)} documents")
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

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
