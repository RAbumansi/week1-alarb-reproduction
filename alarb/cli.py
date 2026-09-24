#!/usr/bin/env python3
"""Command line entry point for the ALARB benchmark harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alarb import corpus as corpus_module
from alarb import data, embeddings, judge as judge_module, mcq, outcomes, providers, runner, tasks


OUTCOME_PATH = Path("artifacts/outcome_distribution.json")


def load_run_config(run_id: str) -> runner.RunConfig:
    path = runner.RUNS_DIR / run_id / "config.json"
    if not path.is_file():
        raise SystemExit(f"No run at {path}. Run the task first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("run_id", None)
    return runner.RunConfig(**payload)


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


def command_run(args: argparse.Namespace) -> None:
    config = runner.RunConfig(
        task=args.task,
        model=args.model,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        omitted_steps=args.omitted_steps,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    items = runner.build_items(config)
    prompt_tokens = sum(providers.estimate_tokens(item.prompt) for item in items)
    print(f"Run {config.run_id}")
    print(f"  items: {len(items)}   estimated input tokens: {prompt_tokens:,} (approximate)")

    if args.estimate_cost:
        print("  Nothing was sent. Drop --estimate-cost to run.")
        return

    model = providers.get_model(args.model, args.max_tokens, args.temperature)
    runner.run_inference(config, model)
    print(f"Predictions: {config.directory / 'predictions.jsonl'}")
    if config.task in tasks.MCQ_TASKS:
        print(f"Next: python -m alarb.cli score --run {config.run_id}")
    else:
        print(f"Next: python -m alarb.cli judge --run {config.run_id}")


def command_judge(args: argparse.Namespace) -> None:
    config = load_run_config(args.run)
    model = None
    if args.judge != "heuristic":
        model = providers.get_model(args.judge, args.max_tokens, args.temperature)
    judge = judge_module.get_judge(args.judge, model)
    print(f"Judging {config.run_id} with {judge.name}")
    runner.run_judging(config, judge)
    print(f"Judgments: {config.directory / 'judgments.jsonl'}")
    print(f"Next: python -m alarb.cli score --run {config.run_id}")


def command_score(args: argparse.Namespace) -> None:
    config = load_run_config(args.run)
    metrics = runner.score(config)
    print(f"{metrics['run_id']}   model={metrics['model']}   items={metrics['items']}")
    if "accuracy" in metrics:
        print(f"  accuracy {metrics['accuracy']:7.1%}   (random floor {metrics['random_floor']:.0%})")
    else:
        print(f"  judge: {metrics['judge']}")
        for label, values in metrics["labels"].items():
            print(f"  {label:20} {values['share']:7.1%}  (n={values['count']})")
    share = metrics["unparseable"]["share"]
    print(f"  unparseable output: {share:.1%}")
    print(f"Metrics: {config.directory / 'metrics.json'}")


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

    run_parser = subcommands.add_parser("run", help="Generate predictions for one task.")
    run_parser.add_argument("--task", choices=tasks.ALL_TASKS, required=True)
    run_parser.add_argument(
        "--model",
        required=True,
        help="A Claude model id, or 'constant' / 'random' for the offline references.",
    )
    run_parser.add_argument("--split", choices=data.DEVELOPMENT_SPLITS, default="test")
    run_parser.add_argument("--limit", type=int, default=150)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--omitted-steps", type=int, default=tasks.DEFAULT_OMITTED_STEPS)
    run_parser.add_argument("--max-tokens", type=int, default=providers.DEFAULT_MAX_TOKENS)
    run_parser.add_argument("--temperature", type=float, default=providers.DEFAULT_TEMPERATURE)
    run_parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Report item and token counts without calling the model.",
    )
    run_parser.set_defaults(handler=command_run)

    judge_parser = subcommands.add_parser("judge", help="Grade a run's generated verdicts.")
    judge_parser.add_argument("--run", required=True)
    judge_parser.add_argument(
        "--judge",
        default="heuristic",
        help="A Claude model id for the paper's Appendix B.1 judge, or 'heuristic' offline.",
    )
    judge_parser.add_argument("--max-tokens", type=int, default=providers.DEFAULT_MAX_TOKENS)
    judge_parser.add_argument("--temperature", type=float, default=providers.DEFAULT_TEMPERATURE)
    judge_parser.set_defaults(handler=command_judge)

    score_parser = subcommands.add_parser("score", help="Aggregate a run into metrics.")
    score_parser.add_argument("--run", required=True)
    score_parser.set_defaults(handler=command_score)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
