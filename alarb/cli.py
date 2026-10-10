#!/usr/bin/env python3
"""Command line entry point for the ALARB benchmark harness."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from alarb import agreement, corpus as corpus_module
from alarb import (
    data,
    embeddings,
    judge as judge_module,
    mcq,
    outcomes,
    provenance,
    providers,
    runner,
    tasks,
)


OUTCOME_PATH = Path("artifacts/outcome_distribution.json")


def load_run_config(run_id: str) -> runner.RunConfig:
    path = runner.RUNS_DIR / run_id / "config.json"
    if not path.is_file():
        raise SystemExit(f"No run at {path}. Run the task first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    # config.json also carries derived metadata (run_id, provenance) that is not
    # part of the config itself. Keep only real fields so adding more metadata
    # later cannot break loading an old run.
    fields = {field.name for field in dataclasses.fields(runner.RunConfig)}
    return runner.RunConfig(**{k: v for k, v in payload.items() if k in fields})


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
    print(f"  concurrency: {args.concurrency}")
    runner.run_inference(config, model, concurrency=args.concurrency, resume=not args.no_resume)
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
    print(f"Judging {config.run_id} with {judge.name} (concurrency {args.concurrency})")
    runner.run_judging(config, judge, concurrency=args.concurrency, resume=not args.no_resume)
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


RESULTS_PATH = Path("artifacts/results.json")


def command_agreement(args: argparse.Namespace) -> None:
    config = load_run_config(args.run)
    sheet = agreement.REVIEW_DIR / f"{config.run_id}.csv"

    if args.score:
        if not sheet.is_file():
            raise SystemExit(f"No review sheet at {sheet}. Create one without --score first.")
        report = agreement.score_agreement(agreement.read_review_sheet(sheet))
        report_path = agreement.REVIEW_DIR / f"{config.run_id}.agreement.json"
        agreement.write_report(report, report_path)

        print(f"Judge agreement for {config.run_id}")
        print(f"  reviewed by hand : {report['reviewed']}")
        print(f"  raw agreement    : {report['raw_agreement']:.1%}")
        if report["cohens_kappa"] is None:
            print(f"  Cohen's kappa    : {report['kappa_reading']}")
        else:
            print(
                f"  Cohen's kappa    : {report['cohens_kappa']:.3f}  ({report['kappa_reading']})"
            )
        if report["unlabelled"]:
            print(f"  still unlabelled : {report['unlabelled']}")
        for label in report["invalid_labels"]:
            print(f"  ignored bad label: {label}")
        if report["disagreements"]:
            print(f"\n  where they differ ({len(report['disagreements'])}):")
            for row in report["disagreements"][:10]:
                print(f"    judge={row['judge']:18} human={row['human']:18} {row['case_id']}")
        print(f"\nReport: {report_path}")
        return

    if sheet.is_file() and not args.force:
        raise SystemExit(
            f"{sheet} already exists and may hold your labels. Pass --force to replace it, "
            "or --score to read it."
        )
    rows = agreement.sample_for_review(
        runner.read_jsonl(config.directory / "predictions.jsonl"),
        runner.read_jsonl(config.directory / "judgments.jsonl"),
        args.sample,
        args.seed,
    )
    agreement.write_review_sheet(rows, sheet)
    print(f"Wrote {len(rows)} items for review, stratified across the judge's labels.")
    print(f"  {sheet}")
    print(
        f"\nFill the '{agreement.HUMAN_COLUMN}' column with one of "
        f"{', '.join(agreement.VALID_LABELS)}, judging each prediction against the"
    )
    print("gold verdict yourself. Then:")
    print(f"  python -m alarb.cli agreement --run {config.run_id} --score")


def command_report(args: argparse.Namespace) -> None:
    runs = runner.collect_runs()
    if not runs:
        raise SystemExit("No scored runs yet. Run a task, then judge and score it.")

    verdict_runs = [run for run in runs if "labels" in run]
    mcq_runs = [run for run in runs if "accuracy" in run]

    if verdict_runs:
        judges = {run.get("judge") for run in verdict_runs if run.get("judge")}
        heuristic_only = judges and all(j.startswith("heuristic") for j in judges)
        print("Verdict prediction\n")
        if heuristic_only:
            print("  NOT the paper's metric: graded offline by word overlap, which")
            print("  scores wording rather than meaning and understates any answer")
            print("  phrased differently from the court. See DEVIATIONS.md section 3.\n")
        print(f"  {'task':22} {'model':24} {'n':>5} {'CORRECT':>9} {'PARTIAL':>9} {'WRONG':>9} {'unparsed':>9}")
        for run in verdict_runs:
            labels = run["labels"]
            print(
                f"  {run['task']:22} {run['model']:24} {run['items']:5} "
                f"{labels['CORRECT']['share']:8.1%} {labels['PARTIALLY CORRECT']['share']:8.1%} "
                f"{labels['INCORRECT']['share']:8.1%} {run['unparseable']['share']:8.1%}"
            )
        print(f"\n  judge: {', '.join(sorted(judges))}")

    if mcq_runs:
        print("\nArticle identification — four-way multiple choice\n")
        print(f"  {'task':22} {'model':24} {'n':>5} {'accuracy':>9} {'floor':>9}")
        for run in mcq_runs:
            print(
                f"  {run['task']:22} {run['model']:24} {run['items']:5} "
                f"{run['accuracy']:8.1%} {run['random_floor']:8.1%}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"runs": runs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ok, differing, unfingerprinted = provenance.comparable(runs)
    if differing:
        print("\n  WARNING: these runs were not built from the same inputs.")
        print(f"  Differing: {', '.join(differing)}.")
        print("  Those scores are not comparable to each other. Re-run the affected")
        print("  tasks against one corpus and one set of prompts.")
    if unfingerprinted:
        print(f"\n  NOTE: {len(unfingerprinted)} run(s) predate provenance tracking and")
        print("  record no fingerprint, so they cannot be confirmed comparable:")
        for run_id in unfingerprinted[:6]:
            print(f"    {run_id}")
        print("  Re-run them to bring them under the same check.")

    print(f"\nResults: {args.output}")
    print("Comparability: see DEVIATIONS.md. These are not the paper's numbers.")


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
    run_parser.add_argument(
        "--concurrency",
        type=int,
        default=runner.DEFAULT_CONCURRENCY,
        help="Parallel requests. Hosted models tolerate 8-16; a local model on a "
        "laptop is better at 2-4.",
    )
    run_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start over instead of continuing from what the run already has.",
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
    judge_parser.add_argument(
        "--concurrency", type=int, default=runner.DEFAULT_CONCURRENCY
    )
    judge_parser.add_argument("--no-resume", action="store_true")
    judge_parser.set_defaults(handler=command_judge)

    agreement_parser = subcommands.add_parser(
        "agreement",
        help="Validate the judge against your own labels on a stratified sample.",
    )
    agreement_parser.add_argument("--run", required=True)
    agreement_parser.add_argument("--sample", type=int, default=30)
    agreement_parser.add_argument("--seed", type=int, default=42)
    agreement_parser.add_argument(
        "--score", action="store_true", help="Read the filled sheet and report agreement."
    )
    agreement_parser.add_argument(
        "--force", action="store_true", help="Replace an existing review sheet."
    )
    agreement_parser.set_defaults(handler=command_agreement)

    score_parser = subcommands.add_parser("score", help="Aggregate a run into metrics.")
    score_parser.add_argument("--run", required=True)
    score_parser.set_defaults(handler=command_score)

    report_parser = subcommands.add_parser(
        "report", help="Render every scored run as a comparison table."
    )
    report_parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    report_parser.set_defaults(handler=command_report)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
