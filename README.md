# Week 1: ALARB parent-paper reproduction

This package reproduces everything that is currently public and executable for the first parent paper, **ALARB: An Arabic Legal Argument Reasoning Benchmark**.

## What succeeded

- The official Hugging Face loading example was executed successfully.
- The official train/test parquet files were integrity-checked against their Git LFS hashes.
- Schema, row counts, descriptive statistics, empty fields, duplicate rows, and train/test overlap were audited.
- A deterministic course-ready training/test/validation split was created with the validation holdout isolated from development files.
- The paper's experimental requirements and missing artifacts were documented.

## Important scope correction

ALARB is about reasoning over anonymized Saudi **commercial court cases**. It is not a criminal-record dataset and cannot determine whether a real person has a record. See `AGENTIC_PROJECT_SCOPE.md` for a research direction that matches the parent paper.

## Run it

Requirements: Python 3.10 or newer, about 500 MB of free disk space, and internet access.

```bash
cd week1_alarb_reproduction
chmod +x run_week1.sh
./run_week1.sh
```

The first run creates `.venv`, installs the pinned dependency, downloads the approximately 21 MB official dataset into `.cache/huggingface`, executes the official loader, writes `artifacts/verification.json`, and creates three local datasets. The audit pins dataset revision `e64bfdc867146294a65434c5ca16c2c4c5288ca2`; the separate loader file remains the exact simple example shown by Hugging Face.

## Training, test, and validation split

The publisher supplies 12,012 training rows and 1,329 test rows. For the course requirement, `prepare_splits.py` preserves the publisher's test set as the unseen **validation** holdout, then deterministically splits the publisher's training set using seed 42:

| Dataset | Rows | Share | Location |
|---|---:|---:|---|
| Training | 9,339 | 70.002% | `data/development/train.parquet` |
| Test | 2,673 | 20.035% | `data/development/test.parquet` |
| Validation | 1,329 | 9.962% | `data/validation_holdout/validation.parquet` |

The validation file is created once, stored separately, and made read-only. Training, debugging, prompt tuning, and model selection must use only `data/development/`. Do not open the holdout until the pipeline is frozen and ready for one final evaluation. See `DATA_SPLIT_POLICY.md`.

To create only the three splits:

```bash
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python prepare_splits.py
```

The split files are intentionally excluded from Git; they can always be regenerated from the pinned official dataset. Their counts and policy are recorded in `artifacts/split_manifest.json`.

To run the audit directly:

```bash
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python audit_alarb.py
```

To audit already downloaded parquet files without fetching the dataset again:

```bash
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python audit_alarb.py \
  --local-parquet-dir /path/to/ALARB/data
```

## Benchmark harness

The `alarb/` package reimplements the paper's six tasks. ALARB's benchmark code
was never released, but Appendices B.1 and B.2 print four of the prompts in
full, which is enough to rebuild the pipeline. Read `DEVIATIONS.md` before
quoting any number from it — results here are **not** comparable to the paper's
tables.

Build the task data once (no API key needed):

```bash
.venv/bin/python -m alarb.cli corpus     # statute articles, recovered from the cases
.venv/bin/python -m alarb.cli outcomes   # verdict breakdown + majority-class reference
.venv/bin/python -m alarb.cli mcq        # both article-identification variants
```

Then run a task, grade it, and score it. Models come from one of three places:

```bash
--model ollama:qwen3:8b      # local, free, no key; also a row in the paper's own table
--model claude-...           # hosted, needs ANTHROPIC_API_KEY
--model constant | random    # offline reference points, no key
```

```bash
ollama serve &                                    # for local models
.venv/bin/python -m alarb.cli run --task verdict_facts --model ollama:qwen3:8b --limit 150
.venv/bin/python -m alarb.cli judge --run verdict_facts__ollama-qwen3-8b__test__n150 --judge heuristic
.venv/bin/python -m alarb.cli score --run verdict_facts__ollama-qwen3-8b__test__n150
```

Any server speaking `/v1/chat/completions` works — Ollama, LM Studio, vLLM —
via `OLLAMA_HOST`.

Generating locally and judging with a hosted model is a reasonable split.
Generation is the expensive half (long prompts, one call per case); judging is
short, because verdicts run 13–26 words. Because generation and judging are
separate passes, you can generate once locally and re-judge later with a better
judge without re-running generation.

Add `--estimate-cost` to `run` to see item and token counts without sending
anything. Every call is cached under `.cache/llm/`, so an interrupted run
resumes for free and re-scoring never re-pays for generation.

`python -m alarb.cli report` renders every scored run as one comparison table
and writes `artifacts/results.json`.

The six tasks are `verdict_facts`, `verdict_laws`, `verdict_reasoning`,
`argument_completion`, `mcq_same_statute` and `mcq_semantic`.

Two models need no credentials and give the results something to be read
against: `--model constant` answers every case with the same generic verdict,
and `--model random` picks multiple-choice letters at random. `--judge
heuristic` grades by word overlap offline — useful for checking the pipeline,
but it is not the paper's metric.

**A model is only interesting if it clearly beats these.** 59.1% of cases end
in an order against the defendant, and the multiple-choice floor is 25%.

```bash
.venv/bin/python -m pytest tests/    # 77 tests, no API key required
```

## Files

- `official_hf_loader.py`: the published ALARB loading example.
- `audit_alarb.py`: a verification wrapper that does not expose case text.
- `prepare_splits.py`: creates deterministic 70/20/10 development and holdout files.
- `alarb/`: the benchmark harness — data loading, article corpus, prompts, tasks,
  MCQ construction, inference, judging and scoring.
- `tests/`: holdout-guard, task-construction and pipeline tests.
- `artifacts/verification.json`: machine-readable result from the verified run.
- `artifacts/split_manifest.json`: counts, ratios, source revision, and holdout policy.
- `artifacts/article_corpus.json`: the 693-article statute corpus rebuilt from the cases.
- `artifacts/mcq_manifest.json`: MCQ counts, seed, embedder, and distractor rule.
- `DEVIATIONS.md`: every place this implementation departs from the paper.
- `DATA_SPLIT_POLICY.md`: rules preventing validation leakage.
- `EXECUTION_REPORT.md`: what was executed and what could not be reproduced.
- `PAPER_METHOD.md`: experiment and fine-tuning settings extracted from the paper.
- `SOURCE_LOCK.json`: exact paper, dataset, file hashes, and related-code commits.
- `AGENTIC_PROJECT_SCOPE.md`: a later-phase agentic direction and safety boundary.

## What to tell the instructor

The official ALARB dataset works, but the ALARB inference, MCQ-construction, judging, and fine-tuning code is not public. A search of the paper's own links, the Hugging Face repository, the THIQAH GitHub organisation and the authors' accounts found no implementation.

We therefore took the independent-reimplementation route: the `alarb/` harness rebuilds the benchmark from the prompts printed in Appendices B.1 and B.2, and `DEVIATIONS.md` records every point where it departs from the paper. The paper's reported scores are **not** claimed as reproduced, and our numbers are not comparable to its tables — different judge, different embedder, a smaller article pool, and a different evaluation split.

Worth confirming with the instructor: that an independent reimplementation is acceptable in place of running the authors' code, and that evaluating on our development split rather than the paper's test set is the right call given the sealed-holdout policy.

Do not present the earlier `Thiqah/ArabLegalEval` repository as ALARB's implementation. It belongs to a different paper and carries no licence.

## Sources

- Paper: https://arxiv.org/abs/2510.00694
- Official dataset: https://huggingface.co/datasets/THIQAH-RD/ALARB
- Related predecessor code (not ALARB): https://github.com/Thiqah/ArabLegalEval
