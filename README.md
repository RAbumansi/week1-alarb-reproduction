# Week 1: ALARB parent-paper reproduction

This project implements an independent reproduction harness for **ALARB: An Arabic Legal Argument Reasoning Benchmark** and an experimental retrieval and verification workflow. See `UPGRADE_REPORT.md` for the October 25 coding update and verified results. The paper's full numerical results have not been reproduced.

## What succeeded

- The official Hugging Face loading example was executed successfully.
- The official train/test parquet files were integrity-checked against their Git LFS hashes.
- Schema, row counts, descriptive statistics, empty fields, duplicate rows, and train/test overlap were audited.
- A deterministic course-ready training/test/validation split was created with the validation holdout isolated from development files.
- The paper's experimental requirements and missing artifacts were documented.

## Important scope correction

ALARB is about Saudi **commercial court cases**. The published release still contains some personal names despite the authors' anonymization step. It cannot determine whether a real person has a criminal record. See `AGENTIC_PROJECT_SCOPE.md` for the scope and observed data limitations.

## Run it

Recommended setup: Python 3.13, about 500 MB of free disk space for the lightweight workflow, and internet access. The launcher accepts Python 3.10 through 3.13, but availability of the pinned binary dependencies also depends on your Python and operating system.

```bash
cd week1_alarb_reproduction
chmod +x run_week1.sh
./run_week1.sh
```

The first run creates `.venv`, installs the core dependencies, downloads the approximately 21 MB official dataset into `.cache/huggingface`, executes the official loader, writes `artifacts/verification.json`, and creates three local datasets. The audit pins dataset revision `e64bfdc867146294a65434c5ca16c2c4c5288ca2`; the separate loader file remains the exact simple example shown by Hugging Face. `requirements-core.txt` supports the TF-IDF benchmark and agent prototype without Torch. Install `requirements.txt` separately if you want the sentence-transformer embedder.

## Collect the dataset for the course

Official first-hand source: [THIQAH-RD/ALARB](https://huggingface.co/datasets/THIQAH-RD/ALARB), used by [the parent paper](https://arxiv.org/abs/2510.00694).

```bash
.venv/bin/python -m alarb.cli collect-dataset
```

This creates `data/collected/ALARB_dataset.zip` and `artifacts/dataset_collection.json`. The ZIP includes the unmodified original train/test Parquet files, the three course splits, the publisher card and citation, the full declared dataset license, SHA256 checksums, and the split policy. Existing downloaded publisher files are checked against `SOURCE_LOCK.json`; missing files are downloaded from the pinned revision. Use `--offline` to require cached files, or `--local-dir /path/to/original/parquet/files` for another verified local copy. The default refuses to replace an existing ZIP; `--force` explicitly rebuilds it.

The two official files contain 12,012 training cases and 1,329 test cases, totaling 13,341. The paper reports 13,344; this discrepancy remains documented. Course files are a second view of these same cases, not an additional dataset. The validation file is copied as opaque bytes without decoding its cases. Keep this research archive out of the public repository because the source release contains residual identifiers.

## Agentic prototype

The `agentic` command implements fact intake, article retrieval, model drafting, deterministic citation/schema verification, and up to two correction requests. It always marks outputs for human review. Retrieval uses a corpus of **654 articles built only from the 9,339 training cases**, plus similar training cases' cited provisions. Test-set verdicts, court reasoning, and cited laws are excluded from model prompts. Test cited laws are used only after inference to calculate retrieval recall.

Offline wiring check, requiring no API key or model download:

```bash
.venv/bin/python -m alarb.cli agentic --model smoke --limit 50 --output runs/agentic-smoke
```

`smoke` is an explicit abstention fixture, not a language model. It checks execution and retrieval; it provides no verdict-quality evidence. Each output directory must be new, so earlier experiments are preserved. The completed 50-case development run and limitations are recorded in `artifacts/agentic_smoke_summary.json`.

To use an installed local model:

```bash
ollama serve
# In another terminal, after installing qwen3:8b in Ollama:
.venv/bin/python -m alarb.cli agentic --model ollama:qwen3:8b --limit 10 --top-k 5 --max-repairs 1 --output runs/agentic-qwen3
```

Local traces are saved under `runs/` and excluded from Git. A verification pass means the JSON is complete and citations name retrieved articles; it does **not** establish legal correctness or whether the articles support every claim. Evaluate verdict quality separately, calibrate a judge against human labels, and keep the final validation data sealed.

## Training, test, and validation split

The publisher supplies 12,012 training rows and 1,329 test rows. For the course requirement, `prepare_splits.py` preserves the publisher's test set as the unseen **validation** holdout, then deterministically splits the publisher's training set using seed 42:

| Dataset | Rows | Share | Location |
|---|---:|---:|---|
| Training | 9,339 | 70.002% | `data/development/train.parquet` |
| Test | 2,673 | 20.036% | `data/development/test.parquet` |
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
.venv/bin/python -m alarb.cli mcq --embedder tfidf  # lightweight task construction
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
and writes `artifacts/results.json`. It warns when the runs in it were built
from different corpora or prompts, since those scores do not mean the same
thing; each run records a fingerprint of its inputs in `config.json`.

Runs resume. Interrupting one and re-running the same command continues from
where it stopped rather than starting over.

### Validating the judge

The judge decides CORRECT against PARTIALLY CORRECT against INCORRECT, so it
*is* the metric and an unchecked judge leaves every number unverified:

```bash
.venv/bin/python -m alarb.cli agreement --run <run_id> --sample 30
# fill the human_label column in the CSV it writes, then
.venv/bin/python -m alarb.cli agreement --run <run_id> --score
```

The sample is stratified across the judge's own labels and shuffled, so row
order gives nothing away. It reports raw agreement and Cohen's kappa — kappa
matters because raw agreement flatters a judge that leans on one label.

### Concurrency

`--concurrency` sets parallel requests. Hosted models benefit substantially;
**local models need the server configured to match, or it does nothing.**
Measured on an M2 with Qwen3-8B:

| setup | per case |
|---|---|
| serial | 50.4s |
| `--concurrency 4`, Ollama default | 57.5s — *slower*, pure queueing overhead |
| `--concurrency 4` with `OLLAMA_NUM_PARALLEL=4` | 38.1s |

Ollama defaults to `OLLAMA_NUM_PARALLEL=1` and serializes everything, so start
it as `OLLAMA_NUM_PARALLEL=4 ollama serve`. The 1.3x ceiling is memory
bandwidth, not dispatch — don't expect more from one laptop.

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
.venv/bin/python -m pytest tests/    # 139 tests, no API key required
```

## Run integrity and tests

Resume now checks the task inputs, decoding settings and provenance before appending predictions. A truncated final JSONL record is preserved in a `.interrupted-tail` file and removed from the append stream; malformed middle records and duplicate IDs are rejected. Judging cannot mix different judges, thresholds or prediction files. Scoring rejects empty predictions, incomplete runs, incomplete judgments and mixed judges. Use `--no-resume` when intentionally restarting an incompatible run, or choose a distinct run configuration/output directory.

```bash
# Dataset-independent checks suitable for CI:
.venv/bin/python -m pytest tests/test_upgrades.py -q
# Full suite after preparing data and both MCQ task files:
.venv/bin/python -m pytest tests/ -q
```

The October 25 verification passed **139 tests**. The supplied GitHub Actions workflow is configured to run the synthetic regression suite without downloading case data or opening validation once it is published. No hosted CI execution is claimed in this update.

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
- `AGENTIC_PROJECT_SCOPE.md`: implemented prototype scope, research plan and data limitations.
- `alarb/agentic.py`: retrieval, model drafting, citation verification, and bounded correction.
- `alarb/dataset_bundle.py`: verified dataset collection and course submission archive.
- `artifacts/dataset_collection.json`: source, counts, file hashes and archive verification.
- `artifacts/upgrade_offline_results.json`: the newly executed six-task offline checks.
- `artifacts/agentic_smoke_summary.json`: aggregate prototype execution and retrieval check.
- `UPGRADE_REPORT.md`: current implementation, evidence, and remaining work.

## What to tell the instructor

The official ALARB dataset works, but the ALARB inference, MCQ-construction, judging, and fine-tuning code is not public. A search of the paper's own links, the Hugging Face repository, the THIQAH GitHub organisation and the authors' accounts found no implementation.

We therefore took the independent-reimplementation route: the `alarb/` harness rebuilds the benchmark from the prompts printed in Appendices B.1 and B.2, and `DEVIATIONS.md` records every point where it departs from the paper. The paper's reported scores are **not** claimed as reproduced, and our numbers are not comparable to its tables — different judge, different embedder, a smaller article pool, and a different evaluation split.

Worth confirming with the instructor: that an independent reimplementation is acceptable in place of running the authors' code, and that evaluating on our development split rather than the paper's test set is the right call given the sealed-holdout policy.

Do not present the earlier `Thiqah/ArabLegalEval` repository as ALARB's implementation. It belongs to a different paper and carries no licence.

## Sources

- Paper: https://arxiv.org/abs/2510.00694
- Official dataset: https://huggingface.co/datasets/THIQAH-RD/ALARB
- Related predecessor code (not ALARB): https://github.com/Thiqah/ArabLegalEval
