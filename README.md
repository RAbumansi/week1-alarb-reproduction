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

## Files

- `official_hf_loader.py`: the published ALARB loading example.
- `audit_alarb.py`: a verification wrapper that does not expose case text.
- `prepare_splits.py`: creates deterministic 70/20/10 development and holdout files.
- `artifacts/verification.json`: machine-readable result from the verified run.
- `artifacts/split_manifest.json`: counts, ratios, source revision, and holdout policy.
- `DATA_SPLIT_POLICY.md`: rules preventing validation leakage.
- `EXECUTION_REPORT.md`: what was executed and what could not be reproduced.
- `PAPER_METHOD.md`: experiment and fine-tuning settings extracted from the paper.
- `SOURCE_LOCK.json`: exact paper, dataset, file hashes, and related-code commits.
- `AGENTIC_PROJECT_SCOPE.md`: a later-phase agentic direction and safety boundary.

## What to tell the instructor

The official ALARB dataset works, but the ALARB inference, MCQ-construction, judging, and fine-tuning code is not public. Ask whether the public-data reproduction is sufficient for Week 1 or whether you should (a) obtain the code from the authors, (b) receive permission to independently implement the paper, or (c) select a parent paper with a released, licensed repository. Do not present the earlier `Thiqah/ArabLegalEval` repository as ALARB's implementation.

## Sources

- Paper: https://arxiv.org/abs/2510.00694
- Official dataset: https://huggingface.co/datasets/THIQAH-RD/ALARB
- Related predecessor code (not ALARB): https://github.com/Thiqah/ArabLegalEval
