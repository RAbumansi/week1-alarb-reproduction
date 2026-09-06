# Week 1: ALARB parent-paper reproduction

This package reproduces everything that is currently public and executable for the first parent paper, **ALARB: An Arabic Legal Argument Reasoning Benchmark**.

## What succeeded

- The official Hugging Face loading example was executed successfully.
- The official train/test parquet files were integrity-checked against their Git LFS hashes.
- Schema, row counts, descriptive statistics, empty fields, duplicate rows, and train/test overlap were audited.
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

The first run creates `.venv`, installs the pinned dependency, downloads the approximately 21 MB official dataset into `.cache/huggingface`, executes the official loader, and writes `artifacts/verification.json`. The audit pins dataset revision `e64bfdc867146294a65434c5ca16c2c4c5288ca2`; the separate loader file remains the exact simple example shown by Hugging Face.

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
- `artifacts/verification.json`: machine-readable result from the verified run.
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
