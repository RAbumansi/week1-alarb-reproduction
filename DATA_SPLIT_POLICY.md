# Data split and validation policy

## Course split

The official ALARB release contains 12,012 training rows and 1,329 test rows.
For the course workflow, the official test split is preserved unchanged and
renamed **validation**. It is the final unseen holdout. The official training
split is deterministically shuffled with seed 42 and divided into new training
and test sets.

| Dataset | Rows | Actual share | Permitted use |
|---|---:|---:|---|
| Training | 9,339 | 70.002% | Fit models and prompts |
| Test | 2,673 | 20.035% | Debugging and model selection |
| Validation | 1,329 | 9.962% | One final evaluation only |

The small deviation from exactly 70/20/10 is caused by preserving all 1,329
rows of the publisher's original test split as the holdout.

## Holdout rule

`prepare_splits.py` performs the one-time mechanical creation of the three
files, but it never prints validation examples or calculates validation content
statistics. It places the holdout separately at
`data/validation_holdout/validation.parquet` and makes the file read-only.

During development, code must load only:

- `data/development/train.parquet`
- `data/development/test.parquet`

Do not load, inspect, search, summarize, tune against, or repeatedly score on
the validation file. Open it only after the complete pipeline and evaluation
method are frozen. Record that final result once.

The generated `data/` directory is excluded from Git because the official data
remains available from its authoritative Hugging Face source. The committed
`artifacts/split_manifest.json` records the reproducible split procedure.
