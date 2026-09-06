# Week 1 execution report

Execution date: 2026-09-06  
Host: Apple M3 MacBook Air, 16 GB RAM  
Python: 3.12  
Hugging Face Datasets: 4.0.0

## Selected public code

The executable code published specifically for ALARB is the Hugging Face dataset-loading example in `official_hf_loader.py`. The paper and dataset page do not link an ALARB GitHub repository.

The THIQAH GitHub organization currently exposes `ArabLegalData` and `ArabLegalEval`. The latter is code for the authors' earlier ArabLegalEval paper. It was inspected at commit `7e9530039324a45cf7b585b618663e3359ed589c`; it must not be described as ALARB code.

## Commands executed

```bash
PYTHONPATH=work/hf-python-packages \
HF_HOME=work/hf-cache \
python3 - <<'PY'
from datasets import load_dataset
ds = load_dataset("THIQAH-RD/ALARB")
print(ds)
print(ds["train"].features)
print({split: len(ds[split]) for split in ds})
PY
```

## Verified result

```text
DatasetDict({
    train: 12012 rows
    test: 1329 rows
})
features: case_facts, court_reasoning, applicable_laws, verdict
```

The two parquet files were also downloaded and matched the SHA-256 values stored in the repository's Git LFS pointers. A local audit found:

- 13,341 total public rows.
- No exact duplicate rows within either split.
- No identical `case_facts` values shared between train and test.
- 988 cases with no `applicable_laws` entry; this is permitted by the published schema.
- 18 raw law-name prefixes and 878 raw `(document, article)` keys. Several prefixes are spelling or naming variants of the same statute.

## Differences from the paper

The paper reports 13,344 cases, while the released data contains 13,341. Its summary table reports a maximum of 15 applicable-law entries and 977 applicable-law words; the current public data reaches 16 entries and 1,512 whitespace-delimited words. The paper reports average applicable-law text length 186 words; the current release averages 198.6751 with simple whitespace tokenization. These may reflect a different dataset revision or a different counting procedure.

## Reproduction status

- Official data access: **PASS**
- Dataset schema and split verification: **PASS**
- File integrity verification: **PASS**
- Parent-paper model inference: **NOT RUNNABLE FROM PUBLIC ARTIFACTS**
- Parent-paper fine-tuning: **NOT RUNNABLE FROM PUBLIC ARTIFACTS**

The last two are evidence gaps, not execution failures: the ALARB model code/configs were not released, and the paper's full fine-tuning run requires three A100 GPUs. Exact benchmark scores should not be claimed as reproduced.

## Course three-way split

The official release contains only publisher-defined train and test splits. To
meet the course requirement without weakening the holdout, the publisher's
1,329-row test split is preserved as validation. Only the 12,012-row publisher
training split is shuffled (seed 42) and divided into 9,339 training rows and
2,673 development-test rows. The resulting overall shares are 70.002%,
20.035%, and 9.962%.

Split preparation was executed successfully. The final validation file is
stored separately, made read-only, and excluded from development code. See
`DATA_SPLIT_POLICY.md` and `artifacts/split_manifest.json`.
