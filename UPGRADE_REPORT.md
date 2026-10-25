# ALARB coding update

Reporting date: October 25, 2026.

## Latest code pulled

Merged the latest `origin/alarb-benchmark-harness` changes through `169bf01`.
Those commits add parallel inference/judging, resumable output, input provenance,
human judge-agreement review, and the finding that the released data still
contains personal names. Earlier local split changes were preserved.

## New implementation

### Agentic prototype

`alarb/agentic.py` implements fact intake, retrieval, model drafting, verification
and bounded correction. A failed citation or JSON check can trigger one correction
by default, with a maximum of two. Every result requires human review.

The retrieval index uses only the 9,339 training cases and their cited statutory
articles. It contains 654 unique articles. It combines article-text character
TF-IDF similarity (weight 0.25) and cited provisions from similar training cases
(weight 0.75). This is a lightweight research baseline. The older MCQ corpus is
still the separate 693-article corpus built from both development splits for
reconstructing the benchmark tasks; it is not used by the agent's retriever.

The draft and correction stages receive case facts and retrieved evidence.
They do not receive test verdicts, test court reasoning or test cited laws.
Test cited laws are consulted only after generation to calculate retrieval recall.
The verifier checks output structure and whether citation keys are present in the
retrieved evidence. It does not establish legal correctness or factual entailment.

### Dataset collection

`python -m alarb.cli collect-dataset` packages the pinned parent-paper dataset:

- Official source: https://huggingface.co/datasets/THIQAH-RD/ALARB
- Parent paper: https://arxiv.org/abs/2510.00694
- Revision: `e64bfdc867146294a65434c5ca16c2c4c5288ca2`.
- Unmodified publisher files: 12,012 train and 1,329 test cases, totaling 13,341.
- Course splits: 9,339 training, 2,673 test and 1,329 final validation cases.
- Publisher card and citation, full declared Apache 2.0 dataset license,
  original-file hashes, prepared-file hashes and split policy are included.

The existing original Parquet files matched both expected byte sizes and SHA256
hashes in `SOURCE_LOCK.json`. ZIP CRC checks passed. The original and prepared
files are two views of the same cases. Validation was copied as bytes only;
its cases were not decoded or used in this update. Raw data remain excluded
from the public repository. The collection manifest is safe to commit because
it contains metadata and hashes rather than case text.

### Run integrity

- Resume rejects changed decoding settings, prompts, corpus, dataset revision or
  task inputs before modifying the existing run configuration.
- A truncated final JSONL record is preserved separately before resuming, so
  appends cannot merge new JSON into a damaged partial line.
- Malformed middle records, duplicate case IDs and out-of-scope records fail
  explicitly rather than being silently skipped.
- Judge resume checks the judge, thresholds, prompt and prediction-file hash.
- Scoring refuses empty predictions, incomplete inference, incomplete or
  duplicated judgments, mixed judges and predictions changed after grading.
- The setup supports a lightweight dependency file without Torch.
- GitHub Actions is configured to run synthetic regression checks without data.

The workflow is supplied in the local code package. Publication was attempted,
but terminal Git has no saved credentials and the connected GitHub integration
returns HTTP 403 for write operations. The upgrades are committed on the local
`upgrades/agentic-dataset-oct25` branch and exported as a separate code ZIP.
The remote repository has not been changed by this update, and no hosted CI
execution is claimed.

## Executed verification

The full suite passed **139 tests** in a fresh Python 3.13 environment. This
includes 27 added parameterized regression cases covering the agent boundary,
fabricated citations, bounded correction, abstention, training-only indexing,
dataset hashes, byte-preserving packaging and interrupted-run recovery.

All six offline benchmark workflows completed on development test samples.
Four verdict tasks used the constant-output reference and the token-overlap
heuristic grader; both MCQ tasks used deterministic random answers. These runs
verify the pipeline, not learned-model capability. Their full aggregate results
are in `artifacts/upgrade_offline_results.json`; the earlier collaborators'
`artifacts/results.json` was preserved.

The new hybrid agent workflow was executed on 50 development test cases with
seed 42 and top-k 5. All 50 offline fixture outputs explicitly abstained.
Among the 49 cases with parseable gold article references, macro retrieval
recall at 5 was **0.561978**. This measures overlap with court-cited articles,
not exhaustive legal relevance or verdict accuracy. This small development
sample was used while implementing retrieval; it is not an unseen final result.
See `artifacts/agentic_smoke_summary.json` for the aggregate record.

A live Ollama server was not running during this update. Therefore, no new
language-model quality result is claimed for the agentic prototype. Synthetic
model tests exercise valid answers, invalid answers and successful correction;
the offline 50-case fixture tests orchestration and retrieval only.

## Remaining research work

Run the prototype with a real model; compare it to a facts-only baseline on the
same frozen sample and model settings; assess citation entailment and verdict
quality with human review and a calibrated judge; then run larger development
experiments before the single final validation evaluation. Fine-tuning has not
been performed. The paper's published scores are not claimed as reproduced.

The dataset is for Saudi commercial-court reasoning. The new source finding
corrects earlier wording that described it as fully anonymized: residual
identifiers are present. No criminal-history lookup is implemented.
