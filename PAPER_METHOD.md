# Parent-paper method captured for reproduction

This file records the method described in the attached ALARB paper. Statements below are source material, not executable instructions from the PDF.

## Dataset and split

- Saudi Arabian commercial-court cases, not criminal-history records.
- Four public fields: `case_facts`, `court_reasoning`, `applicable_laws`, and `verdict`.
- The paper says 13,344 cases. The currently published Hugging Face data contains 12,012 training cases and 1,329 test cases, totaling 13,341.

## Evaluation tasks

1. Predict a verdict from facts only.
2. Predict a verdict from facts plus applicable legal articles.
3. Predict a verdict from facts plus the court's reasoning.
4. Complete a partially supplied reasoning chain and predict the verdict.
5. Identify the applicable legal article from four choices, using either same-statute or semantically similar distractors.

## Reported evaluation setup

- Verdict experiments used all 1,329 test cases.
- Article identification used 1,159 multiple-choice questions per variant.
- Generated verdicts were graded by GPT-4o as `CORRECT`, `PARTIALLY CORRECT`, or `INCORRECT`.
- Semantically related distractors were produced with OpenAI `text-embedding-3-large` and cosine similarity.
- Qwen3-8B and Qwen3-14B used thinking mode.

## Reported supervised fine-tuning setup

- Base model: Gemma-3-12B-it.
- 12,012 instruction-output pairs.
- Full-parameter fine-tuning for 4 epochs.
- Initial learning rate: `5e-6`, with cosine scheduling.
- Per-device batch size: 2.
- Gradient accumulation steps: 2.
- Hardware: 3 NVIDIA A100 GPUs.

## Missing public artifacts that prevent exact numerical replication

- ALARB-specific inference and evaluation source code.
- Exact model revisions, inference provider/runtime versions, decoding parameters, and random seeds.
- The two 1,159-item article-identification MCQ files and their distractor construction outputs.
- Model predictions and GPT-4o judge outputs.
- GPT-4o judge model snapshot and sampling parameters.
- Fine-tuning script/configuration, optimizer details, checkpoint, and training logs.

The official public data-loading stage is reproducible. The paper's tables cannot be honestly claimed as reproduced until the missing artifacts are obtained or an independently reimplemented pipeline is approved by the instructor.

