# Measured ALARB free-GPU pilot — November 7, 2026

Real Qwen3-8B inference completed on a free Google Colab Tesla T4. This is a reduced development replication of ALARB's same-regulation legal-article identification task. It is not a criminal-record lookup system.

| Experiment | Questions | Correct | Accuracy |
|---|---:|---:|---:|
| Parent paper, Qwen3-8B, Table 5 Same Regulation | 1,159 | Not inferred from rounded percentage | 84.60% |
| Our real-model pilot | 20 | 15 | 75.00% |

The pilot is **9.60 percentage points lower**. Its 95% Wilson interval is **53.13%–88.81%**. This wide interval and the protocol differences below do not establish equivalence with the paper. Three answers were unparseable and counted as failures; two parsed answers were wrong. No outputs were marked truncated, and all completed their thinking section. The saved records cannot identify why those three final answers failed strict parsing because generated reasoning/text is deliberately excluded from the export.

## What actually ran

- Inference source commit: `2de32db0ac3fd5c806ea5653d795c0baf755b45f` (`alarb-colab-demo-v1`).
- Separate CPU pilot-export code: `b2cf5d4d9449a5fa7f081438df7d0b48fe3bcd61` (`alarb-colab-pilot-v1`). The original inference checkout and records were preserved.
- Model: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`, thinking enabled, NF4 quantization with float16 compute.
- Python 3.13.16; torch 2.14.1+cu130; transformers 4.57.1; bitsandbytes 0.48.1; accelerate 1.10.1; datasets 4.0.0.
- GPU: Tesla T4, 14.563 GiB available VRAM. Model weights downloaded to the cloud VM, not the Mac. No paid model API was used.
- Dataset: publisher training Parquet only, pinned revision `e64bfdc867146294a65434c5ca16c2c4c5288ca2`, 12,012 rows, SHA256 checked against the source lock. Course training/development split: 9,339/2,673 cases. The 1,329-case final validation set was not downloaded or opened in this experiment.
- The initial frozen sample had 100 questions, selected with seed 42. At the user's request to shorten free-GPU time, execution was stopped after exactly 20 saved predictions. The pilot contains the fixed first 20, with no filtering by correctness. The original 100-question experiment remains **INCOMPLETE**.
- The 20 prediction times sum to 1,360 seconds (22 minutes 40 seconds), excluding setup, the overlapping three-question smoke test, and any interrupted work on the next question.
- Local software verification: 160 tests passed; both source notebooks passed schema and code-cell syntax checks. These checks are separate from model-performance evidence.

## Limits of the comparison

We use different development cases, independently reconstructed prompts and distractors, and a reconstructed 693-article corpus rather than the full statutory corpus. The corpus is development-derived, including development-test material, so this experiment is a task demonstration rather than a clean unseen generalization test. NF4 precision and our recorded generation settings may also differ from the paper. It covers one task, without the paper's supervised fine-tuning or other benchmark tables. The authors' executable full benchmark code was not available in our tracked sources; our harness is an independent implementation.

## Evidence

The [evidence directory](artifacts/colab_pilot_20/) contains original configuration/summary, pilot summary, all 20 prediction records, comparison CSV and an explanation. It includes answer letters, case hashes, timings and token counts, without raw case facts, personal names or generated reasoning. The original summary still reports the larger run as incomplete.

Downloaded evidence ZIP SHA256: `981e3bb8c8d09be3fa23d2409b99bd4a0782665d4f3bfc5c1267145926b009c5`. The archive was transferred from the actual Colab output and independently verified against all 20 records locally. The executed notebook, including original outputs and the deliberate interruption, was saved to the owner's Google Drive. Its sharing permissions were not changed.

## Zoom recording script

1. Show this public GitHub repository and the pinned notebook/source commit.
2. Show the saved executed notebook's Python/T4 check and dataset-access counts.
3. Explain the modules: data preparation, MCQ construction, model backend, answer parser, scoring and evidence export.
4. Show the pilot export and read the comparison below. Explain that the original interruption was intentional after 20 questions.
5. Show the evidence files. Describe retrieval, drafting and verification agents as the proposed extension, separately from this measured baseline. We have not measured an agentic improvement over this pilot.

Suggested narration:

> We ran a reduced development replication of the ALARB same-regulation article-identification task in Python on a free Google Colab Tesla T4. We used Qwen3-8B with thinking enabled and NF4 quantization. Our fixed 20-question pilot answered 15 correctly, giving 75 percent accuracy. Three unparseable answers were counted as failures. The parent paper reports 84.60 percent on 1,159 questions. Our result is 9.60 percentage points lower, and our cases, prompts, article corpus and precision differ, so we are reporting a working pilot rather than claiming an exact reproduction. We kept final validation untouched. Our proposed next stage is to evaluate a retrieval, drafting and verification agent workflow against a frozen baseline.

Use Zoom's desktop computer recording, share the notebook window, and narrate the actual outputs. A recording script is not the required recording. The 20-question pilot does not by itself establish the assignment's requirement for similar results; instructor acceptance of this reduced protocol or further evaluation is still needed.

## Primary sources

- [Parent paper, Table 5](https://arxiv.org/html/2510.00694v1#S5.T5)
- [Official ALARB dataset](https://huggingface.co/datasets/THIQAH-RD/ALARB)
- [Qwen3-8B model](https://huggingface.co/Qwen/Qwen3-8B)
- [Runnable 20-question notebook](https://colab.research.google.com/github/RAbumansi/week1-alarb-reproduction/blob/alarb-colab-pilot-v1/colab/ALARB_Free_GPU_20_Question_Pilot.ipynb)
