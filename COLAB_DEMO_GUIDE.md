# Free Colab demonstration of the ALARB task

## What this assignment still needs

Our public code and previous automated tests demonstrate coding progress and
data access. They do not establish similarity to the parent paper's model scores.
The assignment additionally needs a completed real-model experiment and a Zoom
recording that accurately explains its comparison with the paper.

This notebook targets the paper's same-regulation article-identification task,
Table 5. It uses Qwen3-8B, with thinking enabled. Correctness is an exact
comparison of four answer choices, so no paid judging model is needed.

The reported paper reference is **84.60% on 1,159 questions**. Our frozen initial
experiment uses 100 reconstructed development questions. We must observe its
actual accuracy before deciding what similarity can be claimed. A small numeric
difference alone cannot establish equivalence across different task construction.

## Open and run

[Open the notebook in Colab](https://colab.research.google.com/github/RAbumansi/week1-alarb-reproduction/blob/alarb-colab-demo-v1/colab/ALARB_Free_GPU_Demo.ipynb)

1. Sign in with your Google account. Optionally save a copy to Drive.
2. Select Runtime > Change runtime type > T4 GPU, if a free GPU is available.
   Do not buy a plan or compute units. If no free GPU is assigned, stop and try
   again later rather than launching the 8B model on the laptop or CPU.
3. Run cells in order. Review the frozen sample size, seed, model revision,
   quantization and token budget before any predictions.
4. Verify dataset access and the three-question real-model smoke test.
5. Run the 100-question experiment. Every answer is checkpointed. If interrupted
   within the same runtime, rerun that cell to resume with identical settings.
6. Inspect the measured score, parse failures, truncations and confidence interval.
   Download the evidence ZIP and save the executed notebook before disconnecting.

The first model load downloads weights to the cloud VM. It needs no API key.
Time and free GPU availability depend on Colab; neither is guaranteed. Runtime
files can disappear after disconnection. No Drive mounting is required.

## Experimental differences to disclose

- Only the publisher training Parquet file is downloaded and hash-verified. The
  9,339/2,673 course development split is recreated from it. The publisher test
  split remains the final sealed validation set and is not accessed here.
- MCQs use the development-derived 693-article corpus, not the paper's full
  statute pool. It can make the questions easier.
- The task instruction and question construction are independent reconstructions;
  the paper does not supply executable ALARB benchmark code in the tracked sources.
- The model uses pinned Qwen3-8B weights with NF4 quantization to fit a free GPU.
  The paper does not specify this precision or its exact decoding settings.
- This is one task and a development sample, not all benchmark tables or the
  paper's supervised fine-tuning experiment.

Keep the seed and sample fixed. Do not select favorable questions, drop parsing
failures, or change settings just to make the percentage match. Unparseable and
truncated outputs count as unsuccessful answers. Incomplete runs are labeled
INCOMPLETE, and fixture results cannot be exported as real-model evidence.

## Suggested Zoom recording sequence

Aim for a short, clear recording after the actual experiment completes.

1. Introduce the parent paper, the chosen MCQ task and Table 5's reported score.
2. Show the public GitHub code and the Colab source commit.
3. Show the Python/GPU checks, verified data access and sample configuration.
4. Explain the modular boundaries: data preparation, question construction,
   model backend, strict answer parser, scoring and evidence export.
5. Show the completed run and the comparison table. State the actual numerator,
   denominator, percentage and all protocol differences.
6. Show the evidence ZIP and explain the later agentic extension separately.

A truthful explanation template, to fill only after real inference:

> We implemented a reduced development replication of ALARB's legal-article
> identification task. We used Qwen3-8B with thinking enabled. The paper reports
> 84.60% on 1,159 same-regulation questions. Our frozen run answered [correct]
> out of [total] correctly, or [accuracy] percent. Our evaluation uses a different
> development sample, a reconstructed corpus and NF4 quantization, so it is not
> an exact numerical reproduction. We have documented these differences.

If the observed result or these protocol differences do not meet the assignment's
requirement, ask the instructor whether this independent reduced replication is
acceptable or whether additional artifacts or a different baseline are required.
Do not record a claim of matched results before resolving that point.

The raw release contains residual personal names. Our notebook displays counts,
hashes and answer letters rather than quoting parties or case text. The evidence
export excludes raw facts and model reasoning.

## Primary sources

- Paper: https://arxiv.org/html/2510.00694v1#S5.T5
- Dataset: https://huggingface.co/datasets/THIQAH-RD/ALARB
- Model: https://huggingface.co/Qwen/Qwen3-8B
- Quantization: https://huggingface.co/docs/transformers/v4.57.1/quantization/bitsandbytes
- Free-runtime limitations: https://research.google.com/colaboratory/faq.html
