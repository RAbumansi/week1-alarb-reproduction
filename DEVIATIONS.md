# Deviations from the parent paper

This harness is an independent reimplementation of the tasks in *ALARB: An
Arabic Legal Argument Reasoning Benchmark* ([arXiv:2510.00694](https://arxiv.org/abs/2510.00694)).
The authors released the dataset but no benchmark code, so nothing here is a
port of theirs. Every point below is a place where our implementation and the
paper differ, and why.

**Because of these differences, numbers produced by this harness are not
directly comparable to the paper's tables.** They are internally consistent and
can be compared across models and tasks within this project.

## 1. There is no official code to reproduce

The paper links no repository. Its only artifact link is a Hugging Face dataset
(`HarethahMo/ALARB`, now redirecting to `THIQAH-RD/ALARB`), which holds two
parquet files, a README and `.gitattributes` — no code. The THIQAH GitHub
organisation exposes `ArabLegalData` and `ArabLegalEval`; the latter belongs to
the authors' *earlier* ArabLegalEval paper, carries no licence, and must not be
described as ALARB's implementation.

What made reimplementation possible is that Appendices B.1 and B.2 print four
prompts in full.

## 2. Prompts: four transcribed, two invented

| Task | Source |
|---|---|
| Verdict from facts | Appendix B.2, Figure 8 — transcribed |
| Verdict from facts + articles | Appendix B.2, Figure 10 — transcribed |
| Verdict from facts + reasoning | Appendix B.2, Figure 9 — transcribed |
| LLM-as-a-judge | Appendix B.1, Figure 7 — transcribed |
| **Argument completion** | **No published prompt.** Section 4.1 describes the task in prose only; ours follows Figure 9's wording |
| **Article identification (MCQ)** | **No published prompt.** Section 4.2 describes the task and Figure 11 shows a rendered item, but the instruction text is not given |

Transcriptions keep the paper's own inconsistencies: the backslash closing tags
(`[\VERDICT]`), and Figure 9 ending `Begin` where Figures 8 and 10 end `Begin!`.

One correction was made. Figure 10 reads "Your verdict and reasoning should be
strictly in language", with no braces, where Figures 8 and 9 have `{language}`.
Read as a typesetting slip and restored to `{language}`; passing it literally
would leave the output language unspecified for that task alone.

## 3. Judge: Claude, not GPT-4o

The paper grades generated verdicts with GPT-4o. This project has Anthropic
credentials only, so the Appendix B.1 prompt is run unchanged against a Claude
model. Judge choice moves absolute scores, so this alone breaks comparability
with the paper's tables.

A further caution: when the judged model is also a Claude model, generator and
judge share a family, and self-preference is a known failure mode of
LLM-as-a-judge. The judge model is configured separately from the generation
model and recorded in every run's `config.json` so the overlap is at least
visible. The paper has the same weakness — GPT-4o judging GPT-4o among others —
and does not address it.

An offline word-overlap judge (`--judge heuristic`) also exists. It is **not**
the paper's metric, compares wording rather than meaning, and is there to
exercise the pipeline without credentials and to give the LLM judge something
to be measured against.

## 4. Embeddings: a local model, not text-embedding-3-large

Semantic distractors are ranked with OpenAI's `text-embedding-3-large` in the
paper. Anthropic publishes no embeddings API, so the default here is
`intfloat/multilingual-e5-base`, run locally. A character n-gram TF-IDF backend
(`--embedder tfidf`) builds the same tasks without torch, at lower quality.

Cosine values are **not** comparable between the two. The paper's Figure 11
sample shows distractor similarities of 0.596, 0.584 and 0.578; ours run 0.865
to 0.990 (mean 0.924). That is a property of how each model scales its vector
space, not evidence that our distractors are closer. Only the ranking is used.

## 5. Article corpus: recovered from the cases, not scraped

The paper scraped eight statutes and their implementing regulations from the
Saudi Ministry of Justice. That corpus was never released. Its Table 1 totals
roughly 2,400 articles.

It turns out to be partly recoverable: each `applicable_laws` entry is
formatted `{document}:{article}: {full article text}`, so deduplicating those
entries across the 12,012 development cases yields **693 articles across 11
canonical documents** — only the articles some case actually cites.

The consequence is that our distractor pool is roughly a quarter the size of
the paper's, so the article-identification tasks are **easier here than in the
paper**, particularly the same-statute variant.

Two normalisations were needed, neither described in the paper:

- **Document names.** The same statute is spelled 18 different ways across
  cases (hamza variants, singular/plural, "regulation of" vs "executive
  regulation of"). These fold to 11 canonical documents. A statute and its
  executive regulation are deliberately kept apart — different documents with
  independent article numbering.
- **Transposed sub-article numbers.** The same provision is cited as both
  `1/29` and `29/1`. Rather than guess which half is the paragraph, identical
  article text is taken as proof the two keys are one article; 9 such pairs
  were merged.

## 6. Distractor rule the paper does not state

Distractors exclude **every** article the case cites, not only the one being
asked about. A case citing three articles could otherwise draw a second
genuinely applicable article as a "distractor", making the item unanswerable.
The paper does not say how it handled this.

## 7. Evaluation set: our development test split

The paper evaluates on the publisher's 1,329-case test split, using 1,329 cases
for verdict tasks and 1,159 MCQs per variant.

Under this project's split policy (see `DATA_SPLIT_POLICY.md`) that split is
the **sealed validation holdout** and stays closed until the pipeline is
frozen. Development results therefore come from our own 2,673-case development
test split, which yields 2,472 items per MCQ variant. Different cases, so
different numbers.

## 8. Decoding parameters are ours

The paper never states temperature, top-p, max tokens, random seeds, model
snapshot dates or inference runtime versions. Defaults here are temperature 0.0
and 1024 max tokens, recorded in each run's `config.json`. Exact numerical
replication was never possible from the published material.

## 9. Outcome taxonomy is inferred

Table 3 reports a verdict breakdown — 62% for plaintiff, 5% for defendant, 33%
court dismissal — but the dataset ships no outcome labels. `alarb/outcomes.py`
recovers them with a keyword heuristic, giving 59.1% / 6.9% / 23.9%, plus 7.7%
settlements and 2.4% unclassified. Settlements have no home in the paper's
three categories, and dismissal plus settlement plus unclassified totals ~34%,
close to the paper's 33% — suggesting their dismissal bucket absorbs
settlements. This is a heuristic, not ground truth, and is used only for the
majority-class reference point and for stratified sampling.

## What did reproduce exactly

Established in Week 1 and unchanged here: the official loader runs, file
hashes match their Git LFS pointers, the schema is as documented, and the split
counts are 12,012 train / 1,329 test. The public dataset holds 13,341 cases
against the paper's stated 13,344, and a few of its summary statistics differ
slightly — see `EXECUTION_REPORT.md`.
