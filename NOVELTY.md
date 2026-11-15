# Week 11 novelty: retrieval-grounded verdict prediction

## The idea

ALARB evaluates verdict prediction in two settings: from the facts alone (task 1),
and from the facts plus the statute articles the court cited (task 2). The second
setting is not available to a real legal assistant, which never knows in advance
which articles the court will rely on. The authors tried to close that gap by
retrieving articles from the case facts with text embeddings, found the results
"extremely poor", and dropped retrieval in favour of multiple-choice questions.

We modified the retrieval block of our pipeline and tested two predictions:

1. Retrieval works once the query is matched against **earlier cases** rather than
   against the statute text. Facts narrate a dispute while articles state abstract
   rules, so matching the two directly compares different registers. Matching a new
   case to similar past cases and reusing the articles their courts cited avoids that.
2. Giving a model retrieved articles, through the paper's own task 2 prompt, improves
   its verdicts over the facts alone and recovers part of the gap to the
   court-cited ceiling.

Both experiments use the development split only. The 1,329-case validation holdout
was not opened.

## Experiment 1: eight retrievers, compared

`python -m alarb.cli retrieval-eval` writes
[`artifacts/retrieval_comparison.json`](artifacts/retrieval_comparison.json).
Every index is built from the 9,339 training cases (654 articles). Queries are the
2,673 development-test cases, of which 2,472 cite at least one article (6,925 gold
citations; 45 of them name an article absent from the training index and count as
misses). Dense methods use `intfloat/multilingual-e5-base`; their neighbour
temperature was chosen on 1,000 held-out training cases, not on the test queries.

| Retriever | Signal | R@1 | R@5 | R@10 | Hit@5 | MRR@10 |
|---|---|---:|---:|---:|---:|---:|
| Most-cited articles | none: same list for every case | .162 | .466 | .583 | .727 | .558 |
| Facts to article text, TF-IDF | lexical, facts vs statute | .013 | .037 | .082 | .089 | .061 |
| Facts to article text, dense | semantic, facts vs statute | .019 | .109 | .181 | .265 | .136 |
| Similar cases' citations, TF-IDF | lexical, facts vs facts | .209 | **.562** | .676 | .799 | .636 |
| Similar cases' citations, dense | semantic, facts vs facts | .197 | .533 | .632 | .775 | .603 |
| Hybrid TF-IDF (agentic prototype) | 25% article + 75% case | **.214** | .561 | .663 | **.805** | **.640** |
| Hybrid dense | 25% article + 75% case | .197 | .535 | .635 | .776 | .603 |
| Fusion of both hybrids | reciprocal-rank fusion | .205 | .556 | **.676** | .794 | .631 |

What this shows:

- **The paper's finding reproduces.** Matching facts to statute text retrieves a
  correct article in the top five for under 4% of citations lexically, and about 11%
  with a multilingual embedding model.
- **Prediction 1 holds.** Matching against similar past cases raises recall@5 to
  56%, fifteen times the lexical facts-to-statute approach.
- **But a trivial baseline is close behind.** Returning the five most-cited articles
  for every case already reaches 46.6%. Case-based retrieval adds about 10 points
  over a list that ignores the case entirely, because article choice in these cases
  is dominated by a handful of procedural provisions.
- **No new method beats the prototype's lexical hybrid.** The `hybrid-tfidf` row
  reproduces `alarb/agentic.py` exactly (200 of 200 identical top-10 lists in a
  parity check). Dense embeddings help facts-to-statute matching but slightly hurt
  case matching. Sweeping the neighbour count (10 to 100) and a popularity prior
  (0 to 0.4) on held-out training cases moved recall@5 by at most 3 points, and the
  prototype's setting was already the best.

## Experiment 2: do retrieved articles help the model's verdicts?

[`colab/ALARB_Week11_Retrieval_Novelty.ipynb`](colab/ALARB_Week11_Retrieval_Novelty.ipynb)
ran Qwen3-8B (thinking on, NF4 4-bit) on a free Google Colab Tesla T4, code commit
`8aff291`. The same 20 development-test cases, sampled in proportion to their outcomes
from cases citing at least one article, were run three ways. The two article conditions
use the paper's Figure 10 prompt word for word. Every verdict was graded by Qwen3-8B
(thinking off) with the paper's Appendix B.1 judge prompt. Results are in
[`artifacts/week11_colab/summary.json`](artifacts/week11_colab/summary.json): case
hashes, labels and counts only.

| Condition | Correct | Partially | Incorrect | Correct or partial (95% CI) | Mean grade | vs facts only |
|---|---:|---:|---:|---:|---:|---|
| Facts only (paper task 1) | 15% | 45% | 40% | 60% (39–78%) | 0.75 | |
| Facts + 5 most-cited articles (control) | 10% | 55% | 35% | 65% (43–82%) | 0.75 | 1 better, 1 worse |
| **Facts + 5 retrieved articles (ours)** | **15%** | **55%** | **30%** | **70% (48–85%)** | **0.85** | **3 better, 1 worse (p = 0.63)** |

Mean grade counts correct as 2, partially correct as 1 and incorrect as 0. No output was
truncated or unparseable. On these 20 cases the retriever's recall@5 was 45%, against
41% for the most-cited list.

Retrieved articles produced the best scores: the fewest incorrect verdicts and the
highest mean grade, with three cases improving and one getting worse against facts
alone. Generic legal text did not help in the same way: the most-cited control left
the mean grade unchanged. The direction supports prediction 2, but with 20 cases the
difference is **not statistically significant** (sign test p = 0.63), and the
confidence intervals overlap.

### Limits

- **Small sample.** 20 cases per condition on a free GPU. A larger run is needed
  before claiming an effect.
- **Same model as generator and judge.** Any self-preference applies equally to every
  condition, so it should not favour one, but the judge has not yet been validated
  against human labels (`python -m alarb.cli agreement`).
- **The court-cited ceiling (paper task 2) is missing.** The Colab runtime disconnected
  during that condition; it was not re-run before the deadline.
- **Outcome agreement is uninformative here.** The judge-free keyword check left most
  of Qwen's verdicts unclassified because they are phrased differently from court
  verdicts, so it scored 26% in every condition.
- **Not comparable to the paper's tables.** Different cases, a 654-article index instead
  of the full statute corpus, NF4 precision, and a different judge (see DEVIATIONS.md).

## Next steps

Run the court-cited ceiling and a larger sample, validate the judge against human
labels, and add the citation-checking step from the agentic prototype on top of the
retrieved articles.
