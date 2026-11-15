"""Compare ways of finding the statute articles a case needs.

ALARB's authors embedded every article and each case's facts, matched them by
cosine similarity, called the results "extremely poor", and replaced retrieval
with multiple choice. One likely reason is that facts and statutes are written
in different registers: facts narrate a dispute, articles state abstract rules.
Matching a new case to earlier cases that read like it, and reusing the articles
their courts cited, sidesteps that gap. This module ranks articles both ways,
lexically and with dense embeddings, and fuses them.

Every index is built from the training split only. Development-test cases are
the queries; their gold citations are read only to score a ranking, never to
produce one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from alarb import corpus, data
from alarb.agentic import Hit

E5_MODEL = "intfloat/multilingual-e5-base"
EMBED_CACHE = Path(".cache/embeddings")
RESULTS_PATH = Path("artifacts/retrieval_comparison.json")
NEIGHBOURS = 30
RRF_K = 60
KS = (1, 3, 5, 10)
DENSE_TEMPERATURES = (0.01, 0.02, 0.05, 0.1)
DEFAULT_TEMPERATURE = 0.02

# (texts, e5 prefix) -> one L2-normalised row per text
Encoder = Callable[[list[str], str], np.ndarray]


@dataclass(frozen=True)
class Method:
    name: str
    label: str
    signal: str


METHODS = (
    Method("popularity", "Most-cited articles", "none: the same list for every case"),
    Method("article-tfidf", "Facts to article text, TF-IDF", "lexical, facts vs statute"),
    Method("article-dense", "Facts to article text, dense", "semantic, facts vs statute"),
    Method("case-tfidf", "Similar cases' citations, TF-IDF", "lexical, facts vs facts"),
    Method("case-dense", "Similar cases' citations, dense", "semantic, facts vs facts"),
    Method("hybrid-tfidf", "Hybrid TF-IDF (agentic prototype)", "25% article + 75% case, lexical"),
    Method("hybrid-dense", "Hybrid dense", "25% article + 75% case, semantic"),
    Method("fusion", "Fusion of both hybrids", "reciprocal-rank fusion"),
)
METHOD_NAMES = tuple(method.name for method in METHODS)


def facts_text(facts: tuple[str, ...] | list[str]) -> str:
    return corpus.normalize_arabic("\n".join(facts))


def e5_encoder(model_name: str = E5_MODEL, cache_dir: Path = EMBED_CACHE) -> Encoder:
    """Encode with multilingual-e5. Batches are cached on disk by content, and
    every vector is remembered for the process, so a subset of texts already
    encoded (a tuning split, one query) costs nothing."""
    loaded: dict = {}
    memory: dict[tuple[str, str], np.ndarray] = {}

    def encode(texts: list[str], prefix: str) -> np.ndarray:
        missing = list(dict.fromkeys(t for t in texts if (prefix, t) not in memory))
        if missing:
            digest = hashlib.sha256(
                json.dumps([model_name, prefix, missing], ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            path = cache_dir / f"{digest}.npy"
            if path.is_file():
                vectors = np.load(path)
            else:
                if "model" not in loaded:
                    from sentence_transformers import SentenceTransformer

                    loaded["model"] = SentenceTransformer(model_name)
                vectors = loaded["model"].encode(
                    [f"{prefix}: {text}" for text in missing],
                    batch_size=32,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ).astype(np.float32)
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(path, vectors)
            memory.update({(prefix, text): vector for text, vector in zip(missing, vectors)})
        return np.stack([memory[(prefix, text)] for text in texts])

    return encode


def _row_max_normalise(matrix: np.ndarray) -> np.ndarray:
    peak = matrix.max(axis=1, keepdims=True)
    return np.divide(matrix, peak, out=np.zeros_like(matrix), where=peak > 0)


def _ranks(scores: np.ndarray) -> np.ndarray:
    """0-based rank of every article in each row, best first, ties stable."""
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(scores.shape[1])[None, :]
    return ranks


class Retrieval:
    """Article rankings for case facts, from a training-only index."""

    def __init__(
        self,
        training_cases: list[data.Case],
        encoder: Encoder | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        neighbours: int = NEIGHBOURS,
    ):
        self.corpus, _ = corpus.build_corpus(training_cases)
        self.articles = sorted(self.corpus.articles.values(), key=lambda article: article.key)
        if not self.articles:
            raise ValueError("Training cases contain no usable statutory articles.")
        self.position = {article.key: index for index, article in enumerate(self.articles)}
        self.temperature = temperature
        self.neighbours = neighbours
        self._encoder = encoder
        self._dense_cache: dict[str, np.ndarray] = {}

        self.case_articles = [sorted(self._cited_positions(case)) for case in training_cases]
        rows, cols, vals = [], [], []
        for row, positions in enumerate(self.case_articles):
            for col in positions:
                rows.append(row)
                cols.append(col)
                vals.append(1.0 / len(positions))
        # Case-to-article transfer matrix: each case spreads unit weight over its citations.
        self.transfer = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(len(training_cases), len(self.articles))
        )
        self.popularity = np.bincount(
            [col for positions in self.case_articles for col in positions],
            minlength=len(self.articles),
        ).astype(np.float64)

        # The same construction as alarb.agentic.ArticleRetriever, so "hybrid-tfidf"
        # reproduces the prototype exactly.
        self.article_texts = [
            corpus.normalize_arabic(f"{article.document} {article.text}") for article in self.articles
        ]
        self.case_texts = [facts_text(case.case_facts) for case in training_cases]
        self.article_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        self.article_vectors = self.article_vectorizer.fit_transform(self.article_texts)
        self.fact_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        self.fact_vectors = self.fact_vectorizer.fit_transform(self.case_texts)

    def _cited_positions(self, case: data.Case) -> set[int]:
        positions = set()
        for law in case.applicable_laws:
            article = corpus.parse_law_entry(law)
            if article is None:
                continue
            key = self.corpus.resolve(article.key)
            if key is not None:
                positions.add(self.position[key])
        return positions

    def gold(self, case: data.Case) -> tuple[set[int], int]:
        """Positions of the case's cited articles found in the index, and how many
        it cites in total. Citations missing from the training corpus still count,
        as misses, so no method is rewarded for an article it could never return."""
        cited, positions = set(), set()
        for law in case.applicable_laws:
            article = corpus.parse_law_entry(law)
            if article is None:
                continue
            cited.add(article.key)
            key = self.corpus.resolve(article.key)
            if key is not None:
                positions.add(self.position[key])
        unknown = sum(1 for key in cited if self.corpus.resolve(key) is None)
        return positions, len(positions) + unknown

    # -- similarity components ------------------------------------------------
    def _encoder_or_fail(self) -> Encoder:
        if self._encoder is None:
            self._encoder = e5_encoder()
        return self._encoder

    def _dense(self, name: str, texts: list[str], prefix: str) -> np.ndarray:
        if name not in self._dense_cache:
            self._dense_cache[name] = self._encoder_or_fail()(texts, prefix)
        return self._dense_cache[name]

    def _transfer(self, similarity: np.ndarray, dense: bool) -> np.ndarray:
        """Spread each query's nearest training cases' citations onto articles."""
        n_queries, n_cases = similarity.shape
        keep = min(self.neighbours, n_cases)
        top = np.argpartition(-similarity, keep - 1, axis=1)[:, :keep]
        sims = np.take_along_axis(similarity, top, axis=1)
        if dense:
            # Dense cosines crowd into a narrow band near 1, so a softmax with a
            # small temperature is needed to tell close neighbours from far ones.
            weights = np.exp((sims - sims.max(axis=1, keepdims=True)) / self.temperature)
        else:
            weights = np.where(sims > 0, sims, 0.0) ** 3
        rows = np.repeat(np.arange(n_queries), keep)
        neighbours = sparse.csr_matrix(
            (weights.ravel(), (rows, top.ravel())), shape=(n_queries, n_cases)
        )
        return np.asarray((neighbours @ self.transfer).todense())

    def scores(self, method: str, query_texts: list[str]) -> np.ndarray:
        """One row of article scores per query; higher is better."""
        if method not in METHOD_NAMES:
            raise ValueError(f"Unknown retrieval method {method!r}; expected one of {METHOD_NAMES}")
        if method == "popularity":
            return np.tile(self.popularity, (len(query_texts), 1))
        if method == "article-tfidf":
            return (self.article_vectorizer.transform(query_texts) @ self.article_vectors.T).toarray()
        if method == "case-tfidf":
            similarity = (self.fact_vectorizer.transform(query_texts) @ self.fact_vectors.T).toarray()
            return self._transfer(similarity, dense=False)
        if method == "hybrid-tfidf":
            return 0.25 * _row_max_normalise(self.scores("article-tfidf", query_texts)) + 0.75 * _row_max_normalise(
                self.scores("case-tfidf", query_texts)
            )
        if method in ("article-dense", "case-dense"):
            queries = self._encoder_or_fail()(query_texts, "query")
            if method == "article-dense":
                return queries @ self._dense("articles", self.article_texts, "passage").T
            return self._transfer(queries @ self._dense("cases", self.case_texts, "query").T, dense=True)
        if method == "hybrid-dense":
            return 0.25 * _row_max_normalise(self.scores("article-dense", query_texts)) + 0.75 * _row_max_normalise(
                self.scores("case-dense", query_texts)
            )
        # fusion: reciprocal-rank fusion of the two hybrids
        fused = np.zeros((len(query_texts), len(self.articles)))
        for part in ("hybrid-tfidf", "hybrid-dense"):
            fused += 1.0 / (RRF_K + 1 + _ranks(self.scores(part, query_texts)))
        return fused

    def retrieve_batch(
        self, method: str, facts_list: list[tuple[str, ...]], top_k: int = 5
    ) -> list[list[Hit]]:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        matrix = self.scores(method, [facts_text(facts) for facts in facts_list])
        results = []
        for row in matrix:
            order = np.argsort(-row, kind="stable")[:top_k]
            results.append(
                [
                    Hit(self.articles[i].key, self.articles[i].text, round(float(row[i]), 6))
                    for i in order
                    if row[i] > 0
                ]
            )
        return results

    def retrieve(self, facts: tuple[str, ...], top_k: int = 5, method: str = "fusion") -> list[Hit]:
        return self.retrieve_batch(method, [facts], top_k)[0]


# -- evaluation ---------------------------------------------------------------
def rank_metrics(
    scores: np.ndarray, gold: list[tuple[set[int], int]], ks: tuple[int, ...] = KS
) -> dict:
    """Macro recall@k, hit@5 and MRR@10 over queries citing at least one article."""
    order = np.argsort(-scores, axis=1, kind="stable")[:, : max(ks)]
    recall = {k: [] for k in ks}
    hits5, reciprocal = [], []
    for row, (positions, total) in zip(order, gold):
        if total == 0:
            continue
        ranked = list(row)
        for k in ks:
            recall[k].append(len(positions & set(ranked[:k])) / total)
        hits5.append(float(bool(positions & set(ranked[:5]))))
        first = next((i for i, pos in enumerate(ranked) if pos in positions), None)
        reciprocal.append(0.0 if first is None else 1.0 / (first + 1))
    n = len(reciprocal)
    if n == 0:
        raise ValueError("No query cites an article, so recall is undefined.")
    out = {"queries": n}
    out.update({f"recall@{k}": round(float(np.mean(recall[k])), 4) for k in ks})
    out["hit@5"] = round(float(np.mean(hits5)), 4)
    out["mrr@10"] = round(float(np.mean(reciprocal)), 4)
    return out


def tune_temperature(
    training_cases: list[data.Case], encoder: Encoder, holdout: int = 1000, seed: int = 42
) -> dict:
    """Pick the dense neighbour temperature on training cases alone: hold some
    training cases out as queries against an index of the rest."""
    import random

    shuffled = sorted(training_cases, key=lambda case: case.case_id)
    random.Random(seed).shuffle(shuffled)
    queries, rest = shuffled[:holdout], shuffled[holdout:]
    results = {}
    for temperature in DENSE_TEMPERATURES:
        retrieval = Retrieval(rest, encoder=encoder, temperature=temperature)
        gold = [retrieval.gold(case) for case in queries]
        texts = [facts_text(case.case_facts) for case in queries]
        results[str(temperature)] = rank_metrics(retrieval.scores("case-dense", texts), gold)
    best = max(results, key=lambda t: (results[t]["recall@5"], -float(t)))
    return {"holdout_queries": len(queries), "index_cases": len(rest), "by_temperature": results,
            "chosen": float(best)}


def sweep_case_tfidf(
    training_cases: list[data.Case],
    neighbours: tuple[int, ...] = (10, 30, 60, 100),
    priors: tuple[float, ...] = (0.0, 0.1, 0.2, 0.4),
    holdout: int = 1000,
    seed: int = 42,
) -> dict:
    """Check, on training cases alone, whether more or fewer neighbours or a
    popularity prior would lift the lexical case-transfer retriever."""
    import random

    shuffled = sorted(training_cases, key=lambda case: case.case_id)
    random.Random(seed).shuffle(shuffled)
    queries, rest = shuffled[:holdout], shuffled[holdout:]
    retrieval = Retrieval(rest)
    gold = [retrieval.gold(case) for case in queries]
    texts = [facts_text(case.case_facts) for case in queries]
    similarity = (retrieval.fact_vectorizer.transform(texts) @ retrieval.fact_vectors.T).toarray()
    prior = retrieval.popularity / retrieval.popularity.max()
    grid = []
    for k in neighbours:
        retrieval.neighbours = k
        transfer = _row_max_normalise(retrieval._transfer(similarity, dense=False))
        for weight in priors:
            grid.append({"neighbours": k, "popularity_prior": weight,
                         **rank_metrics(transfer + weight * prior, gold)})
    best = max(grid, key=lambda row: row["recall@5"])
    return {"holdout_queries": len(queries), "grid": grid,
            "best_by_recall@5": {"neighbours": best["neighbours"], "popularity_prior": best["popularity_prior"]}}


def evaluate(output: Path = RESULTS_PATH, encoder: Encoder | None = None) -> dict:
    """Tune on training cases, then score every method once on the development
    test split. The sealed validation holdout is never opened."""
    training, queries = data.load_split("train"), data.load_split("test")
    encoder = encoder or e5_encoder()
    encoder([facts_text(case.case_facts) for case in training], "query")
    tuning = tune_temperature(training, encoder)
    result = compare(training, queries, encoder=encoder, temperature=tuning["chosen"])
    result["dense_temperature_tuning"] = tuning
    result["case_tfidf_sweep"] = sweep_case_tfidf(training)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def compare(
    training_cases: list[data.Case],
    query_cases: list[data.Case],
    encoder: Encoder | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    methods: tuple[str, ...] = METHOD_NAMES,
) -> dict:
    retrieval = Retrieval(training_cases, encoder=encoder, temperature=temperature)
    gold = [retrieval.gold(case) for case in query_cases]
    texts = [facts_text(case.case_facts) for case in query_cases]
    rows = []
    for method in METHODS:
        if method.name not in methods:
            continue
        metrics = rank_metrics(retrieval.scores(method.name, texts), gold)
        rows.append({"method": method.name, "label": method.label, "signal": method.signal, **metrics})
    cited = sum(total for _, total in gold)
    indexed = sum(len(positions) for positions, _ in gold)
    return {
        "index": {
            "source": "training split only",
            "training_cases": len(training_cases),
            "articles": len(retrieval.articles),
        },
        "queries": {
            "source": "development test split",
            "cases": len(query_cases),
            "citing_at_least_one_article": rows[0]["queries"] if rows else 0,
            "gold_citations": cited,
            "gold_citations_absent_from_index": cited - indexed,
        },
        "settings": {"neighbours": NEIGHBOURS, "dense_model": E5_MODEL,
                     "dense_temperature": temperature, "rrf_k": RRF_K},
        "results": rows,
    }
