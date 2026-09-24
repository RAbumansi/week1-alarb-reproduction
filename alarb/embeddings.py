"""Article similarity, used to build the semantically-related MCQ distractors.

The paper embeds articles with OpenAI's text-embedding-3-large and ranks by
cosine similarity. That model is unavailable here, so this offers two local
substitutes. Both expose the same thing: a dense article-by-article cosine
similarity matrix. The corpus is ~700 articles, so the matrix is small.

  sentence-transformers  a multilingual embedding model; the closer analogue,
                         and the default. Needs torch.
  tfidf                  character n-gram TF-IDF. No heavy dependencies, and
                         enough to exercise the pipeline, but it matches on
                         shared wording rather than meaning, so its distractors
                         are weaker. Recorded in the manifest when used.
"""

from __future__ import annotations

import numpy as np


DEFAULT_SENTENCE_TRANSFORMER = "intfloat/multilingual-e5-base"
BACKENDS = ("sentence-transformers", "tfidf")


class Embedder:
    name: str

    def similarity_matrix(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = DEFAULT_SENTENCE_TRANSFORMER) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = f"sentence-transformers:{model_name}"
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    def similarity_matrix(self, texts: list[str]) -> np.ndarray:
        # e5 models expect a prefix; "query:" on both sides is their guidance
        # for symmetric similarity, which is what article-to-article ranking is.
        prepared = [f"query: {text}" for text in texts] if "e5" in self._model_name else texts
        vectors = self._model.encode(
            prepared, normalize_embeddings=True, show_progress_bar=False, batch_size=32
        )
        return np.asarray(vectors) @ np.asarray(vectors).T


class TfidfEmbedder(Embedder):
    def __init__(self) -> None:
        self.name = "tfidf:char_wb(2,4)"

    def similarity_matrix(self, texts: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2).fit_transform(
            texts
        )
        return cosine_similarity(vectors)


def get_embedder(backend: str, model_name: str = DEFAULT_SENTENCE_TRANSFORMER) -> Embedder:
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    if backend == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"Unknown embedder backend {backend!r}; expected one of {BACKENDS}")


def top_k_similar(similarity: np.ndarray, index: int, k: int) -> list[tuple[int, float]]:
    """The k most similar entries to `index`, itself excluded, most similar first."""
    scores = similarity[index].copy()
    scores[index] = -np.inf
    ranked = np.argsort(-scores)[:k]
    return [(int(position), float(scores[position])) for position in ranked]
