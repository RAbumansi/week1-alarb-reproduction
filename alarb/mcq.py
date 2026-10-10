"""Build the two article-identification MCQ variants.

Section 4.2 of the paper: the model sees a case's facts and four articles, one
cited by the court and three distractors. The variants differ in how distractors
are drawn -- at random from the same statute, or as the nearest neighbours of
the correct article by embedding similarity.

One rule is ours. Distractors exclude every article the case cites, not just the
one being asked about. A case citing three articles would otherwise be able to
draw a second genuinely applicable article as a "distractor", making the item
unanswerable. The paper does not say how it handled this.

Construction is deterministic: each case seeds its own generator from its
case_id, so an item is identical whether built alone or in a batch of ten
thousand.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from alarb import embeddings
from alarb.corpus import Article, Corpus, parse_law_entry
from alarb.data import Case
from alarb.tasks import CHOICE_LETTERS


TASK_DIR = Path("data/tasks")
MANIFEST_PATH = Path("artifacts/mcq_manifest.json")
CHOICES_PER_ITEM = 4
DISTRACTORS_PER_ITEM = CHOICES_PER_ITEM - 1
SEED = 42


@dataclass(frozen=True)
class McqItem:
    task: str
    case_id: str
    correct_key: str
    answer: str
    choices: tuple[tuple[str, str], ...]
    choice_keys: tuple[str, ...]
    metadata: dict


def cited_keys(case: Case, corpus: Corpus) -> list[str]:
    """Canonical keys of the articles this case cites, in corpus terms."""
    keys = []
    for entry in case.applicable_laws:
        article = parse_law_entry(entry)
        if article is None:
            continue
        canonical = corpus.resolve(article.key)
        if canonical is not None:
            keys.append(canonical)
    return keys


def _heading(article: Article) -> str:
    return f"{article.document}:{article.number}"


def _assemble(
    task: str,
    case: Case,
    correct: Article,
    distractors: list[Article],
    rng: random.Random,
    metadata: dict,
) -> McqItem:
    ordered = [correct, *distractors]
    rng.shuffle(ordered)
    answer = CHOICE_LETTERS[ordered.index(correct)]
    return McqItem(
        task=task,
        case_id=case.case_id,
        correct_key=correct.key,
        answer=answer,
        choices=tuple((_heading(article), article.text) for article in ordered),
        choice_keys=tuple(article.key for article in ordered),
        metadata=metadata,
    )


def build_mcqs(
    task: str,
    cases: list[Case],
    corpus: Corpus,
    seed: int = SEED,
    embedder_backend: str = "sentence-transformers",
    embedder_model: str = embeddings.DEFAULT_SENTENCE_TRANSFORMER,
) -> tuple[list[McqItem], dict]:
    if task == "mcq_same_statute":
        return _build_same_statute(cases, corpus, seed)
    if task == "mcq_semantic":
        return _build_semantic(cases, corpus, seed, embedder_backend, embedder_model)
    raise ValueError(f"{task!r} is not an MCQ task")


def _eligible(case: Case, corpus: Corpus, rng: random.Random) -> Article | None:
    """Pick the cited article this case will be asked about."""
    candidates = sorted(set(cited_keys(case, corpus)))
    if not candidates:
        return None
    return corpus.articles[rng.choice(candidates)]


def _build_same_statute(
    cases: list[Case], corpus: Corpus, seed: int
) -> tuple[list[McqItem], dict]:
    grouped = corpus.by_document()
    items: list[McqItem] = []
    skipped = {"no_cited_article": 0, "document_too_small": 0}

    for case in cases:
        rng = random.Random(f"{seed}:{case.case_id}")
        correct = _eligible(case, corpus, rng)
        if correct is None:
            skipped["no_cited_article"] += 1
            continue
        excluded = set(cited_keys(case, corpus))
        pool = [a for a in grouped[correct.document] if a.key not in excluded]
        if len(pool) < DISTRACTORS_PER_ITEM:
            skipped["document_too_small"] += 1
            continue
        distractors = rng.sample(pool, DISTRACTORS_PER_ITEM)
        items.append(
            _assemble(
                "mcq_same_statute",
                case,
                correct,
                distractors,
                rng,
                {"document": correct.document, "document_size": len(grouped[correct.document])},
            )
        )
    return items, skipped


def _build_semantic(
    cases: list[Case],
    corpus: Corpus,
    seed: int,
    backend: str,
    model_name: str,
) -> tuple[list[McqItem], dict]:
    keys = sorted(corpus.articles)
    position = {key: index for index, key in enumerate(keys)}
    embedder = embeddings.get_embedder(backend, model_name)
    similarity = embedder.similarity_matrix([corpus.articles[key].text for key in keys])

    items: list[McqItem] = []
    skipped = {"no_cited_article": 0, "too_few_neighbours": 0}

    for case in cases:
        rng = random.Random(f"{seed}:{case.case_id}")
        correct = _eligible(case, corpus, rng)
        if correct is None:
            skipped["no_cited_article"] += 1
            continue
        excluded = set(cited_keys(case, corpus))
        # Ask for extra neighbours so that dropping the case's other cited
        # articles still leaves three.
        wanted = DISTRACTORS_PER_ITEM + len(excluded)
        ranked = embeddings.top_k_similar(similarity, position[correct.key], wanted)
        distractors, scores = [], []
        for index, score in ranked:
            key = keys[index]
            if key in excluded:
                continue
            distractors.append(corpus.articles[key])
            scores.append(round(score, 4))
            if len(distractors) == DISTRACTORS_PER_ITEM:
                break
        if len(distractors) < DISTRACTORS_PER_ITEM:
            skipped["too_few_neighbours"] += 1
            continue
        items.append(
            _assemble(
                "mcq_semantic",
                case,
                correct,
                distractors,
                rng,
                {"embedder": embedder.name, "similarities": scores},
            )
        )
    return items, skipped


def write_mcqs(items: list[McqItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for item in items:
            stream.write(
                json.dumps(
                    {
                        "task": item.task,
                        "case_id": item.case_id,
                        "correct_key": item.correct_key,
                        "answer": item.answer,
                        "choices": [list(choice) for choice in item.choices],
                        "choice_keys": list(item.choice_keys),
                        "metadata": item.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_mcqs(path: Path) -> list[McqItem]:
    items = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            items.append(
                McqItem(
                    task=payload["task"],
                    case_id=payload["case_id"],
                    correct_key=payload["correct_key"],
                    answer=payload["answer"],
                    choices=tuple(tuple(choice) for choice in payload["choices"]),
                    choice_keys=tuple(payload["choice_keys"]),
                    metadata=payload["metadata"],
                )
            )
    return items
