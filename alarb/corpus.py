"""Reconstruct the statute article corpus from the cases themselves.

The paper's article-identification tasks need a pool of statute articles to draw
distractors from. The authors scraped that pool from the Ministry of Justice but
never released it. It is recoverable anyway: every applicable_laws entry is
formatted "{document}:{article}: {full article text}", so deduplicating those
entries across the development cases rebuilds the corpus.

Document names are spelled inconsistently across cases (hamza variants, singular
vs plural, "lai'hah" vs "executive regulation"). Those variants are folded to
canonical names so that same-statute distractors are drawn from the right pool.
A statute and its executive regulation are deliberately kept apart: they are
different documents with independent article numbering.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from alarb.data import Case, load_development


CORPUS_PATH = Path("artifacts/article_corpus.json")

_DOCUMENT_ALIASES = {
    "نظام المحكمة التجارية": "نظام المحاكم التجارية",
    "لائحة نظام المحاكم التجارية": "اللائحة التنفيذية لنظام المحاكم التجارية",
    "نظام المرافعات": "نظام المرافعات الشرعية",
    "اللائحة التنفيذية لنظام المرافعات": "اللائحة التنفيذية لنظام المرافعات الشرعية",
    "اللوائح التنفيذية لنظام المرافعات الشرعية": "اللائحة التنفيذية لنظام المرافعات الشرعية",
    "لائحة نظام المرافعات الشرعية": "اللائحة التنفيذية لنظام المرافعات الشرعية",
}


@dataclass(frozen=True)
class Article:
    key: str
    document: str
    number: str
    text: str


@dataclass(frozen=True)
class Corpus:
    articles: dict[str, Article]
    aliases: dict[str, str]

    def resolve(self, key: str) -> str | None:
        """Canonical key for a cited article, or None if it is not in the corpus."""
        if key in self.articles:
            return key
        return self.aliases.get(key)

    def by_document(self) -> dict[str, list[Article]]:
        return articles_by_document(self.articles)


def normalize_arabic(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[أإآٱ]", "ا", value)).strip()


def canonical_document(document: str) -> str:
    normalized = normalize_arabic(document)
    aliases = {normalize_arabic(k): normalize_arabic(v) for k, v in _DOCUMENT_ALIASES.items()}
    return aliases.get(normalized, normalized)


def parse_law_entry(entry: str) -> Article | None:
    """Split "{document}:{article}: {text}". Returns None if any part is missing."""
    parts = entry.split(":", 2)
    if len(parts) < 3:
        return None
    document = canonical_document(parts[0])
    number = normalize_arabic(parts[1])
    text = parts[2].strip()
    if not document or not number or not text:
        return None
    return Article(key=f"{document}:{number}", document=document, number=number, text=text)


def _collapse_duplicate_texts(
    articles: dict[str, Article],
) -> tuple[dict[str, Article], dict[str, str]]:
    """Fold keys whose article text is byte-identical into one.

    Sub-article numbers are cited both ways round across cases -- "1/29" in one
    and "29/1" in another for the same provision. Rather than guess which half
    is the article and which the paragraph, treat identical text as proof the
    two keys denote one article and keep the first by sort order. Without this
    a multiple-choice item can offer the same article twice.
    """
    grouped: dict[str, list[str]] = {}
    for key in sorted(articles):
        grouped.setdefault(articles[key].text, []).append(key)

    kept: dict[str, Article] = {}
    aliases: dict[str, str] = {}
    for keys in grouped.values():
        canonical, *duplicates = keys
        kept[canonical] = articles[canonical]
        for duplicate in duplicates:
            aliases[duplicate] = canonical
    return kept, aliases


def build_corpus(cases: list[Case]) -> tuple[Corpus, dict[str, int]]:
    articles: dict[str, Article] = {}
    stats = {"entries_seen": 0, "entries_unparsed": 0, "text_conflicts": 0}
    for case in cases:
        for entry in case.applicable_laws:
            stats["entries_seen"] += 1
            article = parse_law_entry(entry)
            if article is None:
                stats["entries_unparsed"] += 1
                continue
            existing = articles.get(article.key)
            if existing is None:
                articles[article.key] = article
            elif existing.text != article.text:
                # Keep the longest rendering; a shorter one is a truncation.
                stats["text_conflicts"] += 1
                if len(article.text) > len(existing.text):
                    articles[article.key] = article

    kept, aliases = _collapse_duplicate_texts(articles)
    stats["duplicate_text_keys_merged"] = len(aliases)
    return Corpus(articles=kept, aliases=aliases), stats


def articles_by_document(articles: dict[str, Article]) -> dict[str, list[Article]]:
    grouped: dict[str, list[Article]] = {}
    for article in articles.values():
        grouped.setdefault(article.document, []).append(article)
    for items in grouped.values():
        items.sort(key=lambda article: article.key)
    return grouped


def load_corpus(path: Path = CORPUS_PATH) -> Corpus:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Run: python -m alarb.cli corpus")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Corpus(
        articles={item["key"]: Article(**item) for item in payload["articles"]},
        aliases=payload["aliases"],
    )


def write_corpus(path: Path = CORPUS_PATH) -> Corpus:
    cases = load_development()
    corpus, stats = build_corpus(cases)
    grouped = corpus.by_document()

    payload = {
        "source": {
            "scope": "data/development (the publisher's training split)",
            "cases": len(cases),
            "note": (
                "Rebuilt from applicable_laws entries. The sealed validation "
                "holdout is not read, so articles cited only by holdout cases "
                "are absent; distractors never need them."
            ),
        },
        "counts": {
            "articles": len(corpus.articles),
            "documents": len(grouped),
            **stats,
        },
        "documents": {
            document: len(items)
            for document, items in sorted(grouped.items(), key=lambda kv: -len(kv[1]))
        },
        "aliases": corpus.aliases,
        "articles": [
            asdict(article) for article in sorted(corpus.articles.values(), key=lambda a: a.key)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return corpus
