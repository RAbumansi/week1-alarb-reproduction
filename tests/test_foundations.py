"""Phase A guarantees: the holdout stays sealed and the corpus rebuilds correctly.

These tests never open data/validation_holdout/. Only the refusal paths of
load_sealed_validation are exercised, so running the suite leaves no entry in
artifacts/holdout_access_log.json.
"""

from __future__ import annotations

import pytest

from alarb import corpus, data, outcomes


class TestHoldoutGuard:
    @pytest.mark.parametrize("name", ["validation", "holdout", "val", ""])
    def test_load_split_refuses_anything_but_development(self, name: str) -> None:
        with pytest.raises(data.HoldoutSealedError):
            data.load_split(name)

    def test_wrong_token_is_refused(self) -> None:
        with pytest.raises(data.HoldoutSealedError):
            data.load_sealed_validation("please", "final evaluation")

    def test_correct_token_still_requires_a_reason(self) -> None:
        with pytest.raises(data.HoldoutSealedError):
            data.load_sealed_validation(data.FINAL_EVALUATION_TOKEN, "   ")


class TestSplits:
    def test_development_splits_have_the_documented_row_counts(self) -> None:
        assert len(data.load_split("train")) == 9339
        assert len(data.load_split("test")) == 2673

    def test_case_ids_are_unique_and_deterministic(self) -> None:
        cases = data.load_development()
        assert len({case.case_id for case in cases}) == len(cases)
        first = cases[0]
        assert data.case_id(first.case_facts) == first.case_id


class TestCorpus:
    def test_canonical_document_folds_spelling_variants(self) -> None:
        assert corpus.canonical_document("نظام المحكمة التجارية") == corpus.canonical_document(
            "نظام المحاكم التجارية"
        )
        assert corpus.canonical_document("نظام الاثبات") == corpus.canonical_document(
            "نظام الإثبات"
        )

    def test_statute_and_its_regulation_stay_separate(self) -> None:
        statute = corpus.canonical_document("نظام المحاكم التجارية")
        regulation = corpus.canonical_document("اللائحة التنفيذية لنظام المحاكم التجارية")
        assert statute != regulation

    def test_parse_law_entry_reads_document_number_and_text(self) -> None:
        article = corpus.parse_law_entry("نظام الإثبات:17: نص المادة هنا.")
        assert article is not None
        assert article.number == "17"
        assert article.text == "نص المادة هنا."

    @pytest.mark.parametrize("entry", ["", "نظام الإثبات", "نظام الإثبات:17", "نظام الإثبات:17: "])
    def test_parse_law_entry_rejects_incomplete_entries(self, entry: str) -> None:
        assert corpus.parse_law_entry(entry) is None

    def test_corpus_articles_are_unique_and_grouped_by_document(self) -> None:
        articles, stats = corpus.build_corpus(data.load_development())
        assert stats["text_conflicts"] == 0
        assert len(articles) == sum(
            len(items) for items in corpus.articles_by_document(articles).values()
        )

    def test_enough_documents_support_same_statute_distractors(self) -> None:
        articles, _ = corpus.build_corpus(data.load_development())
        grouped = corpus.articles_by_document(articles)
        eligible = [document for document, items in grouped.items() if len(items) >= 4]
        assert len(eligible) >= 9


class TestOutcomes:
    @pytest.mark.parametrize(
        ("verdict", "expected"),
        [
            ("إلزام المدعى عليها بسداد مبلغ 50,000 ريال للمدعية.", "obligation_granted"),
            ("حكمت المحكمة بعدم قبول الدعوى لعدم استيفاء الشروط.", "inadmissible"),
            ("حكمت المحكمة بعدم الاختصاص الولائي بنظر الدعوى.", "no_jurisdiction"),
            ("إثبات الصلح بين الطرفين.", "settlement"),
            ("رفضت المحكمة دعوى المدعي لعدم كفاية البينة.", "claim_rejected"),
        ],
    )
    def test_classify_verdict(self, verdict: str, expected: str) -> None:
        assert outcomes.classify_verdict(verdict) == expected

    def test_distribution_covers_almost_every_case(self) -> None:
        distribution = outcomes.outcome_distribution(data.load_development())
        unclassified = distribution["paper_categories"].get("unclassified", {"share": 0.0})
        assert unclassified["share"] < 0.05

    def test_majority_class_is_near_the_papers_plaintiff_share(self) -> None:
        distribution = outcomes.outcome_distribution(data.load_development())
        majority = distribution["majority_class"]
        assert majority["label"] == "for_plaintiff"
        assert 0.55 <= majority["share"] <= 0.65
