"""Retrieval comparison, the retrieval-augmented verdict task and paired scoring.

Synthetic cases only, so these run without the dataset (and in CI).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from alarb import agentic, comparison, data, retrieval, runner, tasks

LAW_A = "نظام المحاكم التجارية:16: تختص المحكمة بالنظر في المنازعات التجارية بين التجار"
LAW_B = "نظام الإثبات:17: الإقرار القضائي حجة قاطعة على المقر"
LAW_C = "نظام الشركات:8: يجب أن يكون عقد الشركة مكتوبا"
LAW_D = "نظام الإفلاس:2: تسري أحكام النظام على التاجر المتعثر"


def case(facts: str, laws: tuple[str, ...], verdict: str = "إلزام المدعى عليه بالسداد.") -> data.Case:
    return data.Case(
        case_id=data.case_id((facts,)),
        case_facts=(facts,),
        court_reasoning=("سبب",),
        applicable_laws=laws,
        verdict=verdict,
    )


TRAINING = [
    case("مطالبة بمبلغ توريد بضاعة بين تاجرين في جدة", (LAW_A,)),
    case("مطالبة بمبلغ توريد مواد بناء بين تاجرين في الرياض", (LAW_A, LAW_B)),
    case("إقرار المدعى عليه بالدين أمام المحكمة وطلب السداد", (LAW_B,)),
    case("نزاع بين شركاء حول عقد شركة غير مكتوب", (LAW_C,)),
    case("طلب افتتاح إجراء إفلاس لتاجر متعثر عن السداد", (LAW_D,)),
    case("مطالبة بمبلغ توريد أجهزة بين تاجرين", (LAW_A,)),
]


def bag_encoder(texts: list[str], prefix: str) -> np.ndarray:
    """A deterministic stand-in for e5: hashed bag of words, L2-normalised."""
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in text.split():
            out[row, sum(map(ord, word)) % 64] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms > 0, norms, 1.0)


@pytest.fixture(scope="module")
def index() -> retrieval.Retrieval:
    return retrieval.Retrieval(TRAINING, encoder=bag_encoder, temperature=0.05)


class TestRetrieval:
    def test_hybrid_tfidf_reproduces_the_agentic_prototype(self, index) -> None:
        prototype = agentic.ArticleRetriever(TRAINING)
        query = ("مطالبة بمبلغ توريد بين تاجرين",)
        mine = [hit.key for hit in index.retrieve(query, 4, method="hybrid-tfidf")]
        theirs = [hit.key for hit in prototype.retrieve(query, 4)]
        assert mine == theirs

    def test_case_transfer_borrows_the_nearest_cases_citations(self, index) -> None:
        hits = index.retrieve(("طلب افتتاح إجراء إفلاس لتاجر متعثر",), 1, method="case-tfidf")
        assert hits[0].key == "نظام الافلاس:2"

    def test_popularity_ranks_the_most_cited_article_first(self, index) -> None:
        hits = index.retrieve(("أي وقائع",), 2, method="popularity")
        assert hits[0].key == "نظام المحاكم التجارية:16"

    @pytest.mark.parametrize("method", retrieval.METHOD_NAMES)
    def test_every_method_returns_ranked_indexed_articles(self, index, method) -> None:
        hits = index.retrieve(("مطالبة بمبلغ توريد بين تاجرين",), 3, method=method)
        assert 1 <= len(hits) <= 3
        assert all(hit.key in index.position for hit in hits)
        assert [h.similarity for h in hits] == sorted((h.similarity for h in hits), reverse=True)

    def test_retrieval_never_reads_the_query_cases_citations(self, index) -> None:
        facts = "مطالبة بمبلغ توريد بين تاجرين"
        with_laws, without_laws = case(facts, (LAW_D,)), case(facts, ())
        assert [h.key for h in index.retrieve_batch("fusion", [with_laws.case_facts])[0]] == [
            h.key for h in index.retrieve_batch("fusion", [without_laws.case_facts])[0]
        ]

    def test_citations_missing_from_the_index_count_as_misses(self, index) -> None:
        unseen = "نظام التنفيذ:46: نص غير موجود في بيانات التدريب"
        positions, total = index.gold(case("وقائع", (LAW_A, unseen)))
        assert len(positions) == 1 and total == 2

    def test_unknown_method_is_refused(self, index) -> None:
        with pytest.raises(ValueError):
            index.scores("bm25", ["وقائع"])


class TestRankMetrics:
    def test_hand_computed_values(self) -> None:
        scores = np.array([[0.9, 0.5, 0.1, 0.0], [0.1, 0.2, 0.9, 0.8]])
        gold = [({1}, 1), ({0, 3}, 2)]
        out = retrieval.rank_metrics(scores, gold, ks=(1, 2))
        assert out["queries"] == 2
        assert out["recall@1"] == 0.0
        assert out["recall@2"] == pytest.approx((1.0 + 0.5) / 2)
        assert out["mrr@10"] == pytest.approx((1 / 2 + 1 / 2) / 2)

    def test_queries_citing_nothing_are_skipped(self) -> None:
        out = retrieval.rank_metrics(np.array([[1.0, 0.0], [0.0, 1.0]]), [(set(), 0), ({1}, 1)], ks=(1,))
        assert out["queries"] == 1 and out["recall@1"] == 1.0

    def test_compare_reports_every_method(self) -> None:
        queries = [case("مطالبة بمبلغ توريد بين تاجرين", (LAW_A,))]
        result = retrieval.compare(TRAINING, queries, encoder=bag_encoder, temperature=0.05)
        assert [row["method"] for row in result["results"]] == list(retrieval.METHOD_NAMES)
        assert result["index"]["source"] == "training split only"


class TestRetrievalTask:
    def test_rag_prompt_lays_articles_out_like_the_dataset(self) -> None:
        item = tasks.build_verdict_rag(case("وقائع", ()), [("نظام الإثبات:17", "نص المادة")], "fusion")
        assert "نظام الإثبات:17: نص المادة" in item.prompt
        assert item.prompt.startswith(tasks.prompts.VERDICT_FROM_FACTS_AND_LAWS.split("{")[0])
        assert item.metadata == {"retriever": "fusion", "retrieved_keys": ["نظام الإثبات:17"]}

    def test_rag_task_needs_a_known_retriever(self) -> None:
        with pytest.raises(ValueError, match="retriever"):
            runner.RunConfig(task="verdict_rag", model="m")
        with pytest.raises(ValueError, match="retriever"):
            runner.RunConfig(task="verdict_rag", model="m", retriever="bm25")

    def test_other_tasks_refuse_a_retriever(self) -> None:
        with pytest.raises(ValueError, match="retriever"):
            runner.RunConfig(task="verdict_facts", model="m", retriever="fusion")

    def test_run_id_names_the_retriever(self) -> None:
        config = runner.RunConfig(task="verdict_rag", model="m", limit=30, retriever="hybrid-dense")
        assert config.run_id.endswith("__hybrid-dense")


class TestOutcomeAgreement:
    def test_counts_only_matching_outcome_categories(self) -> None:
        records = [
            {"reference": "إلزام المدعى عليه بسداد المبلغ.", "parsed_ok": True, "parsed": {"verdict": "ألزمت المحكمة المدعى عليه بالسداد."}},
            {"reference": "رفضت المحكمة الدعوى.", "parsed_ok": True, "parsed": {"verdict": "إلزام المدعى عليه بالسداد."}},
            {"reference": "إلزام المدعى عليه بسداد المبلغ.", "parsed_ok": False, "parsed": {"verdict": None}},
        ]
        assert runner.outcome_agreement(records) == {"count": 1, "of": 3, "share": 0.3333}


class TestPairedComparison:
    def test_sign_test(self) -> None:
        assert comparison.sign_test(0, 0) == 1.0
        assert comparison.sign_test(5, 5) == 1.0
        assert comparison.sign_test(8, 0) == pytest.approx(2 / 256, abs=1e-4)

    def test_wilson_interval_brackets_the_proportion(self) -> None:
        low, high = comparison.wilson(15, 20)
        assert low < 0.75 < high

    def test_runs_are_paired_on_shared_cases(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)

        def write_run(run_id, task, labels, retriever=None):
            d = tmp_path / run_id
            d.mkdir()
            (d / "config.json").write_text(json.dumps({"task": task, "retriever": retriever}), encoding="utf-8")
            with (d / "predictions.jsonl").open("w", encoding="utf-8") as preds, (d / "judgments.jsonl").open("w", encoding="utf-8") as judg:
                for cid, label in labels.items():
                    preds.write(json.dumps({"case_id": cid, "reference": "إلزام المدعى عليه.", "parsed_ok": True,
                                            "parsed": {"verdict": "إلزام المدعى عليه."}, "input_tokens": 10}) + "\n")
                    judg.write(json.dumps({"case_id": cid, "label": label, "judge": "llm:test"}) + "\n")

        write_run("base", "verdict_facts", {"a": "INCORRECT", "b": "PARTIALLY CORRECT", "c": "CORRECT", "extra": "CORRECT"})
        write_run("rag", "verdict_rag", {"a": "CORRECT", "b": "PARTIALLY CORRECT", "c": "INCORRECT"}, "fusion")
        result = comparison.compare_runs(["base", "rag"], output=None)
        assert result["shared_cases"] == 3
        assert result["rows"][1]["vs_baseline"] == {"better": 1, "worse": 1, "same": 1, "sign_test_p": 1.0}
        assert result["rows"][1]["condition"] == "Facts + retrieved articles (fusion)"

    def test_mixed_judges_are_refused(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        for run_id, judge in (("one", "llm:a"), ("two", "llm:b")):
            d = tmp_path / run_id
            d.mkdir()
            (d / "config.json").write_text(json.dumps({"task": "verdict_facts"}), encoding="utf-8")
            (d / "predictions.jsonl").write_text(json.dumps({"case_id": "x", "reference": "r", "parsed_ok": True, "parsed": {"verdict": "v"}, "input_tokens": 1}) + "\n", encoding="utf-8")
            (d / "judgments.jsonl").write_text(json.dumps({"case_id": "x", "label": "CORRECT", "judge": judge}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="different judges"):
            comparison.compare_runs(["one", "two"], output=None)
