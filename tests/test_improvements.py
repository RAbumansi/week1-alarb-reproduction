"""Concurrency, resume, provenance and judge-agreement guarantees."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alarb import agreement, provenance, providers, runner, tasks


@pytest.fixture
def runs_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    return tmp_path


class TestConcurrency:
    def test_concurrent_and_serial_runs_agree_exactly(self, runs_dir) -> None:
        """Parallelism must not reorder or alter the output file."""
        serial = runner.RunConfig(task="mcq_semantic", model="random", limit=40)
        runner.run_inference(serial, providers.RandomChoiceModel(), concurrency=1, progress_every=0)
        one = (serial.directory / "predictions.jsonl").read_text(encoding="utf-8")

        parallel = runner.RunConfig(task="mcq_semantic", model="random", limit=40, seed=42)
        parallel.notes["variant"] = "parallel"
        # Same run_id, so clear it and rebuild with workers.
        (parallel.directory / "predictions.jsonl").unlink()
        runner.run_inference(
            parallel, providers.RandomChoiceModel(), concurrency=8, progress_every=0
        )
        many = (parallel.directory / "predictions.jsonl").read_text(encoding="utf-8")

        assert one == many

    def test_results_stay_in_item_order(self, runs_dir) -> None:
        config = runner.RunConfig(task="verdict_facts", model="constant", limit=30)
        runner.run_inference(config, providers.ConstantModel(), concurrency=8, progress_every=0)
        written = [
            row["case_id"]
            for row in runner.read_jsonl(config.directory / "predictions.jsonl")
        ]
        assert written == [item.case_id for item in runner.build_items(config)]


class TestResume:
    def test_completed_case_ids_reads_what_is_there(self, tmp_path) -> None:
        path = tmp_path / "predictions.jsonl"
        path.write_text(
            json.dumps({"case_id": "aaa"}) + "\n" + json.dumps({"case_id": "bbb"}) + "\n",
            encoding="utf-8",
        )
        assert runner.completed_case_ids(path) == {"aaa", "bbb"}

    def test_a_truncated_final_line_is_ignored(self, tmp_path) -> None:
        """A run killed mid-write leaves a partial line; it must not crash resume."""
        path = tmp_path / "predictions.jsonl"
        path.write_text(json.dumps({"case_id": "aaa"}) + "\n" + '{"case_id": "bb', encoding="utf-8")
        assert runner.completed_case_ids(path) == {"aaa"}

    def test_missing_file_means_nothing_done(self, tmp_path) -> None:
        assert runner.completed_case_ids(tmp_path / "absent.jsonl") == set()

    def test_resume_keeps_earlier_rows_and_adds_the_rest(self, runs_dir) -> None:
        config = runner.RunConfig(task="verdict_facts", model="constant", limit=20)
        items = runner.build_items(config)
        runner.run_inference(config, providers.ConstantModel(), progress_every=0)
        predictions = config.directory / "predictions.jsonl"
        complete = runner.read_jsonl(predictions)

        # Simulate a crash after the first 5 items.
        with predictions.open("w", encoding="utf-8") as stream:
            for row in complete[:5]:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

        runner.run_inference(config, providers.ConstantModel(), progress_every=0)
        resumed = runner.read_jsonl(predictions)
        assert len(resumed) == len(items)
        assert [row["case_id"] for row in resumed] == [row["case_id"] for row in complete]

    def test_no_resume_starts_over(self, runs_dir) -> None:
        config = runner.RunConfig(task="verdict_facts", model="constant", limit=10)
        runner.run_inference(config, providers.ConstantModel(), progress_every=0)
        runner.run_inference(config, providers.ConstantModel(), resume=False, progress_every=0)
        assert len(runner.read_jsonl(config.directory / "predictions.jsonl")) == 10


class TestProvenance:
    def test_a_run_records_what_produced_it(self, runs_dir) -> None:
        config = runner.RunConfig(task="verdict_facts", model="constant", limit=5)
        runner.run_inference(config, providers.ConstantModel(), progress_every=0)
        stored = json.loads((config.directory / "config.json").read_text(encoding="utf-8"))
        assert set(stored["provenance"]) == {
            "dataset_revision",
            "corpus_digest",
            "prompt_digest",
            "harness_commit",
        }
        assert stored["provenance"]["corpus_digest"]

    def test_each_task_fingerprints_its_own_prompt(self) -> None:
        digests = {task: provenance.prompt_digest(task) for task in tasks.VERDICT_TASKS}
        assert all(digests.values())
        assert len(set(digests.values())) == len(digests)

    def test_both_mcq_variants_share_one_prompt(self) -> None:
        assert provenance.prompt_digest("mcq_same_statute") == provenance.prompt_digest(
            "mcq_semantic"
        )

    def test_matching_runs_are_comparable(self) -> None:
        run = {"task": "verdict_facts", "provenance": {"corpus_digest": "a", "prompt_digest": "p"}}
        ok, differing, missing = provenance.comparable([dict(run, run_id="a"), dict(run, run_id="b")])
        assert ok and not differing and not missing

    def test_a_changed_corpus_is_flagged(self) -> None:
        first = {"task": "verdict_facts", "provenance": {"corpus_digest": "a"}}
        second = {"task": "verdict_facts", "provenance": {"corpus_digest": "b"}}
        ok, differing, _ = provenance.comparable([dict(first, run_id="a"), dict(second, run_id="b")])
        assert not ok and "corpus_digest" in differing

    def test_a_changed_prompt_is_flagged(self) -> None:
        first = {"task": "verdict_facts", "provenance": {"prompt_digest": "p"}}
        second = {"task": "verdict_facts", "provenance": {"prompt_digest": "q"}}
        ok, differing, _ = provenance.comparable([dict(first, run_id="a"), dict(second, run_id="b")])
        assert not ok and "prompt_digest" in differing

    def test_a_run_with_no_fingerprint_is_reported_separately(self) -> None:
        """Predating provenance tracking is not the same problem as an input
        having changed, and the fix differs."""
        tracked = {"run_id": "new", "task": "verdict_facts", "provenance": {"corpus_digest": "a"}}
        legacy = {"run_id": "old", "task": "verdict_facts"}
        ok, differing, missing = provenance.comparable([tracked, legacy])
        assert not ok
        assert not differing
        assert missing == ["old"]

    def test_different_tasks_may_differ_in_prompt(self) -> None:
        """Two tasks naturally use different prompts; that is not a mismatch."""
        first = {"task": "verdict_facts", "provenance": {"prompt_digest": "p"}}
        second = {"task": "verdict_laws", "provenance": {"prompt_digest": "q"}}
        ok, differing, _ = provenance.comparable([dict(first, run_id="a"), dict(second, run_id="b")])
        assert ok and not differing


def review_rows(pairs: list[tuple[str, str]]) -> list[dict]:
    return [
        {
            "case_id": f"c{index}",
            "judge_label": judge,
            agreement.HUMAN_COLUMN: human,
            "gold_verdict": "g",
            "predicted_verdict": "p",
        }
        for index, (judge, human) in enumerate(pairs)
    ]


class TestJudgeAgreement:
    def test_sample_spreads_across_the_judges_labels(self) -> None:
        predictions = [
            {"case_id": f"c{i}", "reference": "gold", "parsed": {"verdict": "pred"}}
            for i in range(60)
        ]
        judgments = [
            {"case_id": f"c{i}", "label": "CORRECT" if i < 54 else "INCORRECT"}
            for i in range(60)
        ]
        rows = agreement.sample_for_review(predictions, judgments, size=12)
        labels = {row["judge_label"] for row in rows}
        assert labels == {"CORRECT", "INCORRECT"}

    def test_sheet_round_trips_arabic(self, tmp_path) -> None:
        rows = [
            {
                "case_id": "abc",
                "judge_label": "CORRECT",
                agreement.HUMAN_COLUMN: "",
                "gold_verdict": "إلزام المدعى عليه بالسداد.",
                "predicted_verdict": "ألزمت المحكمة المدعى عليه.",
            }
        ]
        path = tmp_path / "sheet.csv"
        agreement.write_review_sheet(rows, path)
        back = agreement.read_review_sheet(path)
        assert back[0]["gold_verdict"] == rows[0]["gold_verdict"]

    def test_perfect_agreement_scores_one(self) -> None:
        report = agreement.score_agreement(
            review_rows([("CORRECT", "CORRECT")] * 5 + [("INCORRECT", "INCORRECT")] * 5)
        )
        assert report["raw_agreement"] == 1.0
        assert report["cohens_kappa"] == 1.0

    def test_kappa_punishes_a_judge_riding_one_label(self) -> None:
        """Raw agreement flatters a constant judge; kappa should not."""
        pairs = [("INCORRECT", "INCORRECT")] * 9 + [("INCORRECT", "CORRECT")]
        report = agreement.score_agreement(review_rows(pairs))
        assert report["raw_agreement"] == 0.9
        assert report["cohens_kappa"] <= 0.0

    def test_kappa_is_undefined_when_everything_got_one_grade(self) -> None:
        """Chance correction is meaningless with a single label, and NaN is not
        valid JSON -- so it must come back as None with an explanation."""
        report = agreement.score_agreement(review_rows([("PARTIALLY CORRECT", "PARTIALLY CORRECT")] * 5))
        assert report["cohens_kappa"] is None
        assert "undefined" in report["kappa_reading"]
        assert report["raw_agreement"] == 1.0
        json.dumps(report)  # would raise on NaN under a strict encoder

    def test_the_report_is_strict_json(self) -> None:
        report = agreement.score_agreement(review_rows([("CORRECT", "CORRECT")] * 4))
        json.loads(json.dumps(report, allow_nan=False))

    def test_disagreements_are_listed(self) -> None:
        report = agreement.score_agreement(
            review_rows([("CORRECT", "CORRECT"), ("INCORRECT", "CORRECT")])
        )
        assert len(report["disagreements"]) == 1
        assert report["disagreements"][0]["judge"] == "INCORRECT"

    def test_blank_rows_are_counted_not_scored(self) -> None:
        report = agreement.score_agreement(
            review_rows([("CORRECT", "CORRECT"), ("CORRECT", "")])
        )
        assert report["reviewed"] == 1 and report["unlabelled"] == 1

    def test_an_unrecognised_label_is_reported_not_silently_dropped(self) -> None:
        report = agreement.score_agreement(
            review_rows([("CORRECT", "CORRECT"), ("CORRECT", "probably fine")])
        )
        assert report["reviewed"] == 1
        assert report["invalid_labels"][0][agreement.HUMAN_COLUMN] == "PROBABLY FINE"

    def test_labels_are_case_insensitive(self) -> None:
        report = agreement.score_agreement(review_rows([("CORRECT", "correct")]))
        assert report["raw_agreement"] == 1.0

    def test_an_entirely_unlabelled_sheet_explains_itself(self) -> None:
        with pytest.raises(ValueError, match=agreement.HUMAN_COLUMN):
            agreement.score_agreement(review_rows([("CORRECT", ""), ("INCORRECT", "")]))

    @pytest.mark.parametrize(
        ("kappa", "expected"),
        [(0.9, "almost perfect"), (0.7, "substantial"), (0.5, "moderate"), (-0.1, "worse than chance")],
    )
    def test_kappa_reading(self, kappa: float, expected: str) -> None:
        assert agreement.interpret_kappa(kappa) == expected
