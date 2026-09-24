"""Phase C guarantees: output is parsed faithfully, caching holds, scoring adds up."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alarb import judge as judge_module
from alarb import data, parsing, providers, runner, tasks


class TestVerdictParsing:
    def test_reads_the_papers_backslash_closing_tag(self) -> None:
        parsed = parsing.parse_verdict("[VERDICT]\nإلزام المدعى عليه.\n[\\VERDICT]")
        assert parsed.verdict == "إلزام المدعى عليه."

    def test_reads_the_forward_slash_models_actually_emit(self) -> None:
        parsed = parsing.parse_verdict("[VERDICT]\nإلزام المدعى عليه.\n[/VERDICT]")
        assert parsed.verdict == "إلزام المدعى عليه."

    def test_reads_a_section_with_no_closing_tag(self) -> None:
        parsed = parsing.parse_verdict("[VERDICT]\nإلزام المدعى عليه.")
        assert parsed.verdict == "إلزام المدعى عليه."

    def test_keeps_reasoning_out_of_the_verdict(self) -> None:
        parsed = parsing.parse_verdict(
            "[REASONING]\nتحليل الوقائع.\n[\\REASONING]\n[VERDICT]\nالحكم.\n[\\VERDICT]"
        )
        assert parsed.verdict == "الحكم."
        assert parsed.reasoning == "تحليل الوقائع."

    def test_discards_thinking_blocks(self) -> None:
        parsed = parsing.parse_verdict(
            "<think>should not leak</think>[VERDICT]\nالحكم.\n[\\VERDICT]"
        )
        assert parsed.verdict == "الحكم."
        assert "should not leak" not in (parsed.verdict or "")

    def test_strips_the_quotes_the_template_invites(self) -> None:
        assert parsing.parse_verdict('[VERDICT]\n"الحكم."\n[\\VERDICT]').verdict == "الحكم."

    def test_accepts_a_short_untagged_answer(self) -> None:
        assert parsing.parse_verdict("إلزام المدعى عليه بالسداد.").ok

    def test_refuses_to_treat_an_essay_as_a_verdict(self) -> None:
        assert not parsing.parse_verdict("word " * 200).ok

    def test_empty_output_is_unparseable_not_wrong(self) -> None:
        assert not parsing.parse_verdict("").ok


class TestMcqParsing:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("[ANSWER]\nB\n[\\ANSWER]", "B"),
            ("[ANSWER]\n\"C\"\n[/ANSWER]", "C"),
            ("The answer is D", "D"),
            ("<think>maybe A</think>[ANSWER]\nA\n[\\ANSWER]", "A"),
        ],
    )
    def test_recovers_the_letter(self, text: str, expected: str) -> None:
        assert parsing.parse_mcq_answer(text) == expected

    def test_returns_none_when_no_letter_is_present(self) -> None:
        assert parsing.parse_mcq_answer("لا أعرف") is None


class TestJudgementParsing:
    def test_partially_correct_is_not_read_as_correct(self) -> None:
        assert parsing.parse_judgement("[EVALUATION]\nPARTIALLY CORRECT") == "PARTIALLY CORRECT"

    def test_incorrect_is_not_read_as_correct(self) -> None:
        assert parsing.parse_judgement("[EVALUATION]\nINCORRECT") == "INCORRECT"

    def test_reads_a_label_without_the_evaluation_tag(self) -> None:
        assert parsing.parse_judgement("I judge this CORRECT.") == "CORRECT"

    def test_unlabelled_output_returns_none(self) -> None:
        assert parsing.parse_judgement("no verdict label here") is None


class TestCache:
    def test_round_trips_a_completion(self, tmp_path: Path) -> None:
        cache = providers.CompletionCache(tmp_path)
        key = cache.key("m", "prompt", {"temperature": 0.0})
        assert cache.get(key) is None
        cache.put(key, providers.Completion("hello", "m", 10, 5))
        hit = cache.get(key)
        assert hit is not None and hit.text == "hello" and hit.cached

    def test_key_depends_on_model_prompt_and_params(self, tmp_path: Path) -> None:
        cache = providers.CompletionCache(tmp_path)
        base = cache.key("m", "p", {"temperature": 0.0})
        assert base != cache.key("other", "p", {"temperature": 0.0})
        assert base != cache.key("m", "other", {"temperature": 0.0})
        assert base != cache.key("m", "p", {"temperature": 1.0})


class TestOfflineModels:
    def test_constant_model_output_parses_as_a_verdict(self) -> None:
        completion = providers.ConstantModel().complete("anything")
        assert parsing.parse_verdict(completion.text).verdict == providers.MAJORITY_CLASS_VERDICT

    def test_random_model_is_deterministic_per_prompt(self) -> None:
        first = providers.RandomChoiceModel().complete("prompt")
        second = providers.RandomChoiceModel().complete("prompt")
        assert first.text == second.text
        assert parsing.parse_mcq_answer(first.text) in tasks.CHOICE_LETTERS

    def test_random_model_varies_across_prompts(self) -> None:
        letters = {
            parsing.parse_mcq_answer(providers.RandomChoiceModel().complete(f"p{i}").text)
            for i in range(40)
        }
        assert len(letters) == 4

    def test_a_real_model_without_a_key_fails_clearly(self, monkeypatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            providers.get_model("claude-sonnet-5")


class TestLocalModels:
    def test_ollama_prefix_selects_a_local_server(self) -> None:
        model = providers.get_model("ollama:qwen3:8b")
        assert isinstance(model, providers.OpenAICompatibleModel)
        # Only the leading "ollama:" is stripped; the tag colon is part of the name.
        assert model.name == "qwen3:8b"

    def test_local_models_need_no_api_key(self, monkeypatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert providers.get_model("ollama:qwen3:8b") is not None

    def test_base_url_follows_ollama_host(self, monkeypatch) -> None:
        monkeypatch.setenv("OLLAMA_HOST", "http://elsewhere:9999")
        assert providers.ollama_base_url() == "http://elsewhere:9999/v1"

    def test_bare_host_gets_a_scheme(self, monkeypatch) -> None:
        monkeypatch.setenv("OLLAMA_HOST", "localhost:11434")
        assert providers.ollama_base_url() == "http://localhost:11434/v1"

    def test_unreachable_server_explains_how_to_start_one(self, tmp_path: Path) -> None:
        model = providers.OpenAICompatibleModel(
            "qwen3:8b",
            "http://localhost:59999/v1",
            cache=providers.CompletionCache(tmp_path),
            timeout=2,
        )
        with pytest.raises(RuntimeError, match="ollama serve"):
            model.complete("hello")


class TestSampling:
    def test_sample_is_deterministic(self) -> None:
        cases = data.load_split("test")
        first = runner.stratified_sample(cases, 100, seed=42)
        second = runner.stratified_sample(cases, 100, seed=42)
        assert [case.case_id for case in first] == [case.case_id for case in second]

    def test_sample_respects_the_requested_size(self) -> None:
        cases = data.load_split("test")
        assert len(runner.stratified_sample(cases, 100, seed=42)) <= 100

    def test_sample_keeps_the_outcome_mix(self) -> None:
        from alarb import outcomes

        cases = data.load_split("test")
        sampled = runner.stratified_sample(cases, 400, seed=42)
        full = outcomes.outcome_distribution(cases)["paper_categories"]
        drawn = outcomes.outcome_distribution(sampled)["paper_categories"]
        for label, values in full.items():
            if values["share"] >= 0.05:
                assert abs(drawn[label]["share"] - values["share"]) < 0.06

    def test_asking_for_more_than_exists_returns_everything(self) -> None:
        cases = data.load_split("test")
        assert len(runner.stratified_sample(cases, 99_999, seed=42)) == len(cases)


class TestHeuristicJudge:
    def test_identical_verdicts_are_correct(self) -> None:
        verdict = "إلزام المدعى عليه بسداد مبلغ خمسين ألف ريال للمدعي"
        assert judge_module.HeuristicJudge().judge(verdict, verdict).label == "CORRECT"

    def test_unrelated_verdicts_are_incorrect(self) -> None:
        result = judge_module.HeuristicJudge().judge(
            "إلزام المدعى عليه بسداد مبلغ خمسين ألف ريال",
            "عدم اختصاص المحكمة ولائيا بنظر الدعوى",
        )
        assert result.label == "INCORRECT"


class TestScoring:
    def test_mcq_scoring_counts_only_exact_letter_matches(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        config = runner.RunConfig(task="mcq_semantic", model="random", limit=3)
        directory = config.directory
        directory.mkdir(parents=True)
        rows = [
            {"case_id": "a", "reference": "A", "parsed": {"answer": "A"}, "parsed_ok": True},
            {"case_id": "b", "reference": "B", "parsed": {"answer": "C"}, "parsed_ok": True},
            {"case_id": "c", "reference": "C", "parsed": {"answer": None}, "parsed_ok": False},
        ]
        (directory / "predictions.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        metrics = runner.score(config)
        assert metrics["correct"] == 1
        assert metrics["accuracy"] == pytest.approx(1 / 3, abs=1e-4)
        assert metrics["unparseable"]["count"] == 1
