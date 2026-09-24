"""Phase B guarantees: prompts match the paper, and MCQ items are well formed."""

from __future__ import annotations

import pytest

from alarb import corpus, data, mcq, prompts, tasks
from alarb.data import Case


def make_case(
    facts: tuple[str, ...] = ("fact one", "fact two"),
    reasoning: tuple[str, ...] = ("step one", "step two", "step three"),
    laws: tuple[str, ...] = ("نظام الإثبات:17: نص المادة.",),
    verdict: str = "إلزام المدعى عليه بالسداد.",
) -> Case:
    return Case(
        case_id=data.case_id(facts),
        case_facts=facts,
        court_reasoning=reasoning,
        applicable_laws=laws,
        verdict=verdict,
    )


class TestPromptsMatchThePaper:
    def test_figure_8_opening_is_verbatim(self) -> None:
        assert prompts.VERDICT_FROM_FACTS.startswith(
            "You are a legal assistant specialized in Saudi Arabian law."
        )

    def test_closing_tags_use_the_papers_backslash_form(self) -> None:
        assert "[\\REASONING]" in prompts.VERDICT_FROM_FACTS
        assert "[\\VERDICT]" in prompts.VERDICT_FROM_FACTS

    def test_figure_9_asks_for_a_verdict_only(self) -> None:
        assert "[REASONING]" not in prompts.VERDICT_FROM_FACTS_AND_REASONING
        assert prompts.VERDICT_FROM_FACTS_AND_REASONING.endswith("Begin")

    def test_judge_prompt_offers_the_three_published_labels(self) -> None:
        for label in ('"CORRECT"', '"INCORRECT"', '"PARTIALLY CORRECT"'):
            assert label in prompts.JUDGE
        assert "[THINK]" in prompts.JUDGE and "[EVALUATION]" in prompts.JUDGE


class TestVerdictTasks:
    @pytest.mark.parametrize("task", tasks.VERDICT_TASKS)
    def test_rendered_prompt_leaves_no_placeholder(self, task: str) -> None:
        items = tasks.build_verdict_items(task, [make_case()])
        prompt = items[0].prompt
        assert "{" not in prompt and "}" not in prompt

    def test_verdict_laws_skips_cases_citing_nothing(self) -> None:
        assert tasks.build_verdict_laws(make_case(laws=())) is None
        assert tasks.build_verdict_laws(make_case()) is not None

    def test_argument_completion_withholds_the_closing_steps(self) -> None:
        case = make_case(reasoning=("a", "b", "c", "d", "e"))
        item = tasks.build_argument_completion(case, omitted_steps=2)
        assert item is not None
        assert item.metadata == {"omitted_steps": 2, "shown_steps": 3, "total_steps": 5}
        assert "c" in item.prompt and "\nd" not in item.prompt

    def test_argument_completion_skips_when_nothing_would_remain(self) -> None:
        case = make_case(reasoning=("only step",))
        assert tasks.build_argument_completion(case, omitted_steps=1) is None

    def test_reference_is_always_the_gold_verdict(self) -> None:
        case = make_case()
        for task in tasks.VERDICT_TASKS:
            for item in tasks.build_verdict_items(task, [case]):
                assert item.reference == case.verdict


class TestChoiceRendering:
    def test_choices_are_lettered_a_to_d(self) -> None:
        rendered = tasks.format_choices(
            (("doc:1", "first"), ("doc:2", "second"), ("doc:3", "third"), ("doc:4", "fourth"))
        )
        for letter in tasks.CHOICE_LETTERS:
            assert f"{letter}) doc:" in rendered


@pytest.fixture(scope="module")
def built() -> dict[str, list[mcq.McqItem]]:
    cases = data.load_split("test")[:120]
    loaded = corpus.load_corpus()
    return {
        task: mcq.build_mcqs(task, cases, loaded, embedder_backend="tfidf")[0]
        for task in tasks.MCQ_TASKS
    }


class TestMcqConstruction:
    @pytest.mark.parametrize("task", tasks.MCQ_TASKS)
    def test_items_have_four_distinct_answerable_choices(self, built, task: str) -> None:
        for item in built[task]:
            assert len(item.choices) == 4
            assert len(set(item.choice_keys)) == 4
            assert len({text for _, text in item.choices}) == 4
            assert item.correct_key in item.choice_keys
            position = item.choice_keys.index(item.correct_key)
            assert tasks.CHOICE_LETTERS[position] == item.answer

    @pytest.mark.parametrize("task", tasks.MCQ_TASKS)
    def test_no_distractor_is_also_cited_by_the_case(self, built, task: str) -> None:
        loaded = corpus.load_corpus()
        by_id = {case.case_id: case for case in data.load_split("test")[:120]}
        for item in built[task]:
            cited = set(mcq.cited_keys(by_id[item.case_id], loaded))
            distractors = set(item.choice_keys) - {item.correct_key}
            assert not (distractors & cited)

    def test_same_statute_distractors_stay_inside_one_document(self, built) -> None:
        for item in built["mcq_same_statute"]:
            assert len({key.rsplit(":", 1)[0] for key in item.choice_keys}) == 1

    def test_construction_is_deterministic(self) -> None:
        cases = data.load_split("test")[:60]
        loaded = corpus.load_corpus()
        first, _ = mcq.build_mcqs("mcq_same_statute", cases, loaded, embedder_backend="tfidf")
        second, _ = mcq.build_mcqs("mcq_same_statute", cases, loaded, embedder_backend="tfidf")
        assert [item.choice_keys for item in first] == [item.choice_keys for item in second]
        assert [item.answer for item in first] == [item.answer for item in second]

    def test_an_item_does_not_depend_on_the_batch_around_it(self) -> None:
        cases = data.load_split("test")[:60]
        loaded = corpus.load_corpus()
        batch, _ = mcq.build_mcqs("mcq_same_statute", cases, loaded, embedder_backend="tfidf")
        alone, _ = mcq.build_mcqs(
            "mcq_same_statute", [cases[-1]], loaded, embedder_backend="tfidf"
        )
        matching = [item for item in batch if item.case_id == alone[0].case_id]
        assert matching and matching[0].choice_keys == alone[0].choice_keys
