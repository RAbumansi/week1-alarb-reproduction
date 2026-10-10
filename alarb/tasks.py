"""Render the ALARB tasks into prompts.

Facts, reasoning steps and law entries are joined with newlines and nothing
else: the dataset already carries its own enumeration ("1- ...", "2- ..."), so
re-numbering here would double it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from alarb import prompts
from alarb.data import Case


DEFAULT_LANGUAGE = "Arabic"
DEFAULT_OMITTED_STEPS = 2

VERDICT_TASKS = (
    "verdict_facts",
    "verdict_laws",
    "verdict_reasoning",
    "argument_completion",
)
MCQ_TASKS = ("mcq_same_statute", "mcq_semantic")
ALL_TASKS = VERDICT_TASKS + MCQ_TASKS

CHOICE_LETTERS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class TaskItem:
    task: str
    case_id: str
    prompt: str
    reference: str
    metadata: dict = field(default_factory=dict)


def _block(lines: tuple[str, ...]) -> str:
    return "\n".join(lines)


def build_verdict_facts(case: Case, language: str = DEFAULT_LANGUAGE) -> TaskItem:
    return TaskItem(
        task="verdict_facts",
        case_id=case.case_id,
        prompt=prompts.VERDICT_FROM_FACTS.format(
            language=language, case_facts=_block(case.case_facts)
        ),
        reference=case.verdict,
    )


def build_verdict_laws(case: Case, language: str = DEFAULT_LANGUAGE) -> TaskItem | None:
    """None when the case cites no articles; 988 of them do not."""
    if not case.applicable_laws:
        return None
    return TaskItem(
        task="verdict_laws",
        case_id=case.case_id,
        prompt=prompts.VERDICT_FROM_FACTS_AND_LAWS.format(
            language=language,
            case_facts=_block(case.case_facts),
            case_laws=_block(case.applicable_laws),
        ),
        reference=case.verdict,
        metadata={"cited_articles": len(case.applicable_laws)},
    )


def build_verdict_reasoning(case: Case, language: str = DEFAULT_LANGUAGE) -> TaskItem:
    return TaskItem(
        task="verdict_reasoning",
        case_id=case.case_id,
        prompt=prompts.VERDICT_FROM_FACTS_AND_REASONING.format(
            language=language,
            case_facts=_block(case.case_facts),
            case_reasoning=_block(case.court_reasoning),
        ),
        reference=case.verdict,
    )


def build_argument_completion(
    case: Case,
    omitted_steps: int = DEFAULT_OMITTED_STEPS,
    language: str = DEFAULT_LANGUAGE,
) -> TaskItem | None:
    """None when omitting that many steps would leave no reasoning to build on."""
    shown = len(case.court_reasoning) - omitted_steps
    if shown < 1:
        return None
    return TaskItem(
        task="argument_completion",
        case_id=case.case_id,
        prompt=prompts.ARGUMENT_COMPLETION.format(
            language=language,
            case_facts=_block(case.case_facts),
            case_reasoning=_block(case.court_reasoning[:shown]),
        ),
        reference=case.verdict,
        metadata={
            "omitted_steps": omitted_steps,
            "shown_steps": shown,
            "total_steps": len(case.court_reasoning),
        },
    )


def format_choices(choices: tuple[tuple[str, str], ...]) -> str:
    """Lay out (heading, text) pairs as lettered options, as in Figure 11."""
    return "\n\n".join(
        f"{letter}) {heading}\n{text}"
        for letter, (heading, text) in zip(CHOICE_LETTERS, choices)
    )


def build_article_mcq(
    task: str,
    case_id: str,
    case_facts: tuple[str, ...],
    choices: tuple[tuple[str, str], ...],
    answer: str,
    metadata: dict | None = None,
) -> TaskItem:
    return TaskItem(
        task=task,
        case_id=case_id,
        prompt=prompts.ARTICLE_MCQ.format(
            case_facts=_block(case_facts), choices=format_choices(choices)
        ),
        reference=answer,
        metadata=metadata or {},
    )


def build_verdict_items(
    task: str,
    cases: list[Case],
    omitted_steps: int = DEFAULT_OMITTED_STEPS,
    language: str = DEFAULT_LANGUAGE,
) -> list[TaskItem]:
    if task not in VERDICT_TASKS:
        raise ValueError(f"{task!r} is not a verdict task; expected one of {VERDICT_TASKS}")
    built = []
    for case in cases:
        if task == "verdict_facts":
            item = build_verdict_facts(case, language)
        elif task == "verdict_laws":
            item = build_verdict_laws(case, language)
        elif task == "verdict_reasoning":
            item = build_verdict_reasoning(case, language)
        else:
            item = build_argument_completion(case, omitted_steps, language)
        if item is not None:
            built.append(item)
    return built
