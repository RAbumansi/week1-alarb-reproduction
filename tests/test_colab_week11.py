"""The week 11 Colab pipeline, run end to end with a fake model (no GPU, no data)."""

from __future__ import annotations

import json

from alarb import colab_week11 as w
from alarb import data

LAWS = (
    "نظام المحاكم التجارية:16: تختص المحكمة بالنظر في المنازعات التجارية",
    "نظام الإثبات:17: الإقرار القضائي حجة قاطعة على المقر",
    "نظام الشركات:8: يجب أن يكون عقد الشركة مكتوبا",
)
SECRET = "اسم-سري-لطرف"


def make(i: int, laws=True) -> data.Case:
    facts = (f"وقائع رقم {i} مطالبة بمبلغ توريد بين تاجرين {SECRET}",)
    return data.Case(data.case_id(facts), facts, ("سبب",), (LAWS[i % 3],) if laws else (),
                     "إلزام المدعى عليه بسداد المبلغ." if i % 2 else "رفضت المحكمة الدعوى.")


class FakeBackend:
    metadata = {"backend": "fake"}

    def __init__(self):
        self.calls = 0

    def generate(self, prompts, thinking, max_new_tokens, seed):
        self.calls += len(prompts)
        out = []
        for p in prompts:
            text = ("[THINK]\nمقارنة\n[EVALUATION]\nCORRECT" if "Judge's verdict" in p
                    else "[REASONING]\nتحليل\n[\\REASONING]\n[VERDICT]\nإلزام المدعى عليه بالسداد.\n[\\VERDICT]")
            out.append({"final": text, "truncated": False, "thinking_complete": True,
                        "input_tokens": 100, "output_tokens": 50, "batch_seconds": 1.0})
        return out


def test_pipeline_runs_resumes_and_leaks_no_case_text(tmp_path):
    training = [make(i) for i in range(30)]
    testing = [make(100 + i) for i in range(12)] + [make(200, laws=False)]
    config = w.Week11Config(cases=6, batch_size=4)

    cases = w.select_cases(testing, config.cases, config.seed)
    assert len(cases) == 6 and all(case.applicable_laws for case in cases)

    built, prep = w.build_conditions(training, cases)
    assert set(built) == set(w.CONDITION_IDS)
    assert all(len(items) == 6 for items in built.values())
    assert set(prep["recall@5_on_selected_cases"]) == {"popularity", "hybrid-tfidf"}

    backend = FakeBackend()
    runs = tmp_path / "runs"
    for (task, retriever), cid in zip(w.CONDITIONS, w.CONDITION_IDS):
        w.run_condition(built[cid], backend, runs / cid, config, cid, retriever)
        w.judge_condition(runs / cid, backend, config, cid)
    assert backend.calls == 4 * 6 * 2

    # A rerun resumes: nothing is generated or judged twice.
    for (task, retriever), cid in zip(w.CONDITIONS, w.CONDITION_IDS):
        w.run_condition(built[cid], backend, runs / cid, config, cid, retriever)
        w.judge_condition(runs / cid, backend, config, cid)
    assert backend.calls == 4 * 6 * 2

    summary = w.summarise(tmp_path, config, prep, backend.metadata)
    assert [row["condition"] for row in summary["comparison"]["rows"]][0] == "Facts only (paper task 1)"
    assert summary["comparison"]["shared_cases"] == 6
    assert len(summary["per_case"]) == 6
    assert summary["validation_accessed"] is False
    printed = json.dumps(summary, ensure_ascii=False)
    assert SECRET not in printed and "وقائع رقم" not in printed


def test_unparseable_verdicts_are_not_sent_to_the_judge(tmp_path):
    class Broken(FakeBackend):
        def generate(self, prompts, thinking, max_new_tokens, seed):
            out = super().generate(prompts, thinking, max_new_tokens, seed)
            for row, p in zip(out, prompts):
                if "Judge's verdict" not in p:
                    row.update(final="word " * 200, truncated=True)
            return out

    cases = [make(i) for i in range(3)]
    built, _ = w.build_conditions([make(50 + i) for i in range(10)], cases)
    backend = Broken()
    directory = tmp_path / "runs" / "verdict_facts"
    preds = w.run_condition(built["verdict_facts"], backend, directory, w.Week11Config(cases=3), "f", None)
    assert not any(r["parsed_ok"] for r in preds)
    before = backend.calls
    labels = [r["label"] for r in w.judge_condition(directory, backend, w.Week11Config(cases=3), "f")]
    assert labels == ["UNPARSEABLE"] * 3 and backend.calls == before
