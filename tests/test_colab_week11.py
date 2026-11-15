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

    results = {cid: w.compact(runs / cid) for cid in w.CONDITION_IDS}
    summary = w.summarise(results, config, prep, backend.metadata, w.judge_label(config))
    rows = summary["comparison"]["rows"]
    assert rows[0]["condition"] == "Facts only (paper task 1)"
    assert summary["comparison"]["shared_cases"] == 6
    assert all("vs_baseline" in row for row in rows[1:])
    assert summary["validation_accessed"] is False
    printed = json.dumps(summary, ensure_ascii=False)
    assert SECRET not in printed and "وقائع رقم" not in printed


def test_printed_conditions_can_be_recovered(capsys):
    rows = [{"case_id": "a", "label": "CORRECT", "gold_outcome": "for_plaintiff", "predicted_outcome": "for_plaintiff",
             "parsed_ok": True, "truncated": False, "input_tokens": 1, "output_tokens": 1}]
    w.print_condition("verdict_facts", rows)
    w.print_condition("verdict_laws", rows)
    recovered = w.parse_printed("noise\n" + capsys.readouterr().out + "more noise")
    assert recovered == {"verdict_facts": rows, "verdict_laws": rows}


def test_table_pairs_against_facts_only():
    def row(cid, label, predicted="for_plaintiff"):
        return {"case_id": cid, "label": label, "gold_outcome": "for_plaintiff", "predicted_outcome": predicted,
                "parsed_ok": True, "truncated": False, "input_tokens": 10, "output_tokens": 5}
    result = w.table({
        "verdict_facts": [row("a", "INCORRECT", "court_dismissal"), row("b", "CORRECT")],
        "verdict_laws": [row("a", "CORRECT"), row("b", "CORRECT")],
    })
    facts, laws = result["rows"]
    assert facts["outcome_agreement"] == {"count": 1, "of": 2, "share": 0.5}
    assert laws["vs_baseline"] == {"better": 1, "worse": 0, "same": 1, "sign_test_p": 1.0}


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
