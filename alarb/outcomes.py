"""Heuristic classification of verdict text into outcome categories.

The dataset ships verdicts as free Arabic text with no outcome label, but Table 3
of the paper reports an outcome breakdown, and a majority-class reference point
is needed to read the benchmark results honestly. These rules recover that
breakdown approximately.

This is a keyword heuristic, not ground truth. Rules are applied in priority
order because verdicts routinely combine outcomes -- a settlement that also
orders payment, or a partial award that rejects the remainder. Procedural
terminations are matched first since they end the case regardless of merits;
among substantive outcomes, ordering the defendant to do anything means the
plaintiff prevailed at least in part.
"""

from __future__ import annotations

import re
from collections import Counter

from alarb.data import Case


UNCLASSIFIED = "unclassified"

# (label, pattern) in priority order; first match wins.
_RULES: tuple[tuple[str, str], ...] = (
    ("no_jurisdiction", r"عدم الاختصاص|عدم اختصاص|الاختصاص الولائي"),
    ("inadmissible", r"عدم قبول|عدم جواز نظر|عدم سماع"),
    ("withdrawn", r"ترك الدعوى|ترك المدعي|ترك الخصومة|ترك خصومته|تنازل|التنازل"),
    ("extinguished", r"انقضاء"),
    ("struck_out", r"شطب"),
    ("set_aside", r"صرف النظر"),
    ("settlement", r"الصلح|صلح"),
    ("obligation_granted", r"إلزام|ألزم|أُلزم|ألزمت|بإلزام|أوجبت|أوجب"),
    ("liability_established", r"إثبات المسؤولية|إثبات مسؤولية|بثبوت المسؤولية"),
    ("contract_rescinded", r"فسخ"),
    ("claim_rejected", r"رفض الدعوى|رفض دعوى|برفض|رفضت|رد الدعوى|رد دعوى"),
)

# Coarse mapping onto the three categories of the paper's Table 3. Settlements
# have no home there, so they are reported separately rather than forced into one.
PAPER_CATEGORY = {
    "obligation_granted": "for_plaintiff",
    "liability_established": "for_plaintiff",
    "contract_rescinded": "for_plaintiff",
    "claim_rejected": "for_defendant",
    "no_jurisdiction": "court_dismissal",
    "inadmissible": "court_dismissal",
    "withdrawn": "court_dismissal",
    "extinguished": "court_dismissal",
    "struck_out": "court_dismissal",
    "set_aside": "court_dismissal",
    "settlement": "settlement",
    UNCLASSIFIED: UNCLASSIFIED,
}

_COMPILED = tuple((label, re.compile(pattern)) for label, pattern in _RULES)


def classify_verdict(verdict: str) -> str:
    for label, pattern in _COMPILED:
        if pattern.search(verdict):
            return label
    return UNCLASSIFIED


def outcome_distribution(cases: list[Case]) -> dict[str, object]:
    fine = Counter(classify_verdict(case.verdict) for case in cases)
    coarse = Counter()
    for label, count in fine.items():
        coarse[PAPER_CATEGORY[label]] += count
    total = len(cases)
    majority_label, majority_count = coarse.most_common(1)[0]
    return {
        "cases": total,
        "fine_labels": {
            label: {"count": count, "share": round(count / total, 4)}
            for label, count in fine.most_common()
        },
        "paper_categories": {
            label: {"count": count, "share": round(count / total, 4)}
            for label, count in coarse.most_common()
        },
        "majority_class": {
            "label": majority_label,
            "share": round(majority_count / total, 4),
        },
        "paper_table_3": {
            "for_plaintiff": 0.62,
            "for_defendant": 0.05,
            "court_dismissal": 0.33,
        },
    }
