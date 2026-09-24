"""Load ALARB development splits and keep the validation holdout sealed.

DATA_SPLIT_POLICY.md states the holdout rule; this module enforces it. Ordinary
code calls load_split(), which can only ever reach data/development/. Reading
the holdout requires load_sealed_validation() with an explicit token, and every
such call is recorded in artifacts/holdout_access_log.json.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


DEVELOPMENT_DIR = Path("data/development")
HOLDOUT_PATH = Path("data/validation_holdout/validation.parquet")
HOLDOUT_ACCESS_LOG = Path("artifacts/holdout_access_log.json")
DEVELOPMENT_SPLITS = ("train", "test")
FINAL_EVALUATION_TOKEN = "UNSEAL-ALARB-VALIDATION-FOR-FINAL-LOCKED-EVALUATION"


class HoldoutSealedError(RuntimeError):
    """Raised when development code reaches for the validation holdout."""


@dataclass(frozen=True)
class Case:
    case_id: str
    case_facts: tuple[str, ...]
    court_reasoning: tuple[str, ...]
    applicable_laws: tuple[str, ...]
    verdict: str


def case_id(case_facts: list[str] | tuple[str, ...]) -> str:
    """Stable identifier for a case. The dataset ships no id field."""
    joined = "\n".join(case_facts).strip()
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _read_cases(path: Path) -> list[Case]:
    return [
        Case(
            case_id=case_id(row["case_facts"]),
            case_facts=tuple(row["case_facts"]),
            court_reasoning=tuple(row["court_reasoning"]),
            applicable_laws=tuple(row["applicable_laws"]),
            verdict=row["verdict"],
        )
        for row in pq.read_table(path).to_pylist()
    ]


def load_split(name: str) -> list[Case]:
    if name not in DEVELOPMENT_SPLITS:
        raise HoldoutSealedError(
            f"load_split() serves {DEVELOPMENT_SPLITS} only, not {name!r}. The "
            "validation split is the sealed final holdout; see DATA_SPLIT_POLICY.md."
        )
    path = DEVELOPMENT_DIR / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Run: bash run_week1.sh")
    return _read_cases(path)


def load_development() -> list[Case]:
    """Every development case: the publisher's training split, train + test."""
    return [case for name in DEVELOPMENT_SPLITS for case in load_split(name)]


def load_sealed_validation(token: str, reason: str) -> list[Case]:
    """Open the final holdout. Do this once, when the pipeline is frozen."""
    if token != FINAL_EVALUATION_TOKEN:
        raise HoldoutSealedError(
            "The validation holdout is sealed. Opening it requires the explicit "
            "token and is permitted only for the single final evaluation, after "
            "the pipeline and metrics are frozen. See DATA_SPLIT_POLICY.md."
        )
    if not reason.strip():
        raise HoldoutSealedError("Opening the holdout requires a written reason.")

    entries = []
    if HOLDOUT_ACCESS_LOG.is_file():
        entries = json.loads(HOLDOUT_ACCESS_LOG.read_text(encoding="utf-8"))
    entries.append(
        {
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason.strip(),
            "path": str(HOLDOUT_PATH),
        }
    )
    HOLDOUT_ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    HOLDOUT_ACCESS_LOG.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return _read_cases(HOLDOUT_PATH)
