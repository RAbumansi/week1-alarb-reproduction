#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# NumPy's pinned version requires 3.12+, and the pinned datasets stack uses
# a dill version that fails on 3.14. Report the mismatch before installing.
if ! "$PYTHON_BIN" -c 'import sys; sys.exit(0 if (3,12) <= sys.version_info < (3,14) else 1)'; then
  echo "Need Python 3.12 or 3.13; $PYTHON_BIN is $("$PYTHON_BIN" --version 2>&1)." >&2
  echo "Re-run with, for example: PYTHON_BIN=python3.13 bash run_week1.sh" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

.venv/bin/python -m pip install -r requirements-core.txt
HF_HOME="$PROJECT_DIR/.cache/huggingface" .venv/bin/python official_hf_loader.py
HF_HOME="$PROJECT_DIR/.cache/huggingface" .venv/bin/python audit_alarb.py
HF_HOME="$PROJECT_DIR/.cache/huggingface" .venv/bin/python prepare_splits.py

echo "Week 1 public-data reproduction and three-way split completed."
