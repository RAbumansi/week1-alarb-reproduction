#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

.venv/bin/python -m pip install -r requirements.txt
HF_HOME="$PROJECT_DIR/.cache/huggingface" .venv/bin/python official_hf_loader.py
HF_HOME="$PROJECT_DIR/.cache/huggingface" .venv/bin/python audit_alarb.py

echo "Week 1 public-data reproduction completed."

