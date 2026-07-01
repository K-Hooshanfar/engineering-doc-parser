#!/usr/bin/env bash
# Run the same quality checks as GitHub Actions CI (locally).
# Usage: ./scripts/check.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"

step() {
  echo ""
  echo "==> $1"
  shift
  "$@"
}

step "Black (check formatting)" python -m black --check .
step "Ruff (lint & import order)" python -m ruff check .

step "Pylint (score >= 6.5)" bash -c '
  pylint src/engineering_doc_parser | tee pylint_output.txt
  score=$(python - <<'"'"'EOF'"'"'
import re, pathlib, sys
txt = pathlib.Path("pylint_output.txt").read_text(errors="ignore")
m = re.search(r"rated at ([0-9]+(?:\.[0-9]+)?)/10", txt)
score = float(m.group(1)) if m else 0.0
print(score)
sys.exit(0 if score >= 6.5 else 1)
EOF
)
  echo "Pylint score: ${score}/10"
'

step "Mypy (type checking)" python -m mypy src/engineering_doc_parser
step "Pytest (coverage >= 68%)" python -m pytest --cov=engineering_doc_parser --cov-report=term-missing --cov-fail-under=68

echo ""
echo "All CI checks passed."
