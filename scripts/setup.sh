#!/usr/bin/env bash
# Prepare a local checkout and download the pinned Sharpa Wave hand assets.
set -euo pipefail
cd "$(dirname "$0")/.."
requirements=requirements-worldmodel-lock.txt
case "${1:-}" in
  ""|--worldmodel) ;;
  --minimal) requirements=requirements-lock.txt ;;
  *) echo "Usage: bash scripts/setup.sh [--worldmodel|--minimal]" >&2; exit 2 ;;
esac
if [[ ! -x .venv/bin/python ]]; then
  "${PYTHON_BIN:-python3.12}" -m venv .venv
fi
.venv/bin/python -m pip install -r "$requirements"
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python scripts/fetch_assets.py
.venv/bin/python -c 'from cube_pick.varied_boxes import VariedBoxEnv; VariedBoxEnv(seed=0).close()'
echo 'Environment ready. Run tests with: .venv/bin/python -m unittest discover -s tests -v'
