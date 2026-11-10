#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/runs"
RUN_DIR="$(mktemp -d "$ROOT/runs/assistant_XXXXXX")"
python3 "$ROOT/language/assistant.py" \
  "$ROOT/language/examples/assistant_hinge_baseline.nir" \
  --evaluation "$ROOT/language/examples/assistant_hinge_evaluation.json" \
  --compiler "${CXX:-g++}" \
  --output-dir "$RUN_DIR/result"
printf '\nRESULT_DIR: %s\n' "$RUN_DIR/result"
