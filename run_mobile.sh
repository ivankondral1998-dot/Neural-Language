#!/usr/bin/env bash
# Reproducible offline Neural Language mobile-native smoke and regression checks.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN_BASE="$ROOT/runs_mobile"
mkdir -p "$RUN_BASE"
RUN="$(mktemp -d "$RUN_BASE/run_XXXXXXXX")"
CXX="${CXX:-g++}"
"$CXX" -std=c++17 -O2 -Wall -Wextra -Werror \
  "$ROOT/language/mobile_infer.cpp" -o "$RUN/mobile_infer"
python3 "$ROOT/language/mobile_pack.py" "$ROOT/language/examples/mobile_demo.nlw" -o "$RUN/demo.nlmb" > "$RUN/pack.json"
"$RUN/mobile_infer" --model "$ROOT/language/examples/mobile_demo.nlw" --input 0.25,0.75 > "$RUN/text.log"
"$RUN/mobile_infer" --model "$RUN/demo.nlmb" --input 0.25,0.75 --stats > "$RUN/binary.log"
python3 - "$RUN" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1])
a=(p/'text.log').read_text().splitlines()[0]
b=(p/'binary.log').read_text().splitlines()[0]
assert a==b,(a,b)
assert abs(float(a.split('values=')[1])-0.3)<1e-12
print('MOBILE_SMOKE_PASS',a)
PY
(
  cd "$ROOT"
  python3 -m unittest -v language.tests.test_mobile_infer language.tests.test_mobile_binary
) > "$RUN/tests.log" 2>&1
printf 'MOBILE_NATIVE_TESTS_PASS\nRESULT_DIR: %s\n' "$RUN"
