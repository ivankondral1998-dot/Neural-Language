#!/usr/bin/env bash
# Compile two separate binaries and cross-check a DAG's saved weights.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CXX="${CXX:-g++}"
mkdir -p "$ROOT/runs_graph"
RUN_DIR="$(mktemp -d "$ROOT/runs_graph/run_XXXXXX")"
python3 "$ROOT/language/compiler.py" "$ROOT/language/examples/graph_train.nir" --compiler "$CXX" -o "$RUN_DIR/trainer" > "$RUN_DIR/build_train.log"
python3 "$ROOT/language/compiler.py" "$ROOT/language/examples/graph_inference.nir" --compiler "$CXX" -o "$RUN_DIR/inference" > "$RUN_DIR/build_infer.log"
(cd "$RUN_DIR" && ./trainer > "$RUN_DIR/train.log" && ./inference > "$RUN_DIR/infer.log")
python3 "$ROOT/oracle_graph.py" "$RUN_DIR/graph_trained.nlw" > "$RUN_DIR/oracle.log"
python3 - "$RUN_DIR" <<'PY'
from pathlib import Path
import re,sys
p=Path(sys.argv[1]);t=(p/'train.log').read_text();i=(p/'infer.log').read_text()
first=re.findall(r'PREDICT version=\d+ values=([^\n]+)',t)
second=re.findall(r'PREDICT version=\d+ values=([^\n]+)',i)
assert first==second and first, (first,second)
assert t.count('VALIDATE PASS')==2 and i.count('VALIDATE PASS')==2
assert 'GRAPH_ORACLE_PASS' in (p/'oracle.log').read_text()
print('GRAPH_END_TO_END_PASS full DAG -> SGD -> model v3 -> fresh native inference -> independent oracle')
PY
printf 'GRAPH_RESULT_DIR: %s\n' "$RUN_DIR"
