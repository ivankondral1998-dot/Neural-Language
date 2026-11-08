#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CXX="${CXX:-g++}"
mkdir -p "$ROOT/runs"
RUN_DIR="$(mktemp -d "$ROOT/runs/run_XXXXXX")"
python3 "$ROOT/language/compiler.py" "$ROOT/language/examples/integrated.nir" --compiler "$CXX" -o "$RUN_DIR/trainer" | tee "$RUN_DIR/compile_train.log"
python3 "$ROOT/language/compiler.py" "$ROOT/language/examples/inference.nir" --compiler "$CXX" -o "$RUN_DIR/inference" | tee "$RUN_DIR/compile_infer.log"
(cd "$RUN_DIR" && ./trainer | tee "$RUN_DIR/train.log")
(cd "$RUN_DIR" && ./inference | tee "$RUN_DIR/infer.log")
python3 - "$RUN_DIR" <<'PYTEST'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
train=(p/'train.log').read_text(encoding='utf-8')
infer=(p/'infer.log').read_text(encoding='utf-8')
assert 'MODEL_SAVED trained.nlw' in train
assert 'MODEL_LOADED trained.nlw' in infer
assert 'VALIDATE PASS' in train and 'VALIDATE PASS' in infer
assert (p/'trained.nlw').is_file()
assert (p/'trainer.nir.json').is_file() and (p/'inference.nir.json').is_file()
a=json.loads((p/'integrated_result.json').read_text(encoding='utf-8'))
b=json.loads((p/'inference_result.json').read_text(encoding='utf-8'))
assert a['counts']['train']==1600 and b['counts']['train']==0
assert b['counts']['changes']==0
print('END_TO_END_PASS training native->save->load->native inference; heldout checks PASS')
PYTEST
printf 'RESULT_DIR: %s\n' "$RUN_DIR"
