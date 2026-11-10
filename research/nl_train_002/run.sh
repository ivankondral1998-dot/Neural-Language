#!/bin/sh
# Run from any directory. Requires local C++17 compiler and Python3.
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(CDPATH= cd -- "$here/../.." && pwd)
run=$(mktemp -d)
printf 'run_dir=%s\n' "$run"
cp "$here/targets.txt" "$here/features.txt" "$run/"
python3 "$root/language/compiler.py" "$here/regression_train.nir" -o "$run/train"
python3 "$root/language/compiler.py" "$here/regression_infer.nir" -o "$run/infer"
(cd "$run" && ./train && ./infer > predictions.log)
python3 - "$run/predictions.log" <<'PY'
import sys
from pathlib import Path
preds=[]
for line in Path(sys.argv[1]).read_text().splitlines():
    if "DATA_PREDICT index=" in line:
        preds.append([float(v) for v in line.split("values=")[-1].split(",")])
expected=[(.25,.75),(.8,.2),(.5,.5),(.1,.9)]
assert len(preds)==4, "missing prediction rows"
error=max(abs(got-want) for row,goal in zip(preds,expected)
          for got,want in zip(row,goal))
assert error<0.08, f"regression error too high: {error}"
print(f"REGRESSION_PASS rows=4 max_abs_error={error:.12g}")
PY
