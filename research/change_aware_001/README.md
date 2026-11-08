# Neural Language — Change-Aware 001

Research-only, independent of production syntax and runtime. Product charter:
`../../NEURAL_LANGUAGE_PRODUCT_CHARTER.md`.

The benchmark makes **weight changes at a fixed input** to sequential FP64 networks.
It measures applying all edits **plus** calculating a prediction, after 32 warmup
steps, for 250 recorded steps and five trials. Fixed seed: `20261108`.

Two tests:

* `change_aware_001.cpp`: standalone partial-row-recompute candidate, full scalar
  `-O3` and independent OpenBLAS GEMV, including negative cases.
* `runtime_compare.cpp`: imports the **unmodified** production
  `language/runtime/neural_runtime.hpp` (Git blob `ee71cdab32b248814aa23f26318130bd155c824f`).
  Tests actual `neural::Model::change/infer` in `FULL`, `AUTO`, `DELTA`
  against independent OpenBLAS GEMV, with the *same weights and edit stream*.

Reproduce from repository root (Linux x86-64 with Python 3, NumPy, SciPy,
GCC/Clang; SciPy/NumPy optimized OpenBLAS libraries must exist):

```bash
python3 research/change_aware_001/run_experiment.py \
  --compiler g++ --output /tmp/neural-comp001-independent-gcc
python3 research/change_aware_001/run_experiment.py \
  --compiler clang++ --output /tmp/neural-comp001-independent-clang
python3 research/change_aware_001/run_runtime_compare.py \
  --compiler g++ --output /tmp/neural-comp001-runtime-gcc
python3 research/change_aware_001/run_runtime_compare.py \
  --compiler clang++ --output /tmp/neural-comp001-runtime-clang
python3 research/change_aware_001/verify_evidence.py
```

Scripts compile with `-std=c++17 -O3 -march=native -fno-fast-math` and force
`OPENBLAS_NUM_THREADS=1` (also OMP/MKL=1). The internal runtime uses NumPy's
OpenBLAS ILP64, the external comparator SciPy's OpenBLAS LP64; this is a
limitation. The full-vs-auto comparison in the production runtime uses exactly
one library and matching API costs, and is the primary evidence.

Experiment parameters, acceptance thresholds and falsifiers were written before
measurement in `RESEARCH_PLAN_001_RU.md` and `RUNTIME_PLAN_001_RU.md`.
`evidence/` contains the saved summary data, not the full run logs; the attached
ZIP includes per-scenario logs and native build outputs.

**Do not merge this research branch into main by default.** These are Linux
x86-64 results, not Android, ARM64, whole-application power, or original
computing theory. `REPORT_001_RU.md` gives conclusions and remaining blockers.
