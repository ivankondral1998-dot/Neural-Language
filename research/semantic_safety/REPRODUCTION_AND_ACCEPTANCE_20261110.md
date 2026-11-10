# Neural Language — semantic-safety repair: reproduced tests and boundaries

Date: 2026-11-10. Branch `fix/semantic-safety-20261110`, derived from
`research/arm64-dense-kernel-20261110` HEAD `452e0c67f37fee6e17f74eabe06fa691daed7ca5`.
No changes to `main`; no GitHub Actions, merges, deletes or APK releases.

## Original failing baseline

Before changing the runtime, compiled `language/tests/runtime_semantic_safety.cpp`
with GCC `-std=c++17 -O2 -Wall -Wextra -Werror` against the **original**
runtime blob `4dd25818fb5643ad425c8c26628c4868c20f7701`.
All **9/9 cases failed** on the original runtime (confirmed again after adding DAG SGD):
- cancellation: chained DELTA/AUTO, DAG DELTA/AUTO (expected 5, actual 0);
- FULL after incremental: no fresh canonical pass;
- mixed edits: weight→input and input→weight each caused premature
  invalid intermediate calculation;
- failed SGD: weights mutated partially before exception in both sequential
  and branched DAG models.

## Implemented changes (NOT a global numeric proof)

Runtime blob: `e7e8d53d44a1c5bde00c1dc65dfc3f800dde5600`.

- Combined input+weight events no longer force hidden intermediate `infer`;
  they invalidate a cached plan and evaluate final committed state on demand.
- SGD (sequence and DAG) computes all proposed updates on staged layers, then
  publishes them in a single vector swap only after validation passes.
  **Cost:** temporary copy of weights/masks/biases during learning;
  allocation overhead and mobile RSS not measured.
- FULL after incremental inference now invalidates a noncanonical cache.
  Partial forward errors leave the cache invalid instead of valid-but-partial.
- Incremental calculations with near-total cancellation or out-of-bounds
  intermediate estimates fall back to canonical FULL. This catches the audit
  examples but uses a conservative heuristic, **not an analytical bound on
  accumulated FP64 error**. No claim that FULL/AUTO/DELTA are globally
  equivalent on all finite inputs. Future work: define/verify numerical
  contract and sound propagation error bounds.

## Executed verification in isolated Linux x86-64 sandbox

Original 40 text source files verified against Git blob SHAs.
- Regression program: **9/9 PASS** using both `g++` and `clang++`;
  `-std=c++17 -O2 -Wall -Wextra -Werror`.
- Two permanent Python tests, each testing both compilers:
  `python3 -m unittest tests.test_semantic_safety -v` **2/2 PASS**;
  verifies compiled `.nir` cancellation including `expect [5]`.
- Existing test discovery **94/94 PASS**, 0 skips reported:
  `cd language && python3 -m unittest discover -s tests -v`.
  This discovery preceded addition of the two new test files, but runtime
  was already the patched version. The two new tests ran separately.
- `bash run.sh`: **END_TO_END_PASS** for actual .nir training, checkpoint
  save, separately compiled inference and heldout oracle.
- Clang `-fsanitize=address,undefined` new nine-case regression **PASS**,
  no sanitizer failure or leak diagnostic in this run.
- Native stress with Clang:
  `CHANGE_INPUT_STRESS_PASS comparisons=64512
  worst_scaled_error=4.22653367192464e-16`;
  `DAG_CHANGE_PASS checks=15360
  worst_scaled=6.094909692561681e-16
  delta_steps=649 train_steps=61`.

These are source/host tests; the repaired runtime has **not** been
recompiled for Android or measured on SM-A525F. The A52 benchmark
`1.66x` improvement remains historical evidence for the original dense
kernel, not for this new branch. No TFLite or ONNX comparison.

## Next gates

1. Formally bound DELTA rounding and cache provenance, or use strict FULL
   fallback wherever that bound cannot guarantee the target tolerance.
2. Add negative/tampered IR tests and end-to-end update-order cases beyond
   the present C++ runtime surface.
3. Then design compiler-generated streaming `.nir` steps and compare on A52.
