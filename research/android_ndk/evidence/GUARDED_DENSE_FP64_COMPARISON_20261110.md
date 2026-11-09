# Neural Language — guarded dense FP64 kernel performance and correctness gate

**Status:** code and Android A/B binaries committed. Linux comparisons completed.
**Optimized runtime NOT YET MEASURED on physical Android.** Do not merge to
main or advertise a mobile speedup before on-device acceptance.

## Why this change

Owner's Termux Samsung SM-A525F Android 14 (arm64-v8a) benchmark of the
original FULL path: Neural Language median 13.4736379395 µs per prediction
versus independent plain scalar C++ 7.14565283203 µs. Neural/C++ speed ratio
0.530343242422. All 517 checked vectors matched, checksum error zero.
Raw user output preserved in `A52_BASELINE_OWNER_RUN_20261110.md`.

The previous generic loop called `Layer::effective` for every scalar multiply,
even when all edges were enabled, causing mask and storage dispatch overhead.

## Source changes

- New isolated branch: `research/arm64-dense-kernel-20261110`.
- Patch to `language/runtime/neural_runtime.hpp`: if `storage==0` and
  `disabled==0`, take raw FP64 weights directly in the original row/column
  accumulation order. Preserve per-row `finite` validation, activation and
  lifetime/cache/SGD semantics. Otherwise use the **untouched** generic loop;
  BLAS path takes priority as before.
- Parent runtime Git blob: `faf78c47c60cf325152161e258978177be2e5cf9`.
- Patched runtime Git blob: `4dd25818fb5643ad425c8c26628c4868c20f7701`.
- Independent baseline code unchanged:
  `research/android_ndk/bench_cartpole_compare.cpp`,
  blob `b4a56ec094e1acfc493d3752fa0b179a7f9def51`.
- Pretrained policy original checkpoint SHA-256:
  `a0738c97790d61dbd7b0fe2bde1ece59ff5041457ac4c77c8aae99c0be3cf8da`.

## Executed tests

1. Pulled nine files from the private GitHub branch and parent into an
   isolated Ubuntu sandbox. **9/9 Git blob SHA matched**, including old/new
   runtime and exact weights.
2. Same GCC, flags `-std=c++17 -O3 -DNDEBUG -ffp-contract=off -Wall -Wextra -Werror`.
   Identical 4096 inputs, 7 alternating-order trials; both builds
   `REFERENCE_EQUIVALENCE_PASS vectors=517 max_abs_error=0` and
   `BENCH_PASS`, checksum error zero:
   - Old runtime: median Neural **9.90171630859 µs**, plain C++ **6.24361206055 µs**;
     Neural/C++ ratio **0.63055856843**.
   - New dense runtime: median Neural **5.66034887695 µs**, plain C++ **5.16983984375 µs**;
     Neural/C++ ratio **0.913342968099**.
   Since sequential runs can change CPU frequency and baseline time, the
   ratio must be remeasured in counterbalanced pairs on the target device.
3. Same actual compiler-generated `integrated.nir` and `inference.nir`
   compiled separately with OLD and NEW headers. Both produced training
   `PREDICT version=1600 values=0.3`, separate load
   `PREDICT version=1 values=0.3`, and `VERIFY PASS rounds=40`.
   Their `trained.nlw` checkpoints were **byte-identical**:
   SHA-256 `ef9c8a6ee12847bb785b3e822d3a115bd26122386812f7b5dc60f3f38a30cc0d`.
4. `research/android_ndk/probe_dense_guard.cpp` compiled separately with
   original/patched runtime: 400 states covering FP64 dense, masked
   `gate`, INT8 layers, disabled/re-enabled gates and a weight change.
   **400/400 output rows byte-identical**; both outputs SHA-256
   `49ad79138d106e73cabc540f8bb2b932a5b447d9c9e7ed489a43a89c54664c05`.
5. `run_dense_ab_android.sh` syntax PASS and **6 host runs** (three
   counterbalanced OLD/DENSE pairs) with reference equivalence and
   `BENCH_PASS` on each.
6. Android NDK r30 `aarch64-linux-android26-clang++`, identical
   `-O3 -DNDEBUG -ffp-contract=off -static-libstdc++` for both.
   Two Android ELF64 AArch64 files inspected with
   `/system/bin/linker64`, depending only on system libc, libdl, libm:
   - `bench_old`: SHA-256
     `2845f291290e98065077b569f9fca5a7dd7fd1a605d146342ff17621980e44dc`.
   - `bench_dense`: SHA-256
     `cc501a7426a12ee19809794141e172c1d49a67ae74107e1a38e2efd44cf63a8b`.
7. **Retained** archive:
   `research/android_ndk/artifacts/Neural-Language-A52-Dense-AB.tar.xz`,
   259568 bytes, SHA-256
   `97523f7d67d33c863a4749efc695236d2d79950fca4782c5ef095f98778c921f`,
   Git blob SHA `4735195b53744c8207f30e1df8ff37e82a483bd8`.
   `xz -t` and `tar -tJf` passed in the builder. GitHub tree entries
   verify same original blob.

## Gaps

- No measured dense-kernel latency on Samsung yet. Await device A/B output.
- No full language test suite run after patch; the above scenario regressions
  are strong but bounded. No TFLite, ONNX, optimized BLAS, memory or battery.
- These kernels use ordinary FP64 arithmetic, not a new algorithm. This patch
  removes avoidable overhead in Neural Language and does not alone establish
  an advantage over optimized platforms.
- No Actions were triggered; no merge; no deletion of prior artifacts.
