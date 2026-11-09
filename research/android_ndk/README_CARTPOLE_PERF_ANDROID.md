# CartPole FP64 benchmark: Neural Language vs plain C++ (experimental)

Two inference engines in **one** native Android ARM64 executable, using the **same**
PPO weights, deterministic input states, compiler, numerical precision, and CPU:

- Neural Language: full C++17 `neural_runtime.hpp` with `Model::FULL`;
- Baseline: independent manually coded forward pass (4→64 tanh→64 tanh→2 linear),
  strict independent NLMODEL2 parser. No Neural Language code used for baseline
  computations. **Not** an optimized TFLite/ONNX/OpenBLAS implementation.

Both have `-O3 -DNDEBUG -ffp-contract=off -std=c++17`. Default 4096
precomputed inputs per round, 7 alternating-order paired rounds. Model load,
input generation, output logging, and 256 warmup samples are excluded from
timed regions. Per-prediction medians are compared and output checksums must
agree. 5 independent Python golden states and 512 generated states are
checked before timing, absolute/relative tolerance 1e-9.

## Android/Termux (no root, no USB, no compiler needed)

Download the bundled `Neural-Language-A52-Benchmark.tar.xz` from the research
branch, then open Termux and run:

```sh
mkdir -p ~/neural-benchmark
tar -xJf ~/storage/downloads/Neural-Language-A52-Benchmark.tar.xz -C ~/neural-benchmark
sh ~/neural-benchmark/neural_benchmark/run.sh
```

The runner checks SHA-256, records basic device metadata, and writes
`results.log`/ `errors.log` in a fresh per-run directory. It never deletes
prior outputs or trains a model. Paste the `REFERENCE_EQUIVALENCE_PASS`,
`TRIAL`, and `BENCH_PASS` lines when sharing a device result.

**Status:** until actual device output is returned, Android benchmark is
CROSS-COMPILED ONLY; it is not a physical speed test. The previously confirmed
Termux `ANDROID_DEVICE_3_PROGRAMS_PASS` covered different executables.

## Scientific limitations

The C++ baseline is a reasonably optimized *simple dense scalar* routine,
not a state-of-the-art library. Neural Language performs managed input
copying and shape/numeric checks; the baseline returns a stack array.
Consequently measured ratio includes runtime abstraction overhead rather
than isolating a novel optimization. Running both in one process means RSS
cannot be separately attributed to either engine. Do not extrapolate
beyond this model or imply an advantage in memory, energy, or quality.

**Success condition:** correct outputs and a measured ratio on a physical
Samsung A52. A ratio `neural_speedup_vs_plain_cpp > 1` would favor Neural
Language on this exact workload, while a ratio below 1 would reject a speed
advantage for this case. No passing threshold has been chosen post hoc.
