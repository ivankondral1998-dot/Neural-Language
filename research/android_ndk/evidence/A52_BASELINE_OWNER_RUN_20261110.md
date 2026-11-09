# Baseline A52 benchmark — owner-supplied real device output

Evidence source: pasted Termux text in this project's conversation.
Device identity from separate command: `SM-A525F`, Android `14`, ABI `arm64-v8a`.
Native source: `research/android_ndk/bench_cartpole_compare.cpp` blob
`b4a56ec094e1acfc493d3752fa0b179a7f9def51`, unoptimized generic
runtime `neural_runtime.hpp` blob `faf78c47c60cf325152161e258978177be2e5cf9`.

```text
REFERENCE_EQUIVALENCE_PASS vectors=517 max_abs_error=0
TRIAL index=1 first=neural neural_us_per_infer=13.4571201172 plain_cpp_us_per_infer=7.12732958984
TRIAL index=2 first=plain_cpp neural_us_per_infer=13.4083557129 plain_cpp_us_per_infer=7.14544921875
TRIAL index=3 first=neural neural_us_per_infer=13.3579382324 plain_cpp_us_per_infer=7.14580541992
TRIAL index=4 first=plain_cpp neural_us_per_infer=13.505782959 plain_cpp_us_per_infer=7.15855908203
TRIAL index=5 first=neural neural_us_per_infer=13.4736379395 plain_cpp_us_per_infer=7.14565283203
TRIAL index=6 first=plain_cpp neural_us_per_infer=13.5021845703 plain_cpp_us_per_infer=7.15155273437
TRIAL index=7 first=neural neural_us_per_infer=13.509902832 plain_cpp_us_per_infer=7.1388371582
BENCH_PASS samples=4096 trials=7 neural_median_us_per_infer=13.4736379395 plain_cpp_median_us_per_infer=7.14565283203 neural_speedup_vs_plain_cpp=0.530343242422 max_abs_error=0 checksum_abs_error=0 engine_neural=FULL engine_baseline=independent_cpp_fp64 note=mobile_on_device_needed_no_tflite_comparison
LOG_DIR=/data/data/com.termux/files/home/neural-benchmark/neural_benchmark/bench_run_20261110_023909_3373
```

Interpretation: Neural Language was approximately 1.8856 times slower than
simple independent C++ on this microbenchmark. Correctness checks passed, but
battery, memory and optimized third-party frameworks have not been tested.
This evidence is not a direct file pulled from the device; it is the owner's
console paste. Preserve without rewriting negative findings.

The guarded dense FP64 fast path lives on `research/arm64-dense-kernel-20261110`
and **has not yet been measured on this device**.
