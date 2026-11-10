# SM-A525F Android 14 — OLD vs guarded dense A/B (owner device)

Source: actual Termux stdout pasted by owner in the project conversation.
Run: `dense_ab_20261110_030425_18013`.
Device model, OS, ABI were independently reported **by owner's `getprop` output** in this conversation: SM-A525F, Android 14, arm64-v8a. Logs below are **owner-supplied**, not retrieved directly from the phone.

Executable bundle: `research/android_ndk/artifacts/Neural-Language-A52-Dense-AB.tar.xz`; old and optimized runtime are separate NDK r30 AArch64 binaries with the same checkpoint and independent C++ FP64 comparison.

```text
DEVICE_SAMPLE pair=1 variant=old
BENCH_PASS samples=4096 trials=7 neural_median_us_per_infer=16.8411508789 plain_cpp_median_us_per_infer=8.89067333984 neural_speedup_vs_plain_cpp=0.527913644606 max_abs_error=0 checksum_abs_error=0 engine_neural=FULL engine_baseline=independent_cpp_fp64 note=mobile_on_device_needed_no_tflite_comparison
DEVICE_SAMPLE pair=1 variant=dense
BENCH_PASS samples=4096 trials=7 neural_median_us_per_infer=10.1623918457 plain_cpp_median_us_per_infer=8.88051342773 neural_speedup_vs_plain_cpp=0.873860559853 max_abs_error=0 checksum_abs_error=0 engine_neural=FULL engine_baseline=independent_cpp_fp64 note=mobile_on_device_needed_no_tflite_comparison
DEVICE_SAMPLE pair=2 variant=dense
BENCH_PASS samples=4096 trials=7 neural_median_us_per_infer=10.0894799805 plain_cpp_median_us_per_infer=8.88042456055 neural_speedup_vs_plain_cpp=0.88016672591 max_abs_error=0 checksum_abs_error=0 engine_neural=FULL engine_baseline=independent_cpp_fp64 note=mobile_on_device_needed_no_tflite_comparison
DEVICE_SAMPLE pair=2 variant=old
BENCH_PASS samples=4096 trials=7 neural_median_us_per_infer=16.788050293 plain_cpp_median_us_per_infer=8.89180493164 neural_speedup_vs_plain_cpp=0.529650839524 max_abs_error=0 checksum_abs_error=0 engine_neural=FULL engine_baseline=independent_cpp_fp64 note=mobile_on_device_needed_no_tflite_comparison
DEVICE_SAMPLE pair=3 variant=old
BENCH_PASS samples=4096 trials=7 neural_median_us_per_infer=16.8337124023 plain_cpp_median_us_per_infer=8.96199536133 neural_speedup_vs_plain_cpp=0.532383775315 max_abs_error=0 checksum_abs_error=0 engine_neural=FULL engine_baseline=independent_cpp_fp64 note=mobile_on_device_needed_no_tflite_comparison
DEVICE_SAMPLE pair=3 variant=dense
BENCH_PASS samples=4096 trials=7 neural_median_us_per_infer=10.1227441406 plain_cpp_median_us_per_infer=8.8832980957 neural_speedup_vs_plain_cpp=0.877558295685 max_abs_error=0 checksum_abs_error=0 engine_neural=FULL engine_baseline=independent_cpp_fp64 note=mobile_on_device_needed_no_tflite_comparison
DENSE_AB_ANDROID_PASS
LOG_DIR=/data/data/com.termux/files/home/neural-dense/neural_dense_ab/dense_ab_20261110_030425_18013
```

The median of the three **pair-level medians**: OLD Neural 16.8337124023 µs, guarded-dense Neural 10.1227441406 µs, dense-paired independent C++ 8.88051342773 µs.

- Dense vs old: 1.6629x faster, ~39.87% less latency in this repeated paired test.
- Dense vs plain C++: ~1.1399x longer latency; **Neural Language still loses** on this workload.
- Each pair reported max_abs_error=0 and checksum_abs_error=0; this is a bounded reproducibility result, not proof of general correctness.
- Compared to owner's previous `BENCH_PASS` 13.4736 vs 7.1457 µs (older, separate run), both implementations are slower in the new session; comparisons **must be within paired session**, not across device runs.
- No TFLite/ONNX/BLAS optimized competitor, peak memory, energy, device temperature/frequency or confidence intervals collected.
- Store the negative C++ comparison as evidence. **Do not claim language superiority.**

The remaining question is which portion of the ~1.24 µs/step overhead comes from Neural Language model state management versus per-layer loops. Isolate that with a same-device equivalent benchmark **before** further speculative optimizations; do not replace the compiler or claim a novel algorithm on this evidence alone.
