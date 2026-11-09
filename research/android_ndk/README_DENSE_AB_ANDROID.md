# Neural Language dense-FP64 guarded fast path: A52 A/B test

Purpose: verify whether **removing per-weight masks/storage dispatch for
known fully-enabled dense FP64 layers** improves the true `.nir` runtime
without changing its numerical results or disabling validation.

The experimental branch retains the old native runtime and source history.
The code change is a guarded `l.storage==0 && l.disabled==0` path in
`Model::forward`. Masked, quantized, graph, and BLAS paths remain unchanged.

One package contains TWO Android API26+ ARM64 executables compiled with the
same Android NDK r30 Clang flags (`-O3 -DNDEBUG -ffp-contract=off`):

- `bench_old`: original Neural Language kernel vs independent plain C++.
- `bench_dense`: guarded dense kernel vs that same independent plain C++.

Both binaries independently read the **same pretrained CartPole PPO checkpoint**,
check 517 input vectors including five independent Python-oracle goldens,
then execute 4096 input samples across seven alternating-order paired trials.
Output checksum equality and finite outputs are mandatory.
Same model, FP64 precision and toolchain, no training during measurement.

The wrapper runs OLD/DENSE in three counterbalanced pairs (OLD-DENSE,
DENSE-OLD, OLD-DENSE). It writes a fresh directory of logs and returns
six summary lines. It does not overwrite or delete existing data.

## Launch on Samsung A52 using Termux

Download `Neural-Language-A52-Dense-AB.tar.xz` from the private research branch
while logged into the owner's GitHub account. Then:

```sh
mkdir -p ~/neural-dense
tar -xJf ~/storage/downloads/Neural-Language-A52-Dense-AB.tar.xz -C ~/neural-dense
sh ~/neural-dense/neural_dense_ab/run.sh
```

Result must end in `DENSE_AB_ANDROID_PASS` and show six paired
`BENCH_PASS` lines. If the device is too warm, allow thermal stabilization
first. This is a small microbenchmark, not a memory, battery or general-purpose
language supremacy test; no optimized TFLite/ONNX/BLAS comparison yet.

**Previous phone result (owner-provided):** on SM-A525F, Android 14, ARM64,
Neural Language FULL 13.4736379395 us/inference versus independent C++
7.14565283203 us/inference; speed ratio 0.530343242422, with 517/517
correct vectors. That measurement pertains to the OLD binary, not the
new guarded path. Device results for guarded path are pending.
