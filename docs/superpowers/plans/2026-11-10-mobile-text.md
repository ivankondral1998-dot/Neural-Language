# Mobile language and text experiment implementation plan

Goal: reduce avoidable native memory traffic and close unsupported quantized-storage mobile codegen; add explicit BPE training profile without misrepresenting Python as native .nir.
User authorizes implementation without repeated design questions. Base 32fc5be2. No main, UNO, pretrained controls, persona data or Actions changes.

Product barrier: mobile runtime is still slower than independent C++ on the owner A52 workload; text path has a 128-byte context and no practical tokenizer.
Alternatives: optimized independent C++/BLAS, byte260 baseline, established ByteLevel BPE. No universal C++ superiority is assumed.

- [x] Native FULL overwrite test: poison warm buffers and compare exact scalar/masked/int8/DAG outputs and failure recovery. Remove redundant z/a clearing only after acceptance tests exist; benchmark paired frozen baseline/current under equal flags on fixed input streams. Keep negative/noisy results.
- [x] Mobile int8-storage specialization: add failing mixed-storage/disabled-edge tests, implement per-term FP64 dequantization and ILP rows, retain existing FP64 emitted code and strict numerical order. No integer-accumulator speed claim.
- [x] Text path: established optional ByteLevel BPE, pinned version; embed tokenizer/digest in new checkpoints; preserve old byte260 weights and APIs; streamed UTF8 training inputs and context1024 profile near55M; roundtrip/corruption/compatibility tests. No corpus/persona training in this change.
- [x] Run full native regressions with GCC/Clang if available, Python dialogue tests in separate CPU venv, sample quality-neutral token compression. Package same-source on-phone comparison with explicit unverified device status. Review changes independently, preserve report, commit only research branch.

Review focus: stale activation reads after errors; disabled-edge fallback and int8 scale; numerical reduction ordering; literal delimiter injection; tokenizer/checkpoint mismatch; changed vocabulary parameter count. Tests/benchmarks must distinguish x86 results from ARM64 hardware evidence. Runtime/compiler changes do not turn the existing PyTorch transformer into a native language backend.

Evidence: research/mobile_kernel_003/RESULTS_RU.md and evidence/final_validation.json. Whole diff reviewed independently; no P1/P2 found. Pre-existing Assistant integration errors reproduced and minimally corrected. Linux performance remains separate from ARM64 cross-compilation; no device run.
