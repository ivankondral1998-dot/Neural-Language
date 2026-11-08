# Instructions for successor agents — Neural Language 0.3 integration

- Owner's fixed goal is a specialized language + integrated assistant for neural weight creation, training, code generation and validated optimization. Do not switch to ISKRA, UNO or another project.
- Read README_RU.md and language/AGENTS.md, compiler.py, runtime/neural_runtime.hpp, then tests. This is a prototype; never call it superior to C++ or verified on Android.
- Before modifications record exact SHA of compiler, runtime, source programs and test logs. Preserve all previous archives and source; no deletion. No GitHub Actions without explicit owner permission.
- Grammar implemented here is .nir commands validate, optimize, save_model, load_model; executor is still C++17 generated from Python compiler. The assistant's policy lives in C++ native runtime, called from .nir. Self-hosting and rewriting optimization do NOT exist.
- Optimizer acceptance only for measured single-layer-edit workload. RUNS MUST compare equal outputs; do not market synthetic speed ratio as a general claim. Never infer inference memory/RSS from static budget.
- `trained.nlw` is a trusted local text format NLMODEL2. Do not load untrusted files or claim atomic secure writes. Test corrupted/truncated model inputs and no-overwrite preservation.
- To reproduce: `bash run.sh` then `python3 oracle_integration.py <result_dir>/trained.nlw`. Additional tests: `python3 -m unittest tests.test_integration` from language directory.
- Any movement into UNO, deployment/physical phone tests or architecture change requires explicit project decision. All untested gaps must remain visible in report.