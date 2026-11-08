# Instructions for successor agents — Neural Language 0.3 integration

- Owner's fixed goal is the specialized Neural Language programming language for AI: source semantics, IR, compiler, native execution and neural computation. Separate helpers/agent programs are OPTIONAL research tools, not a required part of the language. Do not switch to ISKRA, UNO or another project.
- Read README_RU.md and language/AGENTS.md, compiler.py, runtime/neural_runtime.hpp, then tests. This is a prototype; never call it superior to C++ or verified on Android.
- Before modifications record exact SHA of compiler, runtime, source programs and test logs. Preserve all previous archives and source; no deletion. No GitHub Actions without explicit owner permission.
- The compiler implements .nir commands including validate, optimize, save_model and load_model. Native executables use generated C++17, with Python as a bootstrap compiler. The optional built-in optimization heuristic only chooses a runtime strategy; it does not rewrite code or train a separate assistant.
- Optimizer acceptance only for measured single-layer-edit workload. RUNS MUST compare equal outputs; do not market synthetic speed ratio as a general claim. Never infer inference memory/RSS from static budget.
- `trained.nlw` is a trusted local text format NLMODEL2. Do not load untrusted files or claim atomic secure writes. Test corrupted/truncated model inputs and no-overwrite preservation.
- To reproduce: `bash run.sh` then `python3 oracle_integration.py <result_dir>/trained.nlw`. Additional tests: `python3 -m unittest tests.test_integration` from language directory.
- Any movement into UNO, deployment/physical phone tests or architecture change requires explicit project decision. All untested gaps must remain visible in report.

## Current foundation review (2026-11-08)

- The baseline is `main` at `1bae50d72bffd1b0859bf75f85fceb523c02f676` (0.3). The 0.4 DAG and 0.5 assistant are independent unmerged experimental branches. Ten shared-weight workers are a separate experiment.
- Read `docs/FOUNDATION_CONTRACT_AND_AUDIT_20261108_RU.md` before any language architecture work. The current audit branch fixes only duplicate `init`; do not merge sidecar research by default.
- A code change needs a failing reproducible case, explicit semantics/criteria, independent check when mathematical behavior changes, and legacy regression tests. Keep the fixed owner goal separate from our proposed implementation choices.
