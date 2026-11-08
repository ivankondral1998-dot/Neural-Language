# Instructions for successor agents — Neural Language 0.3 integration

## MANDATORY FIRST READ — PRODUCT CHARTER (owner direction, 2026-11-08)

**Before planning, changing code, or claiming progress, read [NEURAL_LANGUAGE_PRODUCT_CHARTER.md](NEURAL_LANGUAGE_PRODUCT_CHARTER.md).** The charter controls project identity, priorities, hypotheses, evidence and definition of done. Do not substitute helper tooling or familiar neural infrastructure for the core language. The primary product is **Neural Language (.nir) with demonstrable, reproducible benefits on phone workloads**; superiority on Android has **NOT** been demonstrated.

**Anti-drift gate for every nontrivial task:** state the product barrier, expected language-level benefit, existing alternative/baseline, falsifiable acceptance test and evidence status. Reject or re-scope work that only creates another wrapper, model format or agent without supporting that barrier. Assistants with weights are **optional research**, not a required language component. No claims of originality or performance victory without an independently defensible comparison. The owner's explicit new decision supersedes this charter; agents may not silently redefine it.

**First-read order:** root AGENTS.md → NEURAL_LANGUAGE_PRODUCT_CHARTER.md → README.md → language/AGENTS.md → current source/IR specifications and tests. Root AGENTS is not automatically read by every conceivable external tool, so README also points to the charter.

- Owner's fixed goal is the specialized Neural Language programming language for AI: source semantics, IR, compiler, native execution, neural computation and measured practical advantage on phone workloads. Separate helpers/agent programs are OPTIONAL research tools, not a required part of the language. Do not switch to ISKRA, UNO or another project.
- Read README.md and language/AGENTS.md, language/compiler.py, language/runtime/neural_runtime.hpp, then tests. This is a prototype; never call it superior to C++ or verified on Android.
- Before modifications record exact SHA of compiler, runtime, source programs and test logs. Preserve all previous archives and source; no deletion. No GitHub Actions without explicit owner permission.
- Grammar implements .nir operations including validate, optimize, save_model and load_model; native executables are generated as C++17 by a Python bootstrap compiler. Runtime advisor is an optional built-in heuristic, not a required trained assistant. Self-hosting and rewriting optimization do NOT exist.
- Optimizer acceptance only for measured single-layer-edit workload. RUNS MUST compare equal outputs; do not market synthetic speed ratio as a general claim. Never infer inference memory/RSS from static budget.
- `trained.nlw` is a trusted local text format NLMODEL2. Do not load untrusted files or claim atomic secure writes. Test corrupted/truncated model inputs and no-overwrite preservation.
- To reproduce: `bash run.sh` then `python3 oracle_integration.py <result_dir>/trained.nlw`. Additional tests: `python3 -m unittest tests.test_integration` from language directory.
- Any movement into UNO, deployment/physical phone tests or architecture change requires explicit project decision. All untested gaps must remain visible in report.