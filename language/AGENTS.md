# Instructions for future AI developers

- This project is a **new language and neural compiler** (`.nir`). It is NOT ISKRA. Never replace it with an ISKRA/Python command interpreter or claim the earlier ISKRA project is its implementation.
- Keep UNO and all other user repositories untouched unless the owner explicitly requests an integration.
- Current version is HOST_TESTED on Linux x86-64 only; do not claim Android runtime validation.
- Before changes read `README_RU.md`, `docs/LANGUAGE_SPEC_RU.md`, `references/NEURAL_IR_CONTRACT_DRAFT_002.md`, and the current compiler/runtime source.
- Preserve prior versions, binaries, tests and artifacts for rollback. No deletions without owner authorization. Never trigger GitHub Actions without explicit owner approval.
- Separate compiler, IR, native runtime, DSL program, optional developer assistant. Keep execution deterministic and no implicit network/API dependency.
- The built-in advisor is heuristic and does not constitute a trained model or a self-improving intelligence. Always validate the outputs against full computation and held-out workloads before claiming a performance win.
- Test both GCC and Clang, error/shape handling and numerics. Only report tests actually run.

- Version 0.5 engineering branch adds a bounded Neural IR transform and an evidence-gated architecture assistant. Read `../docs/ENGINEERING_BASELINE_0_5_RU.md`. Never replace numerical verification with a superficial runtime PASS or assume the new syntax alone is useful.
