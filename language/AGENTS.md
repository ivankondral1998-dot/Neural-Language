# Instructions for future AI developers

## Required product gate

This directory follows the **root** [NEURAL_LANGUAGE_PRODUCT_CHARTER.md](../NEURAL_LANGUAGE_PRODUCT_CHARTER.md), which must be read before language/compiler/runtime changes. The product is the .nir language with measurable advantages on phone workloads, **not** an assistant, JNI wrapper, model serializer, generic Python DSL or a collection of utilities. Keep existing semantics and independent numerical evidence. Propose novel mechanisms as falsifiable experiments; compare with relevant existing implementations before calling anything an invention or a win. Explicitly separate achieved results from Android hypotheses.

For each substantial change: identify the language-level barrier, source/IR contract, baseline and acceptance test; run legacy .nir regressions and independent oracle where appropriate. If a change is only another infrastructure module, justify why it is necessary to reach the product goal. Read AGENTS.md at repository root and README.md as well.

- This project is a **new language and neural compiler** (`.nir`). It is NOT ISKRA. Never replace it with an ISKRA/Python command interpreter or claim the earlier ISKRA project is its implementation.
- Keep UNO and all other user repositories untouched unless the owner explicitly requests an integration.
- Current version is HOST_TESTED on Linux x86-64 only; do not claim Android runtime validation.
- Before changes read `README_RU.md`, `docs/LANGUAGE_SPEC_RU.md`, `references/NEURAL_IR_CONTRACT_DRAFT_002.md`, and the current compiler/runtime source.
- Preserve prior versions, binaries, tests and artifacts for rollback. No deletions without owner authorization. Never trigger GitHub Actions without explicit owner approval.
- Separate compiler, IR, native runtime, DSL program, optional developer assistant. Keep execution deterministic and no implicit network/API dependency.
- The built-in advisor is heuristic and does not constitute a trained model or a self-improving intelligence. Always validate the outputs against full computation and held-out workloads before claiming a performance win.
- Test both GCC and Clang, error/shape handling and numerics. Only report tests actually run.
