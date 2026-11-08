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

## NL-COMP-005 (typed changes/effects; isolated feature branch)

Before modifying the active Neural IR compiler, read `../docs/NEURAL_SHAPES_EFFECTS_CHANGES_20261108_RU.md` and run `python3 -m unittest -q tests.test_shapes_changes_integration` from `language/`. The compiler now cross-checks `NeuralShapes/1`, `NeuralEffects/1`, and `NeuralChanges/1` before C++ emission. Preserve the nine frozen native-output SHA-256 values and runtime semantics. This is strictly sequential dense inference/training, NOT typed DAG, ARM64 measurements, or general tensor infrastructure. Do not merge experimental DAG/mobile branches without separate tests and owner-approved architecture decisions.

## Experimental NL-COMP-006 DAG (isolated feature branch only)

Read `../docs/NL_COMP_006_DAG_CHANGES_RU.md`. Explicit ordered graph parents and `NeuralGraph/1` are experimental. Keep legacy `.nir` C++ generation byte-for-byte; do not merge into `main` without a separate compatibility review. Graph weight edits always recover with FULL; only sparse input changes can propagate incrementally on DAG. Canonical FULL activations are mandatory for graph SGD. Never claim ARM64/Android measurements or a novel algorithm on the basis of Linux CPU benchmarks. The graph `NLMODEL3` checkpoint is not supported by the existing mobile binary converter. No GitHub Actions without owner approval.
