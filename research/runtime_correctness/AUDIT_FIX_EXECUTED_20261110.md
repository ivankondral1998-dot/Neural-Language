# Neural Language — audited runtime correctness repair (2026-11-10)

**Scope:** isolated branch `fix/runtime-correctness-audit-20261110`
from `research/arm64-dense-kernel-20261110` at
`452e0c67f37fee6e17f74eabe06fa691daed7ca5`.
No merge, no GitHub Actions, no file deletion.

## Reproduced BEFORE patch

The exact checked-in C++ audit regression compiled with the original runtime
Git blob `4dd25818fb5643ad425c8c26628c4868c20f7701` and failed:

```text
AUDIT_RUNTIME_REGRESSIONS_FAIL FULL/DELTA cancellation mismatch: got=0.000000 expected=5.000000
```

After fixing numerical fallback, the suite progressed to the mixed-events
exception, and after fixing mixed events it progressed to the original SGD
failure (`SGD exception changed full count`). This is evidence of three
separate faults, not one repeated assertion.

## Runtime changes

Patched runtime `language/runtime/neural_runtime.hpp` Git blob
`d044c3153324a2d72fdc5d86ac0d1073b008c763`:

1. **Numerical cancellation and mode switch:** if a cached partial update
   cancels large preactivations (or yields a non-finite/out-of-domain
   intermediate), fall back to a canonical FULL forward. Track whether a
   cached result came from incremental propagation; switching to FULL forces
   recanonicalization. Failed partial execution invalidates cache.
2. **Mixed input and weight changes:** `change_input`/`mutate` no longer
   eagerly infer a temporarily invalid intermediate combination. Instead
   invalidate old cache and commit through the next observable inference or
   learning step.
3. **Strong exception safety for SGD:** train on an `isolated_copy()`,
   including canonical FULL and all weight/bias updates, then swap layer
   arrays only on success. Failed learning preserves weights, cache, version,
   and counters. **Tradeoff:** temporary duplicate model memory during
   learning; mobile peak RSS/throughput must be measured before integration.

The new counterexamples are retained as source, not merely as this note:
`language/tests/audit_runtime_regressions.cpp` and
`language/tests/test_audit_runtime.py`, with a real compiled
`.nir` fixture and matching `NLMODEL2` checkpoint.

## Executed checks on Ubuntu x86-64

Environment: **GCC 15.2.0**, **Python 3.14.4**. No Android execution.

- Baseline C++ regression **RED** before patch.
- Patched C++17 `-O3 -DNDEBUG -ffp-contract=off -Wall -Wextra -Werror`:
  `AUDIT_RUNTIME_REGRESSIONS_PASS` and explicit passes for
  chain AUTO/DELTA, DAG AUTO/DELTA, FULL mode switch, both mixed-event
  orderings and sequential/DAG failed SGD atomicity.
- Compiled `audit_cancellation.nir` with the real compiler and executed it:
  `PREDICT version=1 values=0`,
  `PREDICT version=2 values=5` (twice),
  `EXPECT PASS tolerance=1e-09`.
- Complete Python unit-test discovery after runtime and test changes:
  **96 total, 88 passed, 8 skipped, 0 failed**.
  The skipped cases require Clang not present in this sandbox.
- `bash run.sh`: `END_TO_END_PASS`, native learning, checkpoint
  persistence, independent reload and inference.
- `python3 oracle_integration.py` checked resulting weights against
  independent expected predictions: `ORACLE PASS`.
  Trained model SHA-256:
  `ef9c8a6ee12847bb785b3e822d3a115bd26122386812f7b5dc60f3f38a30cc0d`,
  identical to the earlier known checkpoint.
- All committed source and test files' Git blob SHAs matched the tested
  sandbox copies. `main` remained at
  `1168f3fcc2ecd0bd17de08be69d313844790c845`.

## Not yet established

- **Not a universal numerical proof.** The conservative cancellation
  guard protects the audited failures and tested trajectories, but arbitrary
  FP64 cancellation may still evade the heuristic; a rigorous
  sensitivity/error-budget contract remains an open engineering task.
- No actual patched ARM64/A52 execution, Android measurements or RSS/energy
  testing. No cross-compiler Clang run.
- Full IR semantic closure and the audited invalid-DAG-`verify` IR
  bypass remain **OPEN**, outside this limited runtime repair.
- The isolated SGD copy increases temporary peak memory; evaluate it before
  any mobile integration. No performance-speedup claims are made.

**Gate:** no merge into `main` until numerical semantics, resource limits
and device behavior have been assessed; original code remains available
through branch history.
