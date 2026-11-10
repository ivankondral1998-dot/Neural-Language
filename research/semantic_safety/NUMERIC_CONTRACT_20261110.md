# Neural Language: numeric contract (research, NOT normative final)

Date: 2026-11-10. Branch: `fix/semantic-safety-20261110`.

## Product barrier

The language currently allows `tune full`, `tune incremental` and `tune auto`.
Users must know whether those modes denote exact semantics or numerical
approximations. Silent equivalence claims would invalidate training and
inference benchmarks on phones.

## Canonical reference

For a finite FP64 input, weights and biases, `FULL` is the **reference
evaluation order of this C++ runtime**, not an infinitely precise real-number
oracle. The dense loop starts from bias and accumulates terms left-to-right,
with the compiler and FP environment as deployed. BLAS may use a different
order; therefore bitwise equivalence across BLAS, CPUs, compilers or
`-ffast-math` is NOT promised.

`DELTA` and `AUTO` may reuse cached preactivations. Because IEEE-754
addition is non-associative, `FULL` and `DELTA` **cannot generally promise
bitwise equality** if they use different accumulation orders.

A minimal witness with one linear neuron:

```text
weights = [1e16, 1, -1e16], bias = 0
input0  = [1, 0, 1]
FULL(input0) = (1e16 + 0) - 1e16 = 0
input1  = [1, 1, 1] (change input[1] by +1)
FULL(input1) = (1e16 + 1) - 1e16 = 0 in FP64
naive DELTA(input1) = cached(0) + 1 = 1
```

The pre-2026-11-10 cancellation heuristic only checked the updated
preactivation and the incoming delta; it did **not** catch this witness.
This is a counterexample to any unqualified global FULL/DELTA equivalence
claim. A passing differential stress test does not erase it.

## Implemented risk reduction (NOT a numerical proof)

On canonical forward in DELTA/AUTO mode, record whether a neuron contains
a term with magnitude greater than `1e8 * max(1, abs(final_sum))`.
If so, refuse incremental reuse of the cached state and use FULL on the
next edit. For BLAS-produced caches, also refuse incremental reuse since
its internal accumulation order is not inspected. This catches the witness.

The earlier heuristic also falls back on a large or nearly cancelling
individual incremental update, and on a bounded delta streak. The
combined heuristics are **not a sound forward-error bound**: an
undetected FP64 rounding difference can still occur. The `1e8`
threshold is a policy constant, not a proven accuracy tolerance.

A canonical `FULL` dense evaluation keeps its prior hot loop and marks
its cache ineligible for immediate incremental reuse. A mode switch
requires a new evaluation under the incremental policy. A private
state copy carries the eligibility flag.

## Verified evidence

- Before this change, the new hidden-roundoff case failed:
  `incremental cancellation actual=1 expected=0`.
- After this change, the native GCC `-O2 -Wall -Wextra -Werror`
  semantic-safety binary reports `PASS cases=11 failed=0`.
- An isolated x86-64 microbenchmark of 1200 FULL evaluations of a
  512x512 dense layer, GCC `-O3`, reported:
  old hot loop 275067/276098 ns, new 276817/275564 ns.
  This is **not** an Android performance result or a statistically
  controlled speed comparison; do not generalize.
- Prior 94 legacy Python tests, 2 compiled .nir tests, ASan/UBSan and
  2 stress suites passed on the **previous** version of this branch.
  They have **not yet been rerun** after the new cache eligibility guard.

## Unresolved acceptance gate

A genuine *bounded-error* incremental mode requires a validated
propagated FP64 error bound (including activation and DAG fan-out),
a declared absolute/relative tolerance, and a fallback to FULL when
that bound cannot be met. Alternatively, a *strict* semantic contract
must use canonical FULL for any case not certified equivalent.

Do **not** promote this heuristic to a general equivalence proof, merge
to `main`, or claim Android speedup on its basis.
