# Integration matrix — 2026-11-10

Base is \`opt/nl-mobile-opt-002-reuse-input-20261110\` @ \`5cf5e5b9c96455903c8d898796389a965fd1cf66\`. Existing main untouched. Snapshot includes earlier HAR training, DAG, input changes, effects/shape work and mobile work. This is not a verified integration of all 27 branches.

| Branch | SHA | Status |
|---|---|---|
| main | 1168f3fcc2ecd0bd17de08be69d313844790c845 | base ancestry/reference, unchanged |
| feature/nl-train-001-har-20261110 | 7784ac2b143d132e8cec467f8ed8b4965a2b8e19 | HAR files/evidence retained in base |
| opt/nl-mobile-opt-002-reuse-input-20261110 | 5cf5e5b9c96455903c8d898796389a965fd1cf66 | selected base |
| feature/nl-train-002-regression-20261110 | b95491196debbf5504cb0e19fb396f2a814a28f2 | NOT merged; 5 shared files changed plus 6 new |
| opt/streaming-dataset-inference-20261110 | 457f0df9c1c5b074d87bba9bc4fd38c3f705dd0d | NOT merged; 4 shared files changed plus 2 new |
| fix/runtime-correctness-audit-20261110 | 46e28a5dea4970fa9908519a7db0bdea1d1ca030 | NOT merged; 4 shared files changed plus 5 new |
| fix/canonical-delta-contract-20261110 | 9399314824b7d2a535fabd175c805a9f2c245d21 | divergent semantics; NOT independently revalidated |
| fix/semantic-safety-20261110 | 316fb49015bf22967344994fd14c3d1f64a74af3 | divergent semantics; NOT independently revalidated |

Reconcile each of the remaining branches with a code-level merge and compiler/runtime regression run before claiming the language is unified. Current experiment only adds isolated research files, without changing native Neural Language semantics.
