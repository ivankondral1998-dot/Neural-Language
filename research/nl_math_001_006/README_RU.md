# Neural Language — mathematical sources, 2026-11-11

**Read-only source preservation for independent audit.** This directory is not an approved mathematical specification and does not alter the compiler, runtime, or product charter.

## Files

- `NL_MATH_006_VERIFIED_AUDIT_20261111.zip` — **complete nested mathematical research lineage NL-MATH-001–006**. The 006 ZIP contains 005; 005 contains 003 and 004; 003 contains original 001 and 002. Includes the original handoff, baseline oracle, new independent numerical oracle, tests, audits and manifests.
  - SHA-256: `65d1b538628ec293043ff153a4a717ea94399d7550d79802732dd4e0cd891fe8`
  - Size: 160938 bytes.
- `NEURAL_LANGUAGE_PROJECT_TRANSFER_20261111.zip` — context and owner decisions from a separate conversation, including the **later agreed architecture of independent specialized neural helpers with their own smaller weights**, superseding prior shared-core/100 LoRA-adapter schemes **for the helper architecture only, not the language**.
  - SHA-256: `bf2a5653ac8c77e933dd54a19e01b625d96d2889308eed5e6e0003c773e0d32b`
  - Size: 92586 bytes.

## Independent audit instructions

Read all inner archives before evaluating the mathematics; do not rely on summary claims or earlier PASS. Resolve conflicts by explicit later owner decisions, not chronological order of filenames alone. Compare to current implementation under the branch used for the audit, documenting exact commit and ref.

**Do not modify sources, main, architecture, or GitHub Actions. Do not run training or mobile experiments without owner instruction.**

The code baseline at preservation time: `research/mobile-text-kernels-20261110` @ `106c1c8b5244a84d35ce03da2b4c767a1093705f`. The current implementation has only partial compatibility with the mathematical definitions. **Mathematical readiness has NOT been approved**.

## Extract

`unzip NL_MATH_006_VERIFIED_AUDIT_20261111.zip`, then recursively unpack nested archives found under `sources/`. Validate `SHA256SUMS` at each level.
