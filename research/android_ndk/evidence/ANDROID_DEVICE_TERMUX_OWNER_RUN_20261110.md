# Neural Language — first owner-reported Android device execution (2026-11-10)

**Evidence level:** USER-REPORTED PHYSICAL DEVICE PASS. This file records the owner's actual Termux console output as supplied in the project conversation, **not** a log independently pulled from the device. It upgrades the previous NDK BUILD-ONLY status for the **three covered programs**, without establishing performance superiority, correctness outside this test, or device identity.

## Source and provenance

- Source: owner-provided unedited excerpt of the Termux session, local time indicated in result directory: `20261110_012634`.
- Program source: `research/android_ndk/run_on_device.sh`, Git blob `fcb4d7c0ce72a9c32c41d3ea73c66a9e750145be`, read back in the repository on this evidence commit.
- Bundle source branch: `research/android-ndk-api26-20261109`, based on the ARM64 Android NDK r30 package.
- Retained binary: `research/android_ndk/artifacts/NEURAL_LANGUAGE_ANDROID_NDK_API26.tar.xz`, 320852 bytes, SHA-256 `99c3933f939bd7035e31a435322521f4a00dd7e98d01a5fd52e718f506a0ad79`. **The user's text does not independently confirm they computed this archive-level SHA**, but it does show successful checks of the included executable and model checksums.
- Runtime: Termux on an Android device, per user transcript. **Device model: Samsung SM-A525F; Android release: 14; reported primary ABI: arm64-v8a** (owner supplied three direct `getprop` results, recorded below). Exact SoC, Termux version, native timing and RSS remain unmeasured. The model and ABI were not independently read by this assistant.

## Relevant unedited console evidence from owner

```text
~/downloads $ tar -xJf ~/storage/downloads/NEURAL_LANGUAGE_ANDROID_NDK_API26.tar.xz
~/downloads $
~/downloads $ sha256sum -c SHA256SUMS
pretrained_cartpole: OK
integrated: OK
inference: OK
cartpole_pretrained.nlw: OK
~/downloads $
~/downloads $ chmod +x pretrained_cartpole integrated inference
~/downloads $
~/downloads $ sh run_on_device.sh
RUN_DIR=/data/data/com.termux/files/home/downloads/run_20261110_012634_13170
CARTPOLE_DEVICE_PASS
TRAIN_DEVICE_PASS
INFERENCE_DEVICE_PASS
ANDROID_DEVICE_3_PROGRAMS_PASS
LOG_DIR=/data/data/com.termux/files/home/downloads/run_20261110_012634_13170
~/downloads $
```

Earlier `pkg install tar xzmkdir -p ~/neural-language` and `cd ~/neural-language` failed because multiple commands were accidentally pasted together; this did not affect the successful test from `~/downloads`.

## Device details supplied in the next owner message

```text
~/downloads $ getprop ro.product.model; getprop ro.build.version.release; getprop ro.product.cpu.abi
SM-A525F
14
arm64-v8a
~/downloads $
```

This metadata establishes the reported test device identity for the current physical run; it does not measure execution time, max RSS, battery or thermal conditions.

## What the actual runner checks

The checked-in script uses `set -eu`, makes a new per-execution run directory, copies the 3 executables and the model there, then:

1. Runs `./pretrained_cartpole`, requires the exact `MODEL_LOADED cartpole_pretrained.nlw` and `PREDICT version=1 values=-0.3736812805,0.3745496992` markers, and requires a nonempty `cartpole_pilot_report.json`. Only then prints `CARTPOLE_DEVICE_PASS`.
2. Runs `./integrated`, requires `MODEL_SAVED trained.nlw`, `VERIFY PASS`, `PREDICT version=1600 values=0.3`, a nonempty `trained.nlw`, and `integrated_result.json`. Only then prints `TRAIN_DEVICE_PASS`.
3. Runs `./inference`, requires `MODEL_LOADED trained.nlw`, `PREDICT version=1 values=0.3`, and `inference_result.json`. Only then prints `INFERENCE_DEVICE_PASS`.
4. Only after these checks completes does it print `ANDROID_DEVICE_3_PROGRAMS_PASS`.

**Verdict for the exact scripted scenario:** user-reported functional Android native execution PASS. This is stronger evidence than ELF inspection, host cross-compilation or Linux QEMU; it is still not a controlled benchmark.

## Missing evidence / follow-up without repetition

- Device metadata: **SM-A525F / Android 14 / arm64-v8a supplied**; `uname -m`, SoC, Termux version, and direct archived stdout/stderr and JSON reports were **not supplied**. The existing run directory is the place to retrieve original logs; do **not** rerun training merely to collect this context.
- No direct timing, peak process RSS, battery/energy, repeatability or thermal results captured.
- No matched baseline against native C++, TFLite, ONNX Runtime or other appropriately optimized Android engines. **Do not claim Neural Language is faster or superior.**
- Shell/Termux successful execution does not equal an installable signed APK.
- Do not overwrite the previous build evidence. No GitHub Actions have been triggered for this record, and `main` remains untouched.
