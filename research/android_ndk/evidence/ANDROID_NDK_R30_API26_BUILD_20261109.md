# Neural Language — Android NDK r30 / arm64-v8a API 26 build, 2026-11-09

**Verdict:** BUILD PASS; DEVICE RUNTIME NOT TESTED. Separate research branch only; not an APK and not a claim of mobile performance.

## Provenance and toolchain

- Research parent: `research/pretrained-cartpole-agent-20261109` HEAD `276a066cfc38fd9de3426ccfc02eeb89f1beaa71`.
- Development branch: `research/android-ndk-api26-20261109`; unchanged `main` remains at `1168f3fcc2ecd0bd17de08be69d313844790c845`.
- Fetched nine original private-repository files (including compiler, full native runtime, .nir examples and model); **9/9 Git object SHA matches after transfer**.
- `compiler.py` Git blob `65f4ef45ad4c144e3d6e07535c3dbce78b8adb6b`; `neural_runtime.hpp` Git blob `faf78c47c60cf325152161e258978177be2e5cf9`.
- `pretrained_cartpole.nir` Git blob `57f9d55289f227c4619da1a7f11498ea81eb47af`.
- `cartpole_pretrained.nlw` Git blob `b70bcf774e1f6dbf32c22a366bc0c81350f551ed`; SHA-256 `a0738c97790d61dbd7b0fe2bde1ece59ff5041457ac4c77c8aae99c0be3cf8da`.
- Official Google Android NDK r30 Linux ZIP downloaded from `https://dl.google.com/android/repository/android-ndk-r30-linux.zip`; verified SHA-1 `5107f898313790e449e87eee2183d9a20602dee9` against [android/ndk release list](https://github.com/android/ndk/wiki).
- `Android clang 21.0.0` (r30), target `aarch64-unknown-linux-android26`, `-std=c++17 -O2 -Wall -Wextra -Werror`.
- Build executed in isolated Ubuntu x86-64 sandbox. No Android phone, emulator with Android userspace, or ADB-connected device was available.

## Actually executed checks (new)

1. Direct `.nir → Neural IR/JSON → C++17 → NDK clang++ → Android ELF` compile of all three original .nir examples with the original full `neural_runtime.hpp`: **3/3 BUILD PASS**.
2. Every resulting ELF was checked with NDK `llvm-readelf`: **ELF64, AArch64, /system/bin/linker64**. Initial shared-C++ linkage used `libc++_shared.so`.
3. Confirmed `-static-libstdc++` works with original C++ runtime; verified no `libc++_shared.so` DT_NEEDED after static STL linkage.
4. Added `research/android_ndk/build_android_api26.py` as a repeatable, source-preserving and offline packaging/build gate. Ran `python3 -m py_compile` and `sh -n research/android_ndk/run_on_device.sh`: **PASS**.
5. Executed `python3 research/android_ndk/build_android_api26.py` with `ANDROID_NDK_HOME` set. **BUILD_PASS_NOT_DEVICE_TESTED**. This recompiles and strips all three Android ELF executables, verifies ELF architecture and dependencies and builds a ZIP containing the binaries, original .nlw model, a non-destructive device-side test script, README, checksums and manifest. ZIP CRC verified by the script.

### Built-and-checked ELF SHA-256 (after static STL and strip)

| Program | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `pretrained_cartpole` | 695208 | `e5a61662c74d442208edbdd1a48e636cba31acbf50ff4666eeeaf3fa83659144` |
| `integrated` | 815408 | `b38b045ad8b197468cbfc9ac8bf6db84f5378c2fd3d8538323dc9b49b8061a3e` |
| `inference` | 705128 | `50512d03829508d427816202c94e2127473ed5b1f3a1b91d9cce7b11d5cc10c8` |

Final generated ZIP was **782081 bytes**; SHA-256 `110152fd0a7f7ebba9b8e7a5bd582c849e57920760cc05680f4195240eccc14c`. **Important: the ephemeral build environment expired before this ZIP was published/downloaded as a durable user artifact. The ZIP is not attached here and must not be advertised as a downloadable file.** It can be regenerated from the retained source script and NDK r30; the repository contains the build instructions and test criteria, not the compiled executable.

## Reproduction

From a complete checkout of this branch on Linux with official Android NDK r30 installed:

```bash
export ANDROID_NDK_HOME=/path/to/android-ndk-r30
python3 -m py_compile research/android_ndk/build_android_api26.py
sh -n research/android_ndk/run_on_device.sh
python3 research/android_ndk/build_android_api26.py
```

Build output goes to a **new, unique** `runs/android_ndk_*` directory and paired ZIP; sources and previous runs are never deleted or overwritten. The archive is intended for optional `adb shell` inspection on a real Android arm64-v8a device, following `research/android_ndk/README_ANDROID.md`. Device-side script creates a new run folder per execution.

## Boundaries and next decision

- This is an Android/Bionic arm64 native build, **not a Linux ARM64 binary**.
- No device runtime was executed. Do not claim `ANDROID_DEVICE_PASS`, APK delivery, Android ADB connection, performance, RSS, energy or temperature.
- Previous Linux ARM64/QEMU functionality evidence remains in `research/agent_pilot/evidence/NIR_CROSS_ABI_20261109.md` and cannot be re-labelled Android validation.
- The existing separate `feature/mobile-inference-runtime-20261108` is not merged: its lightweight runtime currently restricts activation types to 0/1/2 whereas this pretrained CartPole model uses 3 (tanh). This separate compatibility gap requires independent changes/tests, not an architectural merge.
- Full Neural Language product goal is unchanged: semantic language/IR/compiler/native execution, useful AI workloads and eventual fair optimized comparisons on a physical phone.
- No deletions, force-push, or GitHub Actions.
