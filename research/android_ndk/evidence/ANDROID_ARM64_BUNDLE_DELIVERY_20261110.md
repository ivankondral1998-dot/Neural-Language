# Neural Language — retained Android ARM64 execution bundle (2026-11-10)

**RESULT:** Android ARM64 build artifact is now durably committed on the isolated research branch. It is not an APK, not an Android runtime PASS, and not a speedup claim.

## Verifiable binary artifact

- Branch: `research/android-ndk-api26-20261109`.
- Artifact: `research/android_ndk/artifacts/NEURAL_LANGUAGE_ANDROID_NDK_API26.tar.xz`.
- Archive size: **320852 bytes**.
- Archive SHA-256: **`99c3933f939bd7035e31a435322521f4a00dd7e98d01a5fd52e718f506a0ad79`**.
- Git object SHA-1 (binary blob): **`3bf3f4740311b85d5c604a127a26e91bf244670c`**.
- Commit installing the binary into the branch: **`a09eaa6670085bfca1892bf924e85273ce397bf6`**.
- Git Trees API follow-up verified nested tree entry `mode=100644,type=blob,sha=3bf3f474...,size=320852` on that commit.
- The original remote sandbox archive was tested with `xz -t`, and `tar -tJf` enumerated all eight expected files. The Git object SHA matched the *same original tar.xz bytes* before binary commit.
- A separate NDK ZIP also compiled (782083 bytes, SHA-256 `70549dde6a52637e1333f433c43d22aeb0a765a9c37dfea4a462942d0506f6bc`) but **is NOT retained in GitHub**. Do not advertise that ZIP as available for download. The durable deliverable is the `.tar.xz`.
- The earlier ZIP (SHA-256 `110152...`) from the previous sandbox is not this artifact: independently generated archives contain timestamps and differ across runs.

## Contents

```
README_ANDROID.md
SHA256SUMS
build_manifest.json
cartpole_pretrained.nlw
inference
integrated
pretrained_cartpole
run_on_device.sh
```

All three programs were compiled from the repository's real `.nir` examples, using the full Neural Language runtime and the official Android NDK r30 `aarch64-linux-android26-clang++`. The script automatically checks ELF type, AArch64 architecture, `/system/bin/linker64`, and absence of dependency on `libc++_shared.so`. The initial compiler toolchain download passed the official release SHA-1 check. Internal model SHA-256 remained `a0738c97790d61dbd7b0fe2bde1ece59ff5041457ac4c77c8aae99c0be3cf8da`.

## Usage (not installable by tapping)

On a Linux machine, or using any archive manager that supports `.tar.xz`:

```bash
mkdir nl_android
tar -xJf NEURAL_LANGUAGE_ANDROID_NDK_API26.tar.xz -C nl_android
cd nl_android
sha256sum -c SHA256SUMS
adb shell 'mkdir -p /data/local/tmp/nl_android_20261110'
adb push pretrained_cartpole integrated inference cartpole_pretrained.nlw run_on_device.sh /data/local/tmp/nl_android_20261110/
adb shell 'cd /data/local/tmp/nl_android_20261110 && chmod 700 run_on_device.sh && ./run_on_device.sh'
```

ADB can use USB or Android's Wireless debugging; the script does not need root or GMS. If run, it creates its own fresh result directory and preserves prior runs.

**Device status:** not run on a physical Android device. The above ADB commands are a protocol, *not a performed test*. Shell syntax for `run_on_device.sh` passed `sh -n`. A host Linux smoke-test of that wrapper was attempted in the fresh sandbox but blocked because `g++` was not present; it is not counted PASS.

**GitHub Actions:** workflow remains manual-only on this research branch; no Actions triggered. `main` unchanged. No deletions of source, prior archives or intermediate transport parts.

**Product gate:** a reproducible Android cross-build does not establish Neural Language speed/RAM/energy superiority over C++, TFLite, ONNX Runtime or other mobile baselines. The next actual acceptance gate is a physical Android run and then fair same-hardware baseline measurement.
