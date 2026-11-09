# Neural Language — Android ARM64 / API 26+ offline test bundle

**Status: Android NDK build only. NOT yet tested on a physical device.**
This ZIP is not an APK and is not installable by tapping it. It contains
three Android/Bionic ELF64 executables compiled from real .nir programs.
There is no cloud API, training download, GMS, root or network requirement.

## Contents

- pretrained_cartpole — load the already trained CartPole policy, predict.
- cartpole_pretrained.nlw — trusted pretrained NLMODEL2 4→64→64→2 weights.
- integrated — train a tiny .nir linear network in 1600 examples,
  validate, and save trained.nlw.
- inference — load that saved trained.nlw, independently predict.
- run_on_device.sh — create a unique log directory and run all three checks.
- SHA256SUMS, build_manifest.json, this README.

Compiled with official Android NDK r30 Clang 21, target aarch64-linux-android26.
C++ standard library is linked statically; Android libc/libm/libdl are system
libraries. This is an Android ELF executable, not a Linux/glibc program.

## Physical test (requires ADB connected by USB or wireless debugging)

Unpack the ZIP on a computer, then from the unpacked directory:

    adb shell 'mkdir -p /data/local/tmp/nl_android_20261109'
    adb push pretrained_cartpole integrated inference cartpole_pretrained.nlw run_on_device.sh /data/local/tmp/nl_android_20261109/
    adb shell 'cd /data/local/tmp/nl_android_20261109 && chmod 700 run_on_device.sh && ./run_on_device.sh'

On success, the device prints:

    CARTPOLE_DEVICE_PASS
    TRAIN_DEVICE_PASS
    INFERENCE_DEVICE_PASS
    ANDROID_DEVICE_3_PROGRAMS_PASS

Each execution creates a fresh run_... folder; logs and checkpoints remain
for inspection. Existing folders, models and results are never deleted.

If the device has no functional USB connection, Android's Wireless debugging
offers an ADB transport without a cable. This is an ADB-based test tool,
not an APK. The terminal on the phone does not automatically have permission
to run unpacked files from shared storage (/sdcard is not an executable
directory).

## Scope

- The model comes from the isolated research/pretrained-cartpole-agent-20261109
  work and has 4610 parameters, of which 4480 are weights and 130 biases.
- This NDK test uses the full Neural Language runtime, not the separate
  mobile inference branch. That branch's thin sequential inference runtime
  currently cannot process tanh activation (type=3) required by CartPole.
- Even when the physical test passes, it proves functional compatibility for
  this scenario, not a speed, RAM, battery or TFLite/ONNX advantage.
- No Android tests have been executed by the builder as there was no device.
- Do not use run_on_device.sh as a benchmark: stdout and file I/O dominate.
