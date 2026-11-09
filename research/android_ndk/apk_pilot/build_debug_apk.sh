#!/usr/bin/env bash
# Offline diagnostic APK builder. Accepts optional existing debug signing keystore.
# Every invocation uses a fresh run directory; sources are never deleted.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SRC="$ROOT/research/android_ndk/apk_pilot"
: "$ANDROID_NDK_HOME" "$ANDROID_BUILD_TOOLS" "$ANDROID_PLATFORM_JAR"
NDK_BIN="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin"
MODEL="$ROOT/research/agent_pilot/models/cartpole_pretrained.nlw"
mkdir -p "$ROOT/runs"
WORK="$(mktemp -d "$ROOT/runs/android_pilot_XXXXXXXX")"
mkdir -p "$WORK/generated" "$WORK/classes" "$WORK/dex"
for n in pretrained_cartpole integrated inference; do
  python3 "$ROOT/language/compiler.py" "$ROOT/language/examples/$n.nir" \
    --emit-only -o "$WORK/generated/$n"
done
"$NDK_BIN/aarch64-linux-android26-clang++" \
  -std=c++17 -O2 -fPIC -shared -static-libstdc++ \
  -Wall -Wextra -Werror -I "$WORK" -I "$ROOT/language/runtime" \
  "$SRC/NativePilot.cpp" -o "$WORK/libneural_pilot.so"
"$NDK_BIN/llvm-strip" --strip-unneeded "$WORK/libneural_pilot.so"
javac -source 8 -target 8 -cp "$ANDROID_PLATFORM_JAR" \
  -d "$WORK/classes" "$SRC/src/dev/neurallanguage/pilot/MainActivity.java"
python3 - "$WORK" "$ANDROID_BUILD_TOOLS" "$ANDROID_PLATFORM_JAR" <<'PY'
import pathlib, subprocess, sys
w=pathlib.Path(sys.argv[1]);bt=pathlib.Path(sys.argv[2]);jar=sys.argv[3]
classes=sorted(str(p) for p in (w/'classes').rglob('*.class'))
subprocess.run([str(bt/'d8'),'--min-api','26','--lib',jar,
                '--output',str(w/'dex'),*classes],check=True)
PY
"$ANDROID_BUILD_TOOLS/aapt2" link --manifest "$SRC/AndroidManifest.xml" \
   -I "$ANDROID_PLATFORM_JAR" --min-sdk-version 26 \
   --target-sdk-version 34 -o "$WORK/base_unsigned.apk"
python3 - "$WORK" "$MODEL" <<'PY'
import pathlib,sys,zipfile
w=pathlib.Path(sys.argv[1]);m=pathlib.Path(sys.argv[2])
with zipfile.ZipFile(w/'base_unsigned.apk','a',strict_timestamps=False) as z:
    z.write(w/'dex/classes.dex','classes.dex')
    z.write(w/'libneural_pilot.so','lib/arm64-v8a/libneural_pilot.so')
    z.write(m,'assets/cartpole_pretrained.nlw')
PY
"$ANDROID_BUILD_TOOLS/zipalign" -f -p 4 \
   "$WORK/base_unsigned.apk" "$WORK/aligned_unsigned.apk"
if [[ "$#" -gt 0 ]]; then
  KEYSTORE="$1"
  test -f "$KEYSTORE" || { echo "Missing supplied keystore" >&2; exit 2; }
else
  KEYSTORE="$WORK/research-debug.keystore"
  keytool -genkeypair -noprompt -keystore "$KEYSTORE" -alias neuralpilot \
    -storepass android -keypass android -keyalg RSA -keysize 3072 \
    -validity 3650 -dname "CN=Neural Language Pilot Research,OU=Research,O=Neural Language,L=Local,C=US"
  echo "Keep the temporary debug keystore for update-signing: $KEYSTORE"
fi
APK="$WORK/NeuralLanguagePilot-v0.1.0-debug.apk"
"$ANDROID_BUILD_TOOLS/apksigner" sign --ks "$KEYSTORE" \
  --ks-key-alias neuralpilot --ks-pass pass:android --key-pass pass:android \
  --out "$APK" "$WORK/aligned_unsigned.apk"
"$ANDROID_BUILD_TOOLS/apksigner" verify --verbose --print-certs "$APK"
"$ANDROID_BUILD_TOOLS/aapt2" dump badging "$APK" | \
   grep -E '^package:|^sdkVersion|^targetSdkVersion|^native-code'
sha256sum "$APK"
echo "APK_BUILD_PASS $APK"
echo "NOT_DEVICE_TESTED"
