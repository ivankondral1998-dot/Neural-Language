#!/usr/bin/env python3
"""Build an Android ARM64 Termux bundle. Does not run it and does not print DEVICE PASS."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "nl_mobile_opt_001"
NDK = Path(os.environ["ANDROID_NDK"])
DATA = Path(os.environ["NL_DATA_DIR"])
OUT = Path(os.environ.get("NL_ANDROID_OUT", "/tmp/nl-android-opt002-out"))
CXX = NDK / "toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang++"
TARGET_FLAGS = [
    f"--target=aarch64-linux-android26", "-std=c++17", "-O2", "-ffp-contract=off",
    "-Wall", "-Wextra", "-Werror", "-static-libstdc++",
]


def run(command):
    print(" ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def digest(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def emit(nir, output, profile):
    command = [sys.executable, str(ROOT / "language/compiler.py"), str(nir), "--emit-only", "-o", str(output)]
    if profile:
        command.extend(["--profile", profile])
    run(command)


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to replace {OUT}")
    package = OUT / "package"
    package.mkdir(parents=True)
    runtime = ROOT / "language/runtime"
    jobs = [
        (SOURCE / "har_infer.nir", "har_generic", None, []),
        (SOURCE / "har_infer.nir", "har_mobile", "mobile-cpu", ["-DNEURAL_MOBILE_PLAN"]),
        (SOURCE / "cartpole_infer.nir", "cartpole_generic", None, []),
        (SOURCE / "cartpole_infer.nir", "cartpole_mobile", "mobile-cpu", ["-DNEURAL_MOBILE_PLAN"]),
    ]
    for nir, name, profile, defines in jobs:
        stem = OUT / name
        emit(nir, stem, profile)
        run([CXX, *TARGET_FLAGS, *defines, "-I", runtime, stem.with_suffix(".cpp"), "-o", package / name])
    run([CXX, *TARGET_FLAGS, "-DNEURAL_MOBILE_PLAN", "-I", runtime, SOURCE / "bench_host.cpp", "-ldl", "-o", package / "bench_arm64"])
    for src, name in (
        (ROOT / "research/nl_train_001/models/har_model.nlw", "har_model.nlw"),
        (ROOT / "research/agent_pilot/models/cartpole_pretrained.nlw", "cartpole_pretrained.nlw"),
        (DATA / "har_test.txt", "har_test.txt"),
        (DATA / "har_labels.txt", "har_labels.txt"),
        (SOURCE / "har_infer.nir", "har_infer.nir"),
        (SOURCE / "cartpole_infer.nir", "cartpole_infer.nir"),
        (SOURCE / "protocol.txt", "protocol.txt"),
    ):
        shutil.copyfile(src, package / name)
    (package / "run.sh").write_text("""#!/bin/sh
set -eu
cd "$(dirname "$0")"
echo "DEVICE_MODEL=$(getprop ro.product.model 2>/dev/null || echo unknown)"
sha256sum -c SHA256SUMS
./har_generic > generic.log
./har_mobile > mobile.log
awk 'BEGIN{g=0;m=0}
  FNR==NR && $1=="DATA_PREDICT" {g++; a[g]=$0; next}
  $1=="DATA_PREDICT" {m++; if ($0 != a[m]) {print "LOGIT_MISMATCH", m; exit 1}}
  END {if (g!=2947 || m!=2947) {print "ROW_MISMATCH", g, m; exit 1} print "HAR_LOGITS_IDENTICAL", m}' generic.log mobile.log
./cartpole_generic > cartpole_generic.log
./cartpole_mobile > cartpole_mobile.log
awk 'BEGIN{g=0;m=0}
  FNR==NR && $1=="PREDICT" {g++; a[g]=$0; next}
  $1=="PREDICT" {m++; if ($0 != a[m]) {print "CARTPOLE_MISMATCH"; exit 1}}
  END {if (g!=1 || m!=1) exit 1; print "CARTPOLE_PREDICT_IDENTICAL"}' cartpole_generic.log cartpole_mobile.log
echo '--- bench har ---'
./bench_arm64 --task har | tee bench_har.log
echo '--- bench cartpole ---'
./bench_arm64 --task cartpole | tee bench_cartpole.log
echo MOBILE_OPT_002_BENCH_COMPLETE
""", encoding="utf-8")
    (package / "README.txt").write_text(
        "NL-MOBILE-OPT-002 Android ARM64 API 26 bundle.\n"
        "DEVICE_NOT_TESTED at build time. Physical runs must report device and logs.\n"
        "Unpack under Termux $HOME, then: sh run.sh\n"
        "har_generic is the previous FULL path. har_mobile is --profile mobile-cpu with FULL-mode input reuse.\n"
        "FP64 row order is unchanged. NEON was not enabled.\n"
        "UCI HAR features are the published test split with whitespace normalized.\n",
        encoding="utf-8")
    files = sorted(p for p in package.iterdir() if p.name != "SHA256SUMS")
    (package / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in files), encoding="utf-8")
    archive = OUT / "NL_MOBILE_OPT_002_A52_TERMUX.tar.gz"
    with tarfile.open(archive, "x:gz") as tf:
        for path in sorted(package.iterdir()):
            tf.add(path, arcname=path.name)
    print(f"archive {archive} bytes {archive.stat().st_size} sha256 {digest(archive)}")


if __name__ == "__main__":
    main()
