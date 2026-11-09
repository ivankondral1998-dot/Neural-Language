#!/usr/bin/env python3
"""Build real .nir programs for Android ARM64 using an official Android NDK.

Output: fresh directory under runs/, a ZIP, checked ELF metadata and hashes.
This is a BUILD gate. It does not run binaries on Android or claim speedups.
No downloads, Gradle, external Python packages or changes to input sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "language" / "examples"
COMPILER = ROOT / "language" / "compiler.py"
RUNTIME = ROOT / "language" / "runtime"
MODEL = ROOT / "research" / "agent_pilot" / "models" / "cartpole_pretrained.nlw"
HERE = Path(__file__).resolve().parent
PROGRAMS = (
    ("pretrained_cartpole.nir", "pretrained_cartpole"),
    ("integrated.nir", "integrated"),
    ("inference.nir", "inference"),
)
TARGET = "aarch64-linux-android26"


def call(command, *, timeout=120):
    proc = subprocess.run(
        [str(item) for item in command], check=True, text=True,
        capture_output=True, timeout=timeout
    )
    return proc.stdout


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(ndk: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    clang_dir = ndk / "toolchains" / "llvm" / "prebuilt" / "linux-x86_64" / "bin"
    clang, strip, readelf = (clang_dir / n for n in
                            ("clang++", "llvm-strip", "llvm-readelf"))
    for path in (clang, strip, readelf, COMPILER, MODEL,
                 HERE / "run_on_device.sh", HERE / "README_ANDROID.md"):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    output.mkdir(parents=True)
    binaries = {}
    build_output = []
    for source, name in PROGRAMS:
        destination = output / name
        build_output.append(call([
            sys.executable, COMPILER, SOURCES / source,
            "--emit-only", "-o", destination
        ]))
        cpp = destination.with_suffix(".cpp")
        build_output.append(call([
            clang, f"--target={TARGET}", "-std=c++17", "-O2",
            "-Wall", "-Wextra", "-Werror", "-static-libstdc++",
            "-I", RUNTIME, cpp, "-o", destination
        ]))
        call([strip, "--strip-unneeded", destination])
        header = call([readelf, "-h", destination])
        program = call([readelf, "-l", destination])
        dynamic = call([readelf, "-d", destination])
        if ("AArch64" not in header or
                "/system/bin/linker64" not in program or
                "libc++_shared.so" in dynamic or
                "libc.so" not in dynamic):
            raise AssertionError("unexpected Android ELF layout for " + name)
        binaries[name] = {
            "sha256": digest(destination), "bytes": destination.stat().st_size,
            "cpp_sha256": digest(cpp), "ir_sha256": digest(destination.with_suffix(".nir.json")),
            "target": TARGET,
        }

    shutil.copyfile(MODEL, output / MODEL.name)
    shutil.copyfile(HERE / "run_on_device.sh", output / "run_on_device.sh")
    shutil.copyfile(HERE / "README_ANDROID.md", output / "README_ANDROID.md")
    (output / "run_on_device.sh").chmod(0o755)

    hashes = [f"{digest(output / name)}  {name}"
              for name in [*(n for _, n in PROGRAMS), MODEL.name]]
    (output / "SHA256SUMS").write_text("\n".join(hashes) + "\n", encoding="ascii")
    manifest = {
        "status": "BUILD_PASS_NOT_DEVICE_TESTED",
        "architecture": "arm64-v8a",
        "android_api_min": 26,
        "ndk_clang_version": call([clang, "--version"]).splitlines()[0],
        "source_compiler_sha256": digest(COMPILER),
        "source_runtime_sha256": digest(RUNTIME / "neural_runtime.hpp"),
        "source_model_sha256": digest(MODEL),
        "binaries": binaries,
        "runtime": "Android Bionic; C++ standard library statically linked",
        "device_test": "NOT RUN; requires an Android arm64-v8a device/ADB",
    }
    (output / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )
    (output / "compile.log").write_text("".join(build_output), encoding="utf-8")
    archive = output.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(f"archive already exists: {archive}")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as result:
        for file in sorted(output.iterdir()):
            if file.name.endswith(".cpp") or file.name.endswith(".nir.json"):
                continue  # full source/IR stays in evidence directory
            result.write(file, arcname=file.name)
    with zipfile.ZipFile(archive) as check:
        if check.testzip() is not None:
            raise AssertionError("ZIP CRC validation failed")
    manifest["zip_sha256"] = digest(archive)
    manifest["zip_bytes"] = archive.stat().st_size
    manifest["zip_path"] = str(archive)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ndk", type=Path, default=Path(os.environ.get(
        "ANDROID_NDK_HOME", "/nonexistent/ndk")))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    if args.output is None:
        path = Path(tempfile.mkdtemp(prefix="android_ndk_", dir=runs))
        path.rmdir()  # only the empty, freshly-created directory is removed
    else:
        path = args.output.resolve()
    run(args.ndk.resolve(), path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
