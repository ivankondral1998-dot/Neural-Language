#!/usr/bin/env python3
"""Build the HAR Android/ARM64 runner and offline verification bundle."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "android_ndk"))
from build_android_api26 import COMPILER, RUNTIME, TARGET, call, digest


def build(ndk, data, native, output, qemu=None):
    package = output / "package"
    package.mkdir(parents=True, exist_ok=False)
    tools = ndk / "toolchains/llvm/prebuilt/linux-x86_64/bin"
    binary = package / "har_infer"
    stem = output / "har_infer"
    commands = [
        [sys.executable, COMPILER, HERE / "infer.nir", "--emit-only", "-o", stem],
        [tools / "clang++", f"--target={TARGET}", "-std=c++17", "-O2",
         "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", "-static-libstdc++",
         "-I", RUNTIME, stem.with_suffix(".cpp"), "-o", binary],
        [tools / "llvm-strip", "--strip-unneeded", binary],
    ]
    (output / "build.log").write_text("".join(call(command) for command in commands))
    elf = call([tools / "llvm-readelf", "-h", "-l", "-d", binary])
    if not all(marker in elf for marker in ("AArch64", "/system/bin/linker64", "libc.so")):
        raise ValueError("unexpected Android ELF")
    if "libc++_shared.so" in elf:
        raise ValueError("unexpected shared C++ library")
    (package / "elf.txt").write_text(elf)

    for src, name in ((data/"har_test.txt", "har_test.txt"), (native/"har_model.nlw", "har_model.nlw"),
                      (HERE/"infer.nir", "infer.nir"), (HERE/"run_arm64.sh", "run.sh"),
                      (HERE/"verify.awk", "verify.awk"), (data/"data_test.json", "data_test.json")):
        shutil.copyfile(src, package / name)
    data_manifest = json.loads((package / "data_test.json").read_text())
    if digest(package/"har_test.txt") != data_manifest["output"]["har_test.txt"]:
        raise ValueError("test feature checksum mismatch")
    logits = (native/"expected_logits.txt").read_text().splitlines()
    labels = (data/"har_labels.txt").read_text().splitlines()
    if len(logits) != 2947 or len(labels) != 2947:
        raise ValueError("reference output must contain 2947 rows")
    (package/"expected.txt").write_text("".join(f"{row} {label}\n" for row,label in zip(logits, labels)))

    manifest = {"device_status":"DEVICE_NOT_TESTED", "architecture":"arm64-v8a", "target":TARGET,
                "ndk_clang":call([tools/"clang++", "--version"]).splitlines()[0],
                "compiler_sha256":digest(COMPILER),
                "generated_cpp_sha256":digest(stem.with_suffix(".cpp")),
                "ir_sha256":digest(stem.with_suffix(".nir.json")),
                "binary_sha256":digest(binary), "model_sha256":digest(package/"har_model.nlw"),
                "data_sha256":digest(package/"har_test.txt"), "absolute_logit_tolerance":2e-8}

    if qemu:
        # A separate static Bionic link is runnable under user-mode QEMU on Linux.
        # ARM64 Bionic startup requires 64-byte alignment of the ELF TLS segment.
        tls = output/"qemu_tls.cpp"
        tls.write_text("alignas(64) thread_local unsigned char nl_bionic_tls_padding "
                       "__attribute__((used, retain)) = 1;\n")
        static = output/"har_infer_qemu"
        call([tools/"clang++", f"--target={TARGET}", "-std=c++17", "-O2", "-ffp-contract=off",
              "-static", "-fno-emulated-tls", "-Wl,-u,nl_bionic_tls_padding",
              "-I", RUNTIME, stem.with_suffix(".cpp"), tls, "-o", static])
        run = subprocess.run([qemu, static], cwd=package, check=True, text=True,
                             capture_output=True, timeout=120)
        log = output/"qemu_logits.log"
        log.write_text(run.stdout)
        verdict = call(["awk", "-f", package/"verify.awk", package/"expected.txt", log])
        (package/"qemu_check.log").write_text(verdict)
        manifest["qemu_check"] = {"output":verdict.strip(), "static_elf_sha256":digest(static),
                                  "scope":"same generated source; separate static Bionic link"}

    (package/"manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (package/"README.txt").write_text(
        "NL-TRAIN-001, UCI HAR FP64. DEVICE_NOT_TESTED.\n"
        "In Termux extract into a fresh directory under HOME, cd there, run: sh run.sh\n"
        "The runner reads precomputed 561-feature vectors, without sensor capture or labels.\n"
        "HAR_PASS validates all logits/classes and prints accuracy.\n"
        "UCI HAR by Reyes-Ortiz, Anguita, Ghio, Oneto, Parra (2013), DOI 10.24432/C54S4K,\n"
        "CC BY 4.0 https://creativecommons.org/licenses/by/4.0/.\n"
        "Test features preserve the published values with whitespace normalized.\n"
    )
    files = sorted(package.iterdir())
    (package/"SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in files))
    archive = output/"NL_TRAIN_001_HAR_A52.tar.gz"
    with tarfile.open(archive, "x:gz", compresslevel=6) as tf:
        for path in sorted(package.iterdir()):
            tf.add(path, arcname=path.name)
    print(json.dumps({**manifest, "archive_sha256":digest(archive), "archive_bytes":archive.stat().st_size},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ndk", "data", "native", "output"):
        parser.add_argument(name, type=lambda p: Path(p).resolve())
    parser.add_argument("--qemu", type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    build(args.ndk, args.data, args.native, args.output, args.qemu)
