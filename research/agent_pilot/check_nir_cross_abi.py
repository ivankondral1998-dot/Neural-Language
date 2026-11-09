#!/usr/bin/env python3
"""Cross-ABI .nir compiler/runtime parity gate (Linux AMD64 and QEMU ARM64).

Read-only with respect to source and trained weights. Writes a unique evidence
directory under runs/. This is NOT an Android or real-device performance test.
Requires Python3, g++, aarch64-linux-gnu-g++, qemu-aarch64 and ARM64 sysroot.
No new Python dependencies, internet, or GitHub Actions.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
COMPILER = ROOT / "language" / "compiler.py"
EXAMPLES = ROOT / "language" / "examples"
MODEL = ROOT / "research" / "agent_pilot" / "models" / "cartpole_pretrained.nlw"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_cartpole_reference import parse_model, predict


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_program(program: str, output: Path, compiler: str) -> None:
    command = [
        sys.executable, str(COMPILER), str(EXAMPLES / program),
        "--compiler", compiler, "-o", str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=90)


def run_program(binary: Path, cwd: Path, arm64: bool) -> str:
    command = [str(binary)]
    if arm64:
        command = [
            "qemu-aarch64", "-L", "/usr/aarch64-linux-gnu", str(binary)
        ]
    completed = subprocess.run(
        command, cwd=cwd, check=True, capture_output=True,
        text=True, timeout=120
    )
    return completed.stdout


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_arch(root: Path, arch: str, cxx: str) -> dict:
    directory = root / arch
    directory.mkdir()
    arm64 = arch == "arm64_qemu"
    evidence = {}
    for program, output in (
        ("pretrained_cartpole.nir", "cartpole"),
        ("integrated.nir", "trainer"),
        ("inference.nir", "inference"),
    ):
        check_program(program, directory / output, cxx)
        evidence[output + "_cpp"] = sha(directory / (output + ".cpp"))
        evidence[output + "_ir"] = sha(directory / (output + ".nir.json"))

    shutil.copyfile(MODEL, directory / "cartpole_pretrained.nlw")
    cartpole_output = run_program(directory / "cartpole", directory, arm64)
    (directory / "cartpole.stdout.log").write_text(cartpole_output)
    require("MODEL_LOADED cartpole_pretrained.nlw" in cartpole_output, "CartPole weights not loaded")
    m = re.search(r"^PREDICT version=1 values=([^\r\n]+)", cartpole_output, re.M)
    require(m is not None, "CartPole prediction missing")
    observed = tuple(float(x) for x in m.group(1).split(","))
    layers = parse_model(MODEL.read_text(encoding="utf-8"))
    expected = predict(layers, (0.02, -0.01, 0.04, -0.03))
    require(len(observed) == len(expected) == 2, "CartPole output shape")
    for got, want in zip(observed, expected):
        require(math.isclose(got, want, abs_tol=1e-9, rel_tol=1e-9),
                f"CartPole oracle mismatch: {got} vs {want}")
    evidence["cartpole_logits"] = list(observed)

    train_output = run_program(directory / "trainer", directory, arm64)
    (directory / "trainer.stdout.log").write_text(train_output)
    require(train_output.count("VALIDATE PASS") == 4, "trainer validation")
    require("VERIFY PASS" in train_output and "OPTIMIZE selected=" in train_output,
            "trainer verify/optimize")
    require("MODEL_SAVED trained.nlw" in train_output, "trainer checkpoint missing")
    saved_model = directory / "trained.nlw"
    require(saved_model.is_file(), "trainer did not produce checkpoint")
    evidence["trained_model_sha256"] = sha(saved_model)

    infer_output = run_program(directory / "inference", directory, arm64)
    (directory / "inference.stdout.log").write_text(infer_output)
    require("MODEL_LOADED trained.nlw" in infer_output, "inference load missing")
    require(infer_output.count("VALIDATE PASS") == 2, "inference validation")
    require("PREDICT version=1 values=0.3" in infer_output, "inference prediction")
    t = json.loads((directory / "integrated_result.json").read_text())
    i = json.loads((directory / "inference_result.json").read_text())
    require(t["counts"]["train"] == 1600, "1600 training steps")
    require(i["counts"]["train"] == 0, "inference must not train")
    evidence["train_steps"] = t["counts"]["train"]
    evidence["inference_steps"] = i["counts"]["train"]
    return evidence


def main() -> int:
    for command in ("g++", "aarch64-linux-gnu-g++", "qemu-aarch64"):
        if shutil.which(command) is None:
            raise RuntimeError(f"missing required tool: {command}")
    require(MODEL.is_file(), "missing pretrained model")
    dest = ROOT / "runs"
    dest.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="crossabi_", dir=dest))
    evidence = {
        "status": "PENDING",
        "scope": "Linux x86_64 native vs AArch64 Linux under QEMU user mode, NOT Android",
        "model_sha256": sha(MODEL),
        "evidence_dir": str(root),
    }
    try:
        host = run_arch(root, "x86_64", "g++")
        arm = run_arch(root, "arm64_qemu", "aarch64-linux-gnu-g++")
        for item in ("cartpole_cpp", "cartpole_ir",
                     "trainer_cpp", "trainer_ir",
                     "inference_cpp", "inference_ir",
                     "trained_model_sha256"):
            require(host[item] == arm[item], f"ABI parity mismatch for {item}")
        for a, b in zip(host["cartpole_logits"], arm["cartpole_logits"]):
            require(a == b, "CartPole prediction mismatch across ABIs")
        evidence.update({
            "status": "PASS", "x86_64": host, "arm64_qemu": arm,
            "source_ir_cpp_equal": True, "checkpoint_bytes_equal": True,
        })
    except Exception as exc:
        evidence["status"] = "FAIL"
        evidence["error"] = str(exc)
        raise
    finally:
        (root / "validation_result.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
