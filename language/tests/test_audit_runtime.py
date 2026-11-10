"""Regressions for GPT-6.1 Work audit counterexamples (2026-11-10)."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent


@unittest.skipUnless(shutil.which("g++"), "requires host g++")
class AuditRuntimeTests(unittest.TestCase):
    def test_cpp_cache_mixed_updates_and_atomic_sgd(self):
        with tempfile.TemporaryDirectory(prefix="nl_audit_cpp_") as tmp:
            executable = Path(tmp) / "audit_runtime"
            build = subprocess.run(
                ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                 "-I", str(ROOT / "runtime"),
                 str(TESTS / "audit_runtime_regressions.cpp"),
                 "-o", str(executable)],
                capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run(
                [str(executable)], capture_output=True, text=True, timeout=30
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("AUDIT_RUNTIME_REGRESSIONS_PASS", run.stdout)

    def test_nir_observe_delta_then_tune_full(self):
        with tempfile.TemporaryDirectory(prefix="nl_audit_nir_") as tmp:
            work = Path(tmp)
            shutil.copyfile(TESTS / "fixtures" / "cancellation.nlw",
                            work / "cancellation.nlw")
            run = subprocess.run(
                ["python3", str(ROOT / "compiler.py"),
                 str(TESTS / "fixtures" / "audit_cancellation.nir"),
                 "--compiler", "g++", "-o", str(work / "cancel"), "--run"],
                cwd=work, capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("PREDICT version=1 values=0", run.stdout)
            self.assertEqual(run.stdout.count("PREDICT version=2 values=5"), 2)
            self.assertIn("EXPECT PASS tolerance=1e-09", run.stdout)


if __name__ == "__main__":
    unittest.main()
