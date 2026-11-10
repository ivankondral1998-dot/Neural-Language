"""Independent-audit counterexamples: compile real .nir and check runtime invariants."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

LANG = Path(__file__).resolve().parents[1]
CPP_REGRESSION = Path(__file__).with_name("runtime_semantic_safety.cpp")

CANCELLATION_NIR = """brain Cancellation {
  senses 2
  neurons hidden 1 relu
  neurons out 1 linear
  init 42
}
load_model "cancellation.nlw"
tune incremental
observe [-900000000,0.00000005]
predict
observe_delta 0 add 900000000
predict
tune full
predict
expect [5] tolerance 0.000000001
"""
SUBTLE_CANCELLATION_NIR = """brain SubtleCancellation {
  senses 3
  neurons hidden 1 linear
  neurons out 1 linear
  init 42
}
load_model "subtle.nlw"
tune incremental
observe [-900000000,0.00000005,900000000]
predict
observe_delta 1 add 0.1
predict
expect [10000002.384185791] tolerance 0.01
"""
SUBTLE_MODEL = """NLMODEL2
3 2
0
hidden 3 1 0 0
0
1 1 1
1 1 1
out 1 1 0 0
0
100000000
1
END
"""

MODEL = """NLMODEL2
2 2
0
hidden 2 1 1 0
0
1 1
1 1
out 1 1 0 0
0
100000000
1
END
"""


class AuditSemanticSafety(unittest.TestCase):
    def test_compiled_nir_cancellation(self):
        """The actual .nir surface must not print 0 for a canonical value 5."""
        for compiler in ("g++", "clang++"):
            if not shutil.which(compiler):
                self.skipTest(f"{compiler} unavailable")
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                src = root / "cancellation.nir"
                src.write_text(CANCELLATION_NIR)
                (root / "cancellation.nlw").write_text(MODEL)
                exe = root / "cancellation"
                build = subprocess.run(
                    [sys.executable, str(LANG / "compiler.py"), str(src),
                     "-o", str(exe), "--compiler", compiler],
                    capture_output=True, text=True, timeout=80,
                )
                self.assertEqual(build.returncode, 0, build.stderr + build.stdout)
                run = subprocess.run(
                    [str(exe)], cwd=root, capture_output=True, text=True, timeout=20
                )
                self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
                self.assertIn("EXPECT PASS tolerance=", run.stdout)
                self.assertIn("PREDICT version=2 values=5", run.stdout)

    def test_compiled_nir_subtle_cancellation(self):
        """The default incremental path must reject a >2-unit rounding error."""
        for compiler in ("g++", "clang++"):
            if not shutil.which(compiler):
                self.skipTest(f"{compiler} unavailable")
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                src = root / "subtle.nir"
                src.write_text(SUBTLE_CANCELLATION_NIR)
                (root / "subtle.nlw").write_text(SUBTLE_MODEL)
                exe = root / "subtle"
                build = subprocess.run(
                    [sys.executable, str(LANG / "compiler.py"), str(src),
                     "-o", str(exe), "--compiler", compiler],
                    capture_output=True, text=True, timeout=80,
                )
                self.assertEqual(build.returncode, 0, build.stderr + build.stdout)
                run = subprocess.run(
                    [str(exe)], cwd=root, capture_output=True, text=True, timeout=20,
                )
                self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
                self.assertIn("EXPECT PASS tolerance=", run.stdout)

    def test_canonical_full_delta_trajectory(self):
        """720 ticks x 4 topologies, checking every activation and preactivation."""
        src = Path(__file__).with_name("runtime_canonical_trace.cpp")
        for compiler in ("g++", "clang++"):
            if not shutil.which(compiler):
                self.skipTest(f"{compiler} unavailable")
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory() as tmp:
                exe = Path(tmp) / "canonical"
                build = subprocess.run(
                    [compiler, "-std=c++17", "-O2", "-ffp-contract=off",
                     "-Wall", "-Wextra", "-Werror", "-I", str(LANG / "runtime"),
                     str(src), "-o", str(exe)],
                    capture_output=True, text=True, timeout=90,
                )
                self.assertEqual(build.returncode, 0, build.stderr + build.stdout)
                run = subprocess.run(
                    [str(exe)], capture_output=True, text=True, timeout=35,
                )
                self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
                self.assertIn(
                    "CANONICAL_TRACE_PASS cases=4 steps=720 checks=2880", run.stdout,
                )

    def test_cpp_runtime_counterexamples(self):
        """Covers sequence/DAG AUTO+DELTA, mode, mixed orders, and sequential/DAG SGD."""
        for compiler in ("g++", "clang++"):
            if not shutil.which(compiler):
                self.skipTest(f"{compiler} unavailable")
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory() as tmp:
                exe = Path(tmp) / "safety"
                build = subprocess.run(
                    [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra",
                     "-Werror", "-I", str(LANG / "runtime"),
                     str(CPP_REGRESSION), "-o", str(exe)],
                    capture_output=True, text=True, timeout=80,
                )
                self.assertEqual(build.returncode, 0, build.stderr + build.stdout)
                run = subprocess.run(
                    [str(exe)], capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
                self.assertIn("SEMANTIC_SAFETY PASS cases=10 failed=0", run.stdout)


if __name__ == "__main__":
    unittest.main()
