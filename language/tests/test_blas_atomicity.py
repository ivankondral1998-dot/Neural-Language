"""Check selective BLAS counter/cache commit after a downstream numeric rejection.

The CBLAS shim is intentionally deterministic and tests the runtime call
contract without claiming performance or ABI validation of a system BLAS.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CBLAS_H = r"""
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
typedef enum { CblasRowMajor = 101, CblasColMajor = 102 } CBLAS_LAYOUT;
typedef enum { CblasNoTrans = 111, CblasTrans = 112 } CBLAS_TRANSPOSE;
void cblas_dgemv(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, double,
                 const double*, int, const double*, int, double, double*, int);
#ifdef __cplusplus
}
#endif
"""
PROGRAM = r"""
#include "neural_runtime.hpp"
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

extern "C" void cblas_dgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans,
                             int m, int n, double alpha, const double *mat,
                             int lda, const double *x, int incx,
                             double beta, double *y, int incy) {
  if (layout != CblasRowMajor || trans != CblasNoTrans || lda != n ||
      incx != 1 || incy != 1) throw std::runtime_error("test GEMV args");
  for (int row = 0; row < m; ++row) {
    double sum = 0.0;
    for (int col = 0; col < n; ++col)
      sum += mat[row * lda + col] * x[col];
    y[row] = alpha * sum + beta * y[row];
  }
}

int main() {
  neural::Model m("atomic", 16);
  m.layer("h", 16, 0, 1, 0); // 256 weights: BLAS eligible.
  m.layer("out", 1, 0, 2, 0); // Scalar, can reject after the first GEMV.
  std::fill(m.layers[0].w.begin(), m.layers[0].w.end(), 0.0);
  std::fill(m.layers[0].b.begin(), m.layers[0].b.end(), 0.0);
  std::fill(m.layers[1].w.begin(), m.layers[1].w.end(), 0.0);
  std::fill(m.layers[1].b.begin(), m.layers[1].b.end(), 0.0);
  m.layers[1].w[0] = 5.0e8;
  m.mode = neural::Model::DELTA;
  m.observe(std::vector<double>(16, 1.0));
  m.infer(false);
  if (m.counters.blas_layers != 1 || !m.cache) return 1;

  const auto prior_z = m.z, prior_a = m.a;
  const auto prior_blas = m.counters.blas_layers;
  const auto prior_delta = m.counters.delta;
  m.change("h", 0, 0, 2.0);
  bool rejected = false;
  try { m.infer(false); }
  catch (const std::runtime_error&) { rejected = true; }
  if (!rejected) return 2;
  if (m.z != prior_z || m.a != prior_a || !m.cache) return 3;
  if (m.counters.blas_layers != prior_blas ||
      m.counters.delta != prior_delta) return 4;

  // The failed inference must not poison the pending edits or next commit.
  m.change("h", 0, 0, -2.0);
  const auto& result = m.infer(false);
  if (result.size() != 1 || result[0] != 0.0) return 5;
  if (m.counters.blas_layers != prior_blas + 1 ||
      m.counters.delta != prior_delta + 1) return 6;
  std::cout << "BLAS_ATOMICITY_PASS\n";
  return 0;
}
"""

class BlasAtomicityTests(unittest.TestCase):
    def test_downstream_rejection_does_not_count_aborted_gemv(self):
        compilers = [shutil.which(n) for n in ("g++", "clang++")]
        if not all(compilers):
            self.skipTest("GCC and Clang both required for this cross-compiler check")
        with tempfile.TemporaryDirectory(prefix="nl_blas_atomicity_") as td:
            root = Path(td)
            (root / "cblas.h").write_text(CBLAS_H, encoding="utf-8")
            (root / "case.cpp").write_text(PROGRAM, encoding="utf-8")
            for compiler in compilers:
                for flags in (["-O2"], ["-O2", "-DNDEBUG"]):
                    with self.subTest(compiler=compiler, flags=flags):
                        exe = root / "blas_case"
                        build = subprocess.run(
                            [compiler, "-std=c++17", *flags, "-Wall", "-Wextra", "-Werror",
                             "-DNEURAL_RUNTIME_CBLAS",
                             "-I", str(root), "-I", str(ROOT / "runtime"),
                             str(root / "case.cpp"), "-o", str(exe)],
                            capture_output=True, text=True, timeout=60)
                        self.assertEqual(build.returncode, 0, build.stderr)
                        ran = subprocess.run([str(exe)], capture_output=True,
                                             text=True, timeout=20)
                        self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
                        self.assertIn("BLAS_ATOMICITY_PASS", ran.stdout)

if __name__ == "__main__":
    unittest.main()
