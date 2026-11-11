"""P0 counterexamples: canonical DELTA versus an independent reduction, and atomic SGD.

Historical logs are not evidence. These tests compile the current runtime and run it.
"""
import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = r'''
#include "neural_runtime.hpp"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
static int bits_differ(double x, double y) {
  uint64_t a, b;
  static_assert(sizeof(double) == 8, "double");
  std::memcpy(&a, &x, 8);
  std::memcpy(&b, &y, 8);
  return a != b;
}
static double oracle_act(double z, int type) {
  if (type == 1) return z > 0.0 ? z : 0.0;
  if (type == 2) return z >= 0.0 ? 1.0 / (1.0 + std::exp(-z)) : std::exp(z) / (1.0 + std::exp(z));
  return z;
}
struct OLayer { int in, out, type; std::vector<double> w, b; std::vector<uint8_t> enabled; };
static void oracle_forward(const std::vector<OLayer>& layers, const std::vector<double>& input,
                            std::vector<double>& out) {
  std::vector<double> prev = input, cur;
  for (const auto& l : layers) {
    cur.assign(l.out, 0.0);
    for (int r = 0; r < l.out; ++r) {
      double total = l.b[r];
      for (int c = 0; c < l.in; ++c) {
        size_t ix = static_cast<size_t>(r) * l.in + c;
        if (l.enabled[ix]) total += l.w[ix] * prev[c];
      }
      cur[r] = oracle_act(total, l.type);
    }
    prev.swap(cur);
  }
  out = prev;
}
static int fail(int code, const char* msg) { std::cerr << msg << "\n"; return code; }
int main() {
  // A. Cancellation inside one FULL reduction. Algebraic z+=dw*x yields 1, not 2.
  {
    neural::Model d("a", 3);
    d.mode = neural::Model::DELTA;
    d.layer("out", 1, 0, 1, 0);
    d.layers[0].w = {1e16, 1.0, -1e16};
    d.layers[0].b = {0.0};
    d.observe({1.0, 1.0, 1.0});
    d.infer(false);
    d.change("out", 0, 1, 1.0);
    double got = d.infer(false)[0];
    double oracle = 0.0;
    double w[3] = {1e16, 2.0, -1e16};
    for (int c = 0; c < 3; ++c) oracle += w[c] * 1.0;
    if (bits_differ(got, oracle)) return fail(2, "case A delta != independent reduction");
    if (d.counters.delta == 0) return fail(3, "case A did not use incremental path");
  }
  // B. Same cancellation through a dependent layer.
  {
    neural::Model d("b", 1);
    d.mode = neural::Model::DELTA;
    d.layer("h", 3, 0, 1, 0);
    d.layer("out", 1, 0, 2, 0);
    d.layers[0].w = {1.0, 1.0, 1.0};
    d.layers[0].b = {0.0, 0.0, 0.0};
    d.layers[1].w = {1e16, 1.0, -1e16};
    d.layers[1].b = {0.0};
    d.observe({1.0});
    d.infer(false);
    d.change("h", 1, 0, 1.0);
    double got = d.infer(false)[0];
    double oracle = 0.0;
    double h[3] = {1.0, 2.0, 1.0};
    double w[3] = {1e16, 1.0, -1e16};
    for (int c = 0; c < 3; ++c) oracle += w[c] * h[c];
    if (bits_differ(got, oracle)) return fail(4, "case B propagation != independent reduction");
  }
  // C. Rejected weight must not keep an earlier assignment from the same step.
  {
    neural::Model m("c", 2);
    m.layer("out", 1, 0, 3, 0);
    m.layers[0].w = {1.0, 1e7};
    m.layers[0].b = {0.0};
    m.observe({1.0, 10.0});
    m.infer(false);
    auto w = m.layers[0].w;
    auto b = m.layers[0].b;
    auto version = m.version;
    bool threw = false;
    try { m.learn({-1e8}, 1.0); }
    catch (const std::runtime_error&) { threw = true; }
    if (!threw) return fail(5, "case C did not reject");
    if (m.layers[0].w != w || m.layers[0].b != b || m.version != version)
      return fail(6, "case C published a partial SGD step");
  }
  // D. Rejected bias must roll back a weight staged in the same step.
  {
    neural::Model m("d", 1);
    m.layer("out", 1, 0, 4, 0);
    m.layers[0].w = {0.0};
    m.layers[0].b = {0.0};
    m.observe({1e-6});
    m.infer(false);
    auto w = m.layers[0].w;
    auto b = m.layers[0].b;
    auto version = m.version;
    bool threw = false;
    try { m.learn({-6e8}, 1.0); }
    catch (const std::runtime_error&) { threw = true; }
    if (!threw) return fail(7, "case D did not reject");
    if (m.layers[0].w != w || m.layers[0].b != b || m.version != version)
      return fail(8, "case D published a partial SGD step");
  }
  // E. A successful step still publishes, and a later rejection does not undo it
  // beyond the failed step. Also compare a short FP64 trajectory with the oracle.
  {
    neural::Model d("e", 2);
    d.mode = neural::Model::DELTA;
    d.layer("h", 2, 1, 5, 0);
    d.layer("out", 1, 0, 6, 0);
    d.layers[0].w = {0.2, -0.4, 0.5, 0.1};
    d.layers[0].b = {0.0, -0.3};
    d.layers[1].w = {0.25, -0.75};
    d.layers[1].b = {0.1};
    d.observe({0.5, -1.25});
    std::vector<OLayer> ref(2);
    ref[0] = {2, 2, 1, d.layers[0].w, d.layers[0].b, d.layers[0].enabled};
    ref[1] = {2, 1, 0, d.layers[1].w, d.layers[1].b, d.layers[1].enabled};
    auto check = [&](int code) {
      std::vector<double> oracle;
      oracle_forward(ref, d.input, oracle);
      double got = d.infer(false)[0];
      if (oracle.size() != 1 || bits_differ(got, oracle[0])) return code;
      return 0;
    };
    if (int code = check(9)) return fail(code, "case E initial");
    d.change("h", 0, 1, 0.125);
    ref[0].w[1] += 0.125;
    if (int code = check(10)) return fail(code, "case E after edit");
    d.gate("out", 0, 0, false);
    ref[1].enabled[0] = 0;
    if (int code = check(11)) return fail(code, "case E after gate");
    auto before = d.layers;
    auto version = d.version;
    d.learn({0.2}, 0.1);
    if (d.version != version + 1) return fail(12, "case E successful learn did not publish");
    if (d.layers[1].b == before[1].b) return fail(13, "case E bias not published");
    auto published = d.layers;
    auto published_version = d.version;
    bool threw = false;
    try { d.learn({-6e8}, 1.0); }
    catch (const std::runtime_error&) { threw = true; }
    if (!threw) return fail(14, "case E second learn did not reject");
    if (d.layers[0].w != published[0].w || d.layers[1].w != published[1].w ||
        d.layers[0].b != published[0].b || d.layers[1].b != published[1].b ||
        d.version != published_version)
      return fail(15, "case E rejection rolled back a committed step or published a new one");
  }
  // F. AUTO must be allowed to take the suffix and still match FULL bit for bit.
  {
    neural::Model a("auto", 40);
    a.mode = neural::Model::AUTO;
    a.layer("big", 40, 0, 7, 0);
    a.layer("out", 1, 0, 8, 0);
    std::vector<double> x(40, 0.01);
    a.observe(x);
    a.infer(false);
    a.change("out", 0, 0, 0.02);
    double got = a.infer(false)[0];
    neural::Model f = a.isolated_copy();
    f.mode = neural::Model::FULL;
    f.cache = false;
    if (bits_differ(got, f.infer(false)[0])) return fail(16, "case F AUTO != FULL");
    if (a.counters.delta == 0) return fail(17, "case F AUTO did not use the suffix");
  }
  auto same_state = [](const neural::Model& d, const neural::Model& f, int code) {
    if (d.layers.size() != f.layers.size()) return code;
    for (size_t k = 0; k < d.layers.size(); ++k) {
      if (d.z[k].size() != f.z[k].size() || d.a[k].size() != f.a[k].size()) return code;
      for (size_t j = 0; j < d.z[k].size(); ++j)
        if (bits_differ(d.z[k][j], f.z[k][j]) || bits_differ(d.a[k][j], f.a[k][j])) return code;
    }
    return 0;
  };
  // G. Many edits of the early layer stay bitwise equal to FULL.
  {
    neural::Model d("g", 4);
    d.mode = neural::Model::DELTA;
    d.layer("h", 6, 1, 9, 0);
    d.layer("out", 3, 0, 10, 0);
    d.observe({1.0, -0.5, 0.25, 0.125});
    d.infer(false);
    for (int i = 0; i < 40; ++i) {
      d.change("h", i % 6, i % 4, (i % 2 ? 0.01 : -0.01));
      d.infer(false);
    }
    neural::Model f = d.isolated_copy();
    f.mode = neural::Model::FULL;
    f.cache = false;
    f.infer(false);
    if (int code = same_state(d, f, 18)) return fail(code, "case G early edits != FULL");
    if (d.counters.delta == 0) return fail(19, "case G did not use the selective path");
  }
  // H. A ReLU row that stays inactive updates z and does not move downstream bits.
  {
    neural::Model d("h", 1);
    d.mode = neural::Model::DELTA;
    d.layer("h", 1, 1, 2, 0);
    d.layer("out", 1, 0, 3, 0);
    d.layers[0].w = {-1.0};
    d.layers[0].b = {-2.0};
    d.layers[1].w = {4.0};
    d.layers[1].b = {0.5};
    d.observe({1.0});
    d.infer(false);
    const double downstream = d.a[1][0];
    d.change("h", 0, 0, -0.25);
    d.infer(false);
    neural::Model f = d.isolated_copy();
    f.mode = neural::Model::FULL;
    f.cache = false;
    f.infer(false);
    if (int code = same_state(d, f, 20)) return fail(code, "case H relu != FULL");
    if (bits_differ(d.a[0][0], 0.0) || bits_differ(d.a[1][0], downstream))
      return fail(21, "case H inactive relu changed downstream");
  }
  // I. A disabled edge is part of the same reduction as FULL.
  {
    neural::Model d("i", 2);
    d.mode = neural::Model::DELTA;
    d.layer("h", 2, 0, 4, 0);
    d.layer("out", 2, 0, 5, 0);
    d.observe({0.5, -1.0});
    d.infer(false);
    d.gate("h", 1, 0, false);
    d.infer(false);
    d.change("out", 0, 1, 0.125);
    d.infer(false);
    neural::Model f = d.isolated_copy();
    f.mode = neural::Model::FULL;
    f.cache = false;
    f.infer(false);
    if (int code = same_state(d, f, 22)) return fail(code, "case I mask != FULL");
  }
  // J. INT8 edits follow the stored code, including a sub-quantum no-op.
  {
    neural::Model d("j", 2);
    d.mode = neural::Model::DELTA;
    d.layer("h", 2, 0, 6, 1);
    d.layer("out", 1, 0, 7, 0);
    d.observe({0.3, -0.7});
    d.infer(false);
    d.change("h", 0, 1, d.layers[0].scale * 0.1);
    d.infer(false);
    d.change("h", 1, 0, d.layers[0].scale * 3.0);
    d.infer(false);
    neural::Model f = d.isolated_copy();
    f.mode = neural::Model::FULL;
    f.cache = false;
    f.infer(false);
    if (int code = same_state(d, f, 23)) return fail(code, "case J int8 != FULL");
  }
  // K. A rejected preactivation must not publish a half-updated cache.
  {
    neural::Model m("k", 1);
    m.mode = neural::Model::DELTA;
    m.layer("out", 1, 0, 8, 0);
    m.layers[0].w = {0.0};
    m.layers[0].b = {0.0};
    m.observe({20.0});
    m.infer(false);
    const double z0 = m.z[0][0], a0 = m.a[0][0];
    bool threw = false;
    try { m.change("out", 0, 0, 5.0e7); m.infer(false); }
    catch (const std::runtime_error&) { threw = true; }
    if (!threw) return fail(24, "case K did not reject");
    if (bits_differ(m.z[0][0], z0) || bits_differ(m.a[0][0], a0))
      return fail(25, "case K published a partial activation");
  }
  std::cout << "P0_NUMERIC_PASS\n";
  return 0;
}
'''


class NumericP0Tests(unittest.TestCase):
    def test_counterexamples_gcc_and_clang(self):
        compilers = []
        for name in ("g++", "clang++"):
            path = shutil.which(name)
            if path:
                compilers.append(path)
        if not compilers:
            self.skipTest("no C++ compiler")
        self.assertIn("g++", {pathlib.Path(c).name for c in compilers})
        self.assertIn("clang++", {pathlib.Path(c).name for c in compilers})
        for compiler in compilers:
            for flags in (["-O2"], ["-O2", "-DNDEBUG"], ["-O0"]):
                with self.subTest(compiler=compiler, flags=flags):
                    with tempfile.TemporaryDirectory() as td:
                        src = pathlib.Path(td) / "p0.cpp"
                        exe = pathlib.Path(td) / "p0"
                        src.write_text(SOURCE)
                        build = subprocess.run(
                            [compiler, "-std=c++17", *flags, "-Wall", "-Wextra", "-Werror",
                             "-I", str(ROOT / "runtime"), str(src), "-o", str(exe)],
                            capture_output=True, text=True, timeout=60)
                        self.assertEqual(build.returncode, 0, build.stderr)
                        ran = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
                        self.assertEqual(ran.returncode, 0, ran.stdout + "\n" + ran.stderr)
                        self.assertIn("P0_NUMERIC_PASS", ran.stdout)


if __name__ == "__main__":
    unittest.main()
