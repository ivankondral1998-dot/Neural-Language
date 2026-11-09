"""Stdlib tests for the independent C++/Python CartPole trace comparator."""
import importlib.util
import contextlib
import io
import tempfile
from unittest import mock
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("compare_native_cartpole", HERE / "compare_native_cartpole.py")
# This works from a clean checkout as well as in the developer's test directory.
import sys
sys.path.insert(0, str(HERE))
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)
from test_verify_cartpole_reference import fixture


class NativeTraceComparisonTest(unittest.TestCase):
    def setUp(self):
        self.layers = cmp.parse_model(fixture())
        self.rows = []
        for ep, tick, state, out, action in cmp.expected_trace(self.layers, 3):
            values = (ep, tick, *state, *out, *out, action)
            self.rows.append("TRACE " + " ".join(str(v) for v in values))
        self.rows.append(f"TRACE_SUMMARY 3 {len(self.rows)}")

    def test_reference_fixture_accepted(self):
        report = cmp.verify_trace(self.rows, self.layers, 3, require_solved=False)
        self.assertEqual(report["steps"], len(self.rows) - 1)
        self.assertEqual(report["solved_500"], 0)

    def test_missing_record_rejected(self):
        with self.assertRaisesRegex(ValueError, "out-of-order"):
            cmp.verify_trace(self.rows[:2] + self.rows[3:], self.layers, 3, require_solved=False)

    def test_wrong_action_rejected(self):
        tampered = self.rows.copy()
        tokens = tampered[0].split()
        tokens[-1] = "1" if tokens[-1] == "0" else "0"
        tampered[0] = " ".join(tokens)
        with self.assertRaisesRegex(ValueError, "wrong action"):
            cmp.verify_trace(tampered, self.layers, 3, require_solved=False)

    def test_wrong_prediction_rejected(self):
        tampered = self.rows.copy()
        tokens = tampered[0].split()
        tokens[7] = "1234.0"
        tampered[0] = " ".join(tokens)
        with self.assertRaisesRegex(ValueError, "prediction/state mismatch"):
            cmp.verify_trace(tampered, self.layers, 3, require_solved=False)

    def test_summary_rejected(self):
        with self.assertRaisesRegex(ValueError, "SUMMARY mismatch"):
            cmp.verify_trace(self.rows[:-1] + ["TRACE_SUMMARY 3 123"], self.layers, 3, require_solved=False)

    def test_nonfinite_rejected(self):
        tampered = self.rows.copy()
        tokens = tampered[0].split()
        tokens[7] = "nan"
        tampered[0] = " ".join(tokens)
        with self.assertRaisesRegex(ValueError, "prediction/state mismatch"):
            cmp.verify_trace(tampered, self.layers, 3, require_solved=False)

    def test_load_notices_are_strict_and_removed(self):
        notices = ["MODEL_LOADED cartpole_pretrained.nlw version=1"] * 2
        result = cmp.strip_model_logs([*notices, *self.rows], "cartpole_pretrained.nlw")
        self.assertEqual(result, self.rows)

    def test_wrong_load_path_or_extra_output_rejected(self):
        bad = ["MODEL_LOADED /tmp/policy.nlw version=1"] * 2
        with self.assertRaisesRegex(ValueError, "unexpected model loader"):
            cmp.strip_model_logs(bad + self.rows, "cartpole_pretrained.nlw")
        with self.assertRaises(ValueError):
            cmp.strip_model_logs(["warning", *bad, *self.rows], "cartpole_pretrained.nlw")

    def test_main_stages_model_with_safe_basename(self):
        # Verify subprocess arguments without pretending to execute C++.
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            model = base / "nested" / "unsafe_absolute_path.nlw"
            model.parent.mkdir()
            model.write_text(fixture(), encoding="utf-8")
            source = base / "agent_cartpole_trace.cpp"
            header_dir = base / "runtime"
            header_dir.mkdir()
            source.write_text("// fake compilation input", encoding="utf-8")
            (header_dir / "neural_runtime.hpp").write_text("// fake header", encoding="utf-8")
            calls = []

            def fake_run(command, **kwargs):
                calls.append((command, kwargs))
                if len(calls) == 1:
                    self.assertEqual(command[0], "g++")
                    self.assertIn("-std=c++17", command)
                    return mock.Mock(stdout="", returncode=0)
                self.assertEqual(len(calls), 2)
                self.assertEqual(command[1:], ["cartpole_pretrained.nlw", "3"])
                self.assertEqual(
                    (Path(kwargs["cwd"]) / command[1]).read_bytes(), model.read_bytes()
                )
                return mock.Mock(stdout="MODEL_LOADED cartpole_pretrained.nlw version=1\n" * 2
                                       + "TRACE_SUMMARY 3 0\n", returncode=0)

            with (mock.patch.object(cmp, "SOURCE", source),
                  mock.patch.object(cmp, "HEADER_DIR", header_dir),
                  mock.patch("sys.argv", ["prog", "--episodes", "3", "--model", str(model)]),
                  mock.patch.object(cmp.subprocess, "run", side_effect=fake_run),
                  mock.patch.object(cmp, "verify_trace", return_value={"steps": 0}),
                  mock.patch.object(cmp, "evaluate", return_value={"solved_500": 0}),
                  contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(cmp.main(), 0)
            self.assertEqual(len(calls), 2)

    def test_unexpected_extra_row_rejected(self):
        with self.assertRaises(ValueError):
            cmp.verify_trace(self.rows[:-1] + [self.rows[0], self.rows[-1]], self.layers, 3, require_solved=False)


if __name__ == "__main__":
    unittest.main()
