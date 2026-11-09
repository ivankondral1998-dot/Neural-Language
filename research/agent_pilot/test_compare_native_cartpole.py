"""Stdlib tests for the independent C++/Python CartPole trace comparator."""
import importlib.util
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

    def test_unexpected_extra_row_rejected(self):
        with self.assertRaises(ValueError):
            cmp.verify_trace(self.rows[:-1] + [self.rows[0], self.rows[-1]], self.layers, 3, require_solved=False)


if __name__ == "__main__":
    unittest.main()
