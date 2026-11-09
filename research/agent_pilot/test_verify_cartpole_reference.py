#!/usr/bin/env python3
"""Focused, stdlib-only unit tests for the independent CartPole oracle."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).with_name("verify_cartpole_reference.py")
spec = importlib.util.spec_from_file_location("verify_cartpole_reference", SCRIPT)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def fixture():
    lines = ["NLMODEL2", "4 3", "0"]
    for name, n_in, n_out, activation in oracle.LAYERS:
        lines += [
            f"{name} {n_in} {n_out} {activation} 0",
            " ".join(["0"] * n_out),
            " ".join(["0"] * (n_in * n_out)),
            " ".join(["1"] * (n_in * n_out)),
        ]
    return "\n".join(lines + ["END", ""])


class VerifyCartpoleReferenceTest(unittest.TestCase):
    def test_zeros_model_matches_expected_shape(self):
        layers = oracle.parse_model(fixture())
        self.assertEqual(sum(len(x[3]) + len(x[4]) for x in layers), 4610)
        self.assertEqual(oracle.predict(layers, (0, 0, 0, 0)), (0.0, 0.0))

    def test_corrupt_or_incompatible_models_rejected(self):
        raw = fixture()
        variants = [
            raw.replace("NLMODEL2", "BADMODEL", 1),
            raw.replace("4 3", "5 3", 1),
            raw.replace("hidden1 4 64 3 0", "hidden1 4 65 3 0", 1),
            raw.replace("hidden1 4 64 3 0", "hidden1 4 64 1 0", 1),
            raw.replace("hidden1 4 64 3 0", "hidden1 4 64 3 1", 1),
            raw.replace("END", "", 1),
            raw.replace("0 0 0 0", "nan 0 0 0", 1),
            raw.replace("1 1 1 1", "0 1 1 1", 1),
        ]
        for invalid in variants:
            with self.subTest(invalid=invalid[:100]):
                with self.assertRaises(ValueError):
                    oracle.parse_model(invalid)

    def test_euler_physics_and_seed_are_deterministic(self):
        self.assertEqual(oracle.initial_state(123), oracle.initial_state(123))
        s = oracle.initial_state(42)
        after, alive = oracle.step(s, 1)
        self.assertTrue(alive)
        self.assertNotEqual(after, s)
        self.assertEqual(oracle.step(s, 1), (after, alive))

    def test_zero_policy_does_not_solve_cartpole(self):
        layers = oracle.parse_model(fixture())
        result = oracle.evaluate(layers, episodes=5)
        self.assertEqual(result["episodes"], 5)
        self.assertLess(result["solved_500"], 5)


if __name__ == "__main__":
    unittest.main()
