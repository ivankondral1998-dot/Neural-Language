#!/usr/bin/env python3
"""Independent CartPole PPO oracle for Neural Language's pretrained-model pilot.

Read-only. Uses Python stdlib (not neural_runtime.hpp) to validate a known
4->64(tanh)->64(tanh)->2(linear) NLMODEL2 policy and deterministic CartPole
Euler trajectories. This checks the *model* and an independent forward path;
it does NOT execute the .nir compiler, native C++ runtime, or Android.

Upstream policy: https://huggingface.co/hwihwalab/cartpole-v1-ppo
Upstream project reports MIT license; check upstream LICENSE when redistributing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

LAYERS = (("hidden1", 4, 64, 3), ("hidden2", 64, 64, 3), ("actions", 64, 2, 0))
EXPECTED_VECTORS = (
    ((0.0, 0.0, 0.0, 0.0), (0.12913951301483756, -0.12815577824153035)),
    ((0.02, -0.01, 0.04, -0.03), (-0.3736812805283707, 0.3745496992378324)),
    ((-0.01, 0.04, -0.04, 0.01), (0.7849797970119223, -0.7840781729061173)),
    ((0.05, 0.05, 0.05, 0.05), (-2.051765586405924, 2.053296764932673)),
    ((-0.05, -0.05, -0.05, -0.05), (2.346353593857841, -2.3461097929302865)),
)


def _numbers(line: str, size: int, label: str) -> tuple[float, ...]:
    values = line.split()
    if len(values) != size:
        raise ValueError(f"{label}: expected {size} values, got {len(values)}")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or abs(value) >= 1e9 for value in result):
        raise ValueError(f"{label}: invalid number")
    return result


def parse_model(raw: str) -> tuple[tuple[int, int, int, tuple[float, ...], tuple[float, ...]], ...]:
    """Strict parser for this specific pretrained model, not a general NLMODEL2 loader."""
    lines = raw.splitlines()
    if len(lines) != 16 or lines[:3] != ["NLMODEL2", "4 3", "0"] or lines[-1] != "END":
        raise ValueError("unexpected NLMODEL2 file structure")
    layers = []
    for layer_num, (name, n_in, n_out, activation) in enumerate(LAYERS):
        start = 3 + layer_num * 4
        if lines[start].split() != [name, str(n_in), str(n_out), str(activation), "0"]:
            raise ValueError(f"{name}: incompatible metadata")
        biases = _numbers(lines[start + 1], n_out, name + " biases")
        weights = _numbers(lines[start + 2], n_in * n_out, name + " weights")
        masks = lines[start + 3].split()
        if len(masks) != n_in * n_out or any(mask != "1" for mask in masks):
            raise ValueError(f"{name}: masks must contain {n_in * n_out} ones")
        layers.append((n_in, n_out, activation, biases, weights))
    return tuple(layers)


def predict(layers, state: tuple[float, ...]) -> tuple[float, ...]:
    x = state
    for n_in, n_out, activation, biases, weights in layers:
        if len(x) != n_in:
            raise ValueError("invalid observation dimension")
        result = []
        for row in range(n_out):
            z = biases[row]
            for col in range(n_in):
                z += weights[row * n_in + col] * x[col]
            result.append(math.tanh(z) if activation == 3 else z)
        x = tuple(result)
    return x


def initial_state(seed: int) -> tuple[float, float, float, float]:
    # Same deterministic LCG as the C++ pilot, NOT Gymnasium's random generator.
    a = seed & 0xFFFFFFFF
    state = []
    for _ in range(4):
        a = (1664525 * a + 1013904223) & 0xFFFFFFFF
        state.append((a / 4294967296.0 - 0.5) * 0.1)
    return tuple(state)


def step(state: tuple[float, ...], action: int) -> tuple[tuple[float, ...], bool]:
    x, x_dot, theta, theta_dot = state
    force = 10.0 if action else -10.0
    costheta, sintheta = math.cos(theta), math.sin(theta)
    temp = (force + 0.05 * theta_dot * theta_dot * sintheta) / 1.1
    thetaacc = (9.8 * sintheta - costheta * temp) / (
        0.5 * (4.0 / 3.0 - 0.1 * costheta * costheta / 1.1)
    )
    xacc = temp - 0.05 * thetaacc * costheta / 1.1
    result = (
        x + 0.02 * x_dot,
        x_dot + 0.02 * xacc,
        theta + 0.02 * theta_dot,
        theta_dot + 0.02 * thetaacc,
    )
    alive = not (abs(result[0]) > 2.4 or abs(result[2]) > math.radians(12.0))
    return result, alive


def evaluate(layers, episodes: int) -> dict:
    if not 1 <= episodes <= 1000:
        raise ValueError("episodes must be 1..1000")
    lengths, min_action_margin = [], float("inf")
    for episode in range(1, episodes + 1):
        state = initial_state(episode)
        length = 0
        for _ in range(500):
            left, right = predict(layers, state)
            if not math.isfinite(left) or not math.isfinite(right):
                raise ValueError("nonfinite prediction")
            min_action_margin = min(min_action_margin, abs(right - left))
            state, alive = step(state, int(right > left))
            length += 1
            if not alive:
                break
        lengths.append(length)
    return {
        "episodes": episodes,
        "steps": sum(lengths),
        "solved_500": sum(length == 500 for length in lengths),
        "min_episode": min(lengths),
        "max_episode": max(lengths),
        "mean_episode": sum(lengths) / episodes,
        "min_action_margin": min_action_margin,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="path to existing cartpole_pretrained.nlw")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    raw_bytes = args.model.read_bytes()
    layers = parse_model(raw_bytes.decode("utf-8"))
    for state, expected in EXPECTED_VECTORS:
        actual = predict(layers, state)
        for got, wanted in zip(actual, expected):
            if not math.isclose(got, wanted, rel_tol=1e-9, abs_tol=1e-9):
                raise AssertionError(f"reference vector mismatch: {got} != {wanted}")
    report = evaluate(layers, args.episodes)
    report["model_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    report["parameter_count"] = sum(len(layer[3]) + len(layer[4]) for layer in layers)
    report["golden_vectors"] = len(EXPECTED_VECTORS)
    print(json.dumps(report, sort_keys=True, indent=2))
    if report["solved_500"] != args.episodes:
        raise AssertionError("pretrained agent did not complete all reference episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
