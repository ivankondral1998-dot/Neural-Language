#!/usr/bin/env python3
"""Compile the real Neural Language C++ runtime and cross-check a pretrained policy.

Run from a checked-out Neural-Language branch containing neural_runtime.hpp and
cartpole_pretrained.nlw: python3 research/agent_pilot/compare_native_cartpole.py
Requires a host C++17 compiler. Does not require Gymnasium, NumPy or network.
Does not assess latency, energy or Android compatibility.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable

from verify_cartpole_reference import evaluate, initial_state, parse_model, predict, step

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MODEL = HERE / "models" / "cartpole_pretrained.nlw"
SOURCE = HERE / "agent_cartpole_trace.cpp"
HEADER_DIR = ROOT / "language" / "runtime"


def expected_trace(layers, episodes: int):
    if not 1 <= episodes <= 1000:
        raise ValueError("episodes must be 1..1000")
    for episode in range(1, episodes + 1):
        state = initial_state(episode)
        for tick in range(500):
            outputs = predict(layers, state)
            action = int(outputs[1] > outputs[0])
            yield episode, tick, state, outputs, action
            state, alive = step(state, action)
            if not alive:
                break


def _close(a: float, b: float, *, tolerance: float = 2e-9) -> bool:
    return math.isfinite(a) and math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


def verify_trace(lines: Iterable[str], layers, episodes: int, *, require_solved: bool = True) -> dict:
    reference = expected_trace(layers, episodes)
    total, worst_error, last_episode_steps = 0, 0.0, 0
    solved = 0
    observed_summary = None
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "TRACE_SUMMARY":
            if observed_summary is not None or len(parts) != 3:
                raise ValueError("invalid or repeated TRACE_SUMMARY")
            observed_summary = (int(parts[1]), int(parts[2]))
            continue
        if parts[0] != "TRACE" or len(parts) != 12 or observed_summary is not None:
            raise ValueError("invalid TRACE row")
        try:
            row_episode, tick = int(parts[1]), int(parts[2])
            numbers = tuple(float(v) for v in parts[3:11])
            row_action = int(parts[11])
            target_episode, target_tick, state, output, action = next(reference)
        except (ValueError, StopIteration) as err:
            raise ValueError("malformed or surplus TRACE row") from err
        if (row_episode, tick) != (target_episode, target_tick):
            raise ValueError(f"out-of-order step: {(row_episode, tick)} vs {(target_episode, target_tick)}")
        expected = (*state, *output, *output)
        for idx, (actual, wanted) in enumerate(zip(numbers, expected)):
            if not _close(actual, wanted):
                raise ValueError(f"prediction/state mismatch at episode={row_episode}, step={tick}, field={idx}: {actual} vs {wanted}")
            worst_error = max(worst_error, abs(actual - wanted))
        if row_action != action:
            raise ValueError(f"wrong action at episode={row_episode}, step={tick}")
        if tick == 0:
            if total and last_episode_steps == 500:
                solved += 1
            last_episode_steps = 0
        last_episode_steps = tick + 1
        total += 1
    if next(reference, None) is not None:
        raise ValueError("native TRACE ended before reference")
    if observed_summary != (episodes, total):
        raise ValueError(f"TRACE_SUMMARY mismatch: {observed_summary} vs {(episodes, total)}")
    if last_episode_steps == 500:
        solved += 1
    # An early-terminated episode is verified by expected_trace; require_solved
    # is an optional performance gate independent of numerical correctness.
    if require_solved and solved != episodes:
        raise ValueError(f"only {solved}/{episodes} episodes reached 500 steps")
    return {"episodes": episodes, "steps": total, "solved_500": solved,
            "max_abs_cpp_vs_python": worst_error, "oracle": "independent stdlib FP64"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--model", type=Path, default=MODEL)
    args = parser.parse_args()
    layers = parse_model(args.model.read_text(encoding="utf-8"))
    if not SOURCE.is_file() or not (HEADER_DIR / "neural_runtime.hpp").is_file():
        raise RuntimeError("missing C++ pilot source or neural_runtime.hpp; run in the complete repository")
    with tempfile.TemporaryDirectory(prefix="nl_cartpole_") as build:
        exe = Path(build) / "agent_cartpole_trace"
        compile_command = [args.compiler, "-std=c++17", "-O2", "-I", str(HEADER_DIR), str(SOURCE), "-o", str(exe)]
        subprocess.run(compile_command, check=True, capture_output=True, text=True)
        executed = subprocess.run([str(exe), str(args.model), str(args.episodes)], check=True, capture_output=True, text=True)
        result = verify_trace(executed.stdout.splitlines(), layers, args.episodes)
    result["baseline_python_episodes"] = evaluate(layers, args.episodes)["solved_500"]
    result["compiler"] = args.compiler
    result["status"] = "PASS"
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
