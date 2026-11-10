#!/usr/bin/env python3
"""Independent FP64 561→16→6 MSE/SGD reference and UCI holdout scoring."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

TOLERANCE = 2e-8
EPOCHS = 20
RATE = 0.01
SEED = 20261110
SCALE = 1. / 127.


def load_weights(path):
    words = iter(path.read_text().split())
    assert next(words) == "NLMODEL2"
    assert (int(next(words)), int(next(words))) == (561, 2)
    assert next(words) in ("0", "1", "2")
    result = []
    for name, size_in, size_out, activation in (("hidden", 561, 16, 3), ("activity", 16, 6, 0)):
        assert next(words) == name
        assert tuple(int(next(words)) for _ in range(3)) == (size_in, size_out, activation)
        assert next(words) in ("0", "1")
        b = np.array([float(next(words)) for _ in range(size_out)])
        w = np.array([float(next(words)) for _ in range(size_in * size_out)]).reshape(size_out, size_in)
        assert all(next(words) == "1" for _ in range(size_in * size_out))
        assert np.isfinite(w).all() and np.isfinite(b).all()
        result.extend((w, b))
    assert next(words) == "END" and next(words, None) is None
    return tuple(result)


def forward(params, x):
    w, b, v, c = params
    return np.tanh(x @ w.T + b) @ v.T + c


def gradient(params, x, label):
    w, b, v, c = params
    h = np.tanh(w @ x + b)
    residual = v @ h + c
    residual[label] -= 1.
    out = residual * (2. / 6.)
    hidden = (v.T @ out) * (1. - h * h)
    return np.outer(hidden, x), hidden, np.outer(out, h), out


def step(params, x, label):
    for param, grad in zip(params, gradient(params, x, label)):
        param -= RATE * grad


def train(params, x, y):
    # NumPy's legacy RandomState with an integer seed supplies raw MT19937
    # outputs. The integer multiply-high and swaps mirror the declared order.
    rng = np.random.RandomState(SEED)
    for _ in range(EPOCHS):
        order = list(range(len(y)))
        draws = rng.randint(0, 2**32, size=len(y)-1, dtype=np.uint32)
        for i, draw in zip(range(len(y), 1, -1), draws):
            j = (int(draw) * i) >> 32
            order[i-1], order[j] = order[j], order[i-1]
        for i in order:
            step(params, x[i], y[i])


def finite_differences(params, x, label):
    target = np.eye(6)[label]
    grads = gradient(params, x, label)
    worst = 0.
    for param, grad in zip(params, grads):
        for i in sorted({0, param.size // 3, param.size // 2, param.size - 1}):
            old = param.flat[i]
            eps = 1e-5
            param.flat[i] = old + eps
            high = np.mean((forward(params, x) - target)**2)
            param.flat[i] = old - eps
            low = np.mean((forward(params, x) - target)**2)
            param.flat[i] = old
            worst = max(worst, abs((high - low) / (2*eps) - grad.flat[i]))
    assert worst <= 2e-8, worst
    return worst


def read_logits(path):
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("DATA_PREDICT index="):
            key, value = line.split(" values=", 1)
            assert key == f"DATA_PREDICT index={len(rows)}"
            rows.append([float(v) for v in value.split(",")])
    result = np.array(rows)
    assert result.ndim == 2 and result.shape[1] == 6 and np.isfinite(result).all()
    return result


def max_error(expected, actual, tolerance=TOLERANCE):
    assert expected.shape == actual.shape
    error = float(np.max(np.abs(expected - actual)))
    assert error <= tolerance, (error, tolerance)
    return error


def score(pred, y):
    assert pred.shape == y.shape
    counts = np.zeros((6, 6), dtype=int)
    np.add.at(counts, (y, pred), 1)
    return {"accuracy": float(np.mean(pred == y)), "correct": int(np.sum(pred == y)),
            "confusion_matrix_rows_true": counts.tolist()}


def quantized_copy(params, output):
    def nums(values):
        return " ".join(format(float(v), ".17g") for v in values.flat)

    lines = ["NLMODEL2", "561 2", "1"]
    for name, code, w, b in (("hidden", 3, *params[:2]), ("activity", 0, *params[2:])):
        # Existing Layer::assign_weight convention, ties away from zero.
        q = np.copysign(np.floor(np.abs(w / SCALE) + .5), w).clip(-127, 127)
        lines.extend((f"{name} {w.shape[1]} {w.shape[0]} {code} 1",
                      nums(b), nums(q * SCALE), " ".join(["1"] * w.size)))
    with output.open("x") as f:
        f.write("\n".join([*lines, "END", ""]))


def run(data, native, step_model):
    train_file = np.loadtxt(data / "har_train.txt")
    train_x, train_y = train_file[:, 1:].copy(), train_file[:, 0].astype(int)
    test_x = np.loadtxt(data / "har_test.txt")
    test_y = np.loadtxt(data / "har_labels.txt", dtype=int)
    assert train_x.shape == (7352, 561) and test_x.shape == (2947, 561)
    init = load_weights(native / "har_initial.nlw")
    trained = load_weights(native / "har_model.nlw")
    grad_error = finite_differences(init, train_x[0], train_y[0])
    one = tuple(a.copy() for a in init)
    step(one, train_x[0], train_y[0])
    step_error = max(max_error(a, b, 2e-12) for a, b in zip(one, load_weights(step_model)))
    reference = tuple(a.copy() for a in init)
    start = time.perf_counter()
    train(reference, train_x, train_y)
    duration = time.perf_counter() - start
    weights_error = max(max_error(a, b) for a, b in zip(trained, reference))
    host_logits = read_logits(native / "logits.log")
    ref_logits = forward(reference, test_x)
    reference_error = max_error(ref_logits, host_logits)
    actual_error = max_error(forward(trained, test_x), host_logits)
    assert np.array_equal(ref_logits.argmax(axis=1), host_logits.argmax(axis=1))
    majority = int(np.bincount(train_y).argmax())
    centroids = np.array([train_x[train_y == i].mean(axis=0) for i in range(6)])
    distances = np.sum(centroids**2, axis=1) - 2 * test_x @ centroids.T
    qmodel = native / "har_quantized.nlw"
    quantized_copy(trained, qmodel)
    qpred = forward(load_weights(qmodel), test_x).argmax(axis=1)
    np.savetxt(native / "expected_logits.txt", ref_logits, fmt="%.17g")
    result = {"declared_tolerance": TOLERANCE, "finite_difference_max_error": grad_error,
              "native_first_step_max_error": step_error, "retrained_parameter_max_error": weights_error,
              "retrained_logit_max_error": reference_error, "same_weights_logit_max_error": actual_error,
              "numpy_reference_train_seconds_excluding_io": duration,
              "neural_language": score(host_logits.argmax(axis=1), test_y),
              "numpy_same_mlp": score(ref_logits.argmax(axis=1), test_y),
              "majority": score(np.full_like(test_y, majority), test_y),
              "nearest_centroid": score(distances.argmin(axis=1), test_y),
              "fixed_int8_storage_probe": score(qpred, test_y),
              "max_abs_trained_weight": max(float(np.max(np.abs(trained[i]))) for i in (0, 2))}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "native", "step_model"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.data, args.native, args.step_model)
