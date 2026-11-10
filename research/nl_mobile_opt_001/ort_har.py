#!/usr/bin/env python3
"""FP32 ONNX Runtime comparison. Not the FP64 mobile-cpu contract."""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


def load_nlmodel2(path, width, layers):
    tokens = Path(path).read_text(encoding="utf-8").split()
    marker, ninput, count, mode = tokens[:4]
    if marker != "NLMODEL2" or int(ninput) != width or int(count) != len(layers):
        raise SystemExit("unexpected model header")
    cursor = 4
    weights = []
    biases = []
    for name, in_w, out_w, activation, storage in layers:
        got = [tokens[cursor + i] for i in range(5)]
        cursor += 5
        if got != [name, str(in_w), str(out_w), str(activation), str(storage)]:
            raise SystemExit(f"layer mismatch: {got}")
        bias = np.array(tokens[cursor:cursor + out_w], dtype=np.float64)
        cursor += out_w
        matrix = np.array(tokens[cursor:cursor + in_w * out_w], dtype=np.float64).reshape(out_w, in_w)
        cursor += in_w * out_w
        gates = np.array(tokens[cursor:cursor + in_w * out_w], dtype=np.int32)
        cursor += in_w * out_w
        if not np.all(gates == 1):
            raise SystemExit("model is not dense")
        weights.append(matrix)
        biases.append(bias)
    if tokens[cursor] != "END" or cursor + 1 != len(tokens):
        raise SystemExit("model terminator mismatch")
    return weights, biases, int(mode)


def percentile(values, index):
    ordered = sorted(values)
    return ordered[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    weights, biases, mode = load_nlmodel2(
        args.model, 561,
        [("hidden", 561, 16, 3, 0), ("activity", 16, 6, 0, 0)])
    features = []
    for line in args.features.read_text(encoding="utf-8").splitlines():
        values = np.fromstring(line, sep=" ", dtype=np.float64)
        if values.size != 561:
            raise SystemExit("bad feature row")
        features.append(values)
    labels = np.array([int(x) for x in args.labels.read_text().split()], dtype=np.int32)
    x64 = np.vstack(features)
    if len(labels) != len(x64) or len(x64) != 2947:
        raise SystemExit("expected 2947 rows")

    def fp64_logits(data):
        hidden = np.empty(16, dtype=np.float64)
        out = np.empty(6, dtype=np.float64)
        for row in range(16):
            total = biases[0][row]
            for col in range(561):
                total += weights[0][row, col] * data[col]
            hidden[row] = np.tanh(total)
        for row in range(6):
            total = biases[1][row]
            for col in range(16):
                total += weights[1][row, col] * hidden[col]
            out[row] = total
        return out

    # Reference accuracy of the explicit FP64 loop on every row. This is the
    # quality baseline for the FP32 session, not a claim that ORT is FP64.
    fp64_correct = 0
    worst_fp32 = 0.0
    fp32_correct = 0
    x32 = x64.astype(np.float32)
    w0 = weights[0].astype(np.float32)
    b0 = biases[0].astype(np.float32)
    w1 = weights[1].astype(np.float32)
    b1 = biases[1].astype(np.float32)

    nodes = []
    inits = [
        numpy_helper.from_array(w0, "W0"),
        numpy_helper.from_array(b0, "B0"),
        numpy_helper.from_array(w1, "W1"),
        numpy_helper.from_array(b1, "B1"),
    ]
    nodes.append(helper.make_node("Gemm", ["input", "W0", "B0"], ["z0"], transB=1))
    nodes.append(helper.make_node("Tanh", ["z0"], ["h"]))
    nodes.append(helper.make_node("Gemm", ["h", "W1", "B1"], ["logits"], transB=1))
    graph = helper.make_graph(
        nodes, "har",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 561])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 6])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    feed = {"input": np.zeros((1, 561), dtype=np.float32)}

    def one(row):
        feed["input"][0] = row
        return session.run(None, feed)[0][0]

    for i in range(len(x64)):
        reference = fp64_logits(x64[i])
        got = one(x32[i])
        if int(np.argmax(reference)) == int(labels[i]):
            fp64_correct += 1
        if int(np.argmax(got)) == int(labels[i]):
            fp32_correct += 1
        worst_fp32 = max(worst_fp32, float(np.max(np.abs(got.astype(np.float64) - reference))))
    for _ in range(len(x64)):
        one(x32[0])
    passes = []
    for _ in range(7):
        samples = []
        for i in range(len(x32)):
            t0 = time.perf_counter()
            one(x32[i])
            samples.append((time.perf_counter() - t0) * 1e6)
        n = len(samples)
        ordered = sorted(samples)
        passes.append((ordered[n // 2], ordered[(n - 1) * 95 // 100]))
    p50 = percentile([item[0] for item in passes], 3)
    p95 = percentile([item[1] for item in passes], 3)
    result = {
        "runtime": "onnxruntime",
        "onnxruntime_version": ort.__version__,
        "onnx_version": onnx.__version__,
        "numeric": "fp32",
        "threads": 1,
        "provider": session.get_providers(),
        "saved_mode_ignored": mode,
        "rows": int(len(x64)),
        "fp64_reference_loop_correct": fp64_correct,
        "fp32_ort_correct": fp32_correct,
        "fp64_reference_accuracy": fp64_correct / len(x64),
        "fp32_ort_accuracy": fp32_correct / len(x64),
        "max_abs_fp32_vs_explicit_fp64_loop": worst_fp32,
        "p50_us": p50,
        "p95_us": p95,
        "process_ru_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "note": "FP32 Gemm is not the FP64 row-order contract",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
