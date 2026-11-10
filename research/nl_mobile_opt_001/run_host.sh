#!/bin/sh
# Host acceptance run. Does not claim a phone result.
set -eu
cd "$(dirname "$0")"
repo=$(cd ../.. && pwd)
src="$repo/research/nl_mobile_opt_001"
runtime="$repo/language/runtime"
data_dir=${NL_DATA_DIR:?set NL_DATA_DIR to the prepared UCI test directory}
work=$(mktemp -d)
logdir="$src/evidence"
mkdir -p "$logdir"
echo "work=$work"
{
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "uname=$(uname -a)"
  echo "g++=$(g++ --version | head -1)"
  echo "cpu=$(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //')"
  if [ -d /sys/class/powercap ]; then echo "powercap=present"; else echo "powercap=ABSENT energy=NOT_MEASURED"; fi
  echo "protocol_sha=$(sha256sum "$src/protocol.txt" | awk '{print $1}')"
} | tee "$logdir/host_environment.txt"

ln "$repo/research/nl_train_001/models/har_model.nlw" "$work/har_model.nlw"
ln "$repo/research/agent_pilot/models/cartpole_pretrained.nlw" "$work/cartpole_pretrained.nlw"
ln "$data_dir/har_test.txt" "$work/har_test.txt"
ln "$data_dir/har_labels.txt" "$work/har_labels.txt"
cp "$src/har_infer.nir" "$src/cartpole_infer.nir" "$work/"

g++ -std=c++17 -O2 -ffp-contract=off -DNEURAL_MOBILE_PLAN -Wall -Wextra -Werror -Wno-mismatched-new-delete \
  -I "$runtime" "$src/bench_host.cpp" -ldl -o "$work/bench_host"
blas=/usr/local/lib/python3.10/site-packages/numpy.libs/libscipy_openblas64_-56d6093b.so
export NL_OPENBLAS="$blas"
export OPENBLAS_NUM_THREADS=1

(
  cd "$work"
  taskset -c 0 ./bench_host --task har > "$logdir/har_bench.log"
  taskset -c 0 ./bench_host --task cartpole > "$logdir/cartpole_bench.log"
)
# RSS and load time are the language programs, without the OpenBLAS library mapped.
(
  cd "$work"
  unset NL_OPENBLAS
  taskset -c 0 ./bench_host --task har --only generic > "$logdir/har_rss_generic.log"
  taskset -c 0 ./bench_host --task har --only mobile > "$logdir/har_rss_mobile.log"
  taskset -c 0 ./bench_host --task har --load > "$logdir/har_load.log"
  taskset -c 0 ./bench_host --task cartpole --only generic > "$logdir/cartpole_rss_generic.log"
  taskset -c 0 ./bench_host --task cartpole --only mobile > "$logdir/cartpole_rss_mobile.log"
)

compile_one() {
  name=$1
  profile=$2
  nir=$3
  dir="$work/$name"
  mkdir -p "$dir"
  cp "$work/$nir" "$dir/$nir"
  if [ "$nir" = "har_infer.nir" ]; then
    ln "$work/har_model.nlw" "$dir/har_model.nlw"
    ln "$work/har_test.txt" "$dir/har_test.txt"
  else
    ln "$work/cartpole_pretrained.nlw" "$dir/cartpole_pretrained.nlw"
  fi
  extra=""
  if [ "$profile" = "mobile-cpu" ]; then extra="--profile mobile-cpu"; fi
  # shellcheck disable=SC2086
  python3 "$repo/language/compiler.py" "$dir/$nir" $extra -o "$dir/program"
  (cd "$dir" && ./program > "$logdir/${name}_stdout.txt")
}

compile_one har_generic generic har_infer.nir
compile_one har_mobile mobile-cpu har_infer.nir
compile_one cartpole_generic generic cartpole_infer.nir
compile_one cartpole_mobile mobile-cpu cartpole_infer.nir

python3 - << PY
import json, pathlib, hashlib
log = pathlib.Path("$logdir")
work = pathlib.Path("$work")

def lines(path, prefix):
    return [line for line in path.read_text().splitlines() if line.startswith(prefix)]

def impl(path):
    out = {}
    for line in path.read_text().splitlines():
        if not line.startswith("impl="):
            continue
        item = {}
        for part in line.split():
            key, value = part.split("=", 1)
            item[key] = value
        out[item["impl"]] = item
    return out

def rss(path):
    for line in path.read_text().splitlines():
        if line.startswith("rss_kb="):
            return int(line.split("=", 1)[1])
    raise SystemExit("missing rss in " + str(path))

har = impl(log / "har_bench.log")
cart = impl(log / "cartpole_bench.log")
g = float(har["nl_generic_observe_infer"]["p50_us"])
m = float(har["nl_mobile_observe_infer"]["p50_us"])
cg = float(cart["nl_generic_observe_infer"]["p50_us"])
cm = float(cart["nl_mobile_observe_infer"]["p50_us"])
rg = rss(log / "har_rss_generic.log")
rm = rss(log / "har_rss_mobile.log")
crg = rss(log / "cartpole_rss_generic.log")
crm = rss(log / "cartpole_rss_mobile.log")
mobile_report = json.loads((work/"har_mobile/har_report.json").read_text())
generic_report = json.loads((work/"har_generic/har_report.json").read_text())
summary = {
    "har_speedup_p50": g / m,
    "har_generic_p50_us": g,
    "har_mobile_p50_us": m,
    "har_generic_p95_us": float(har["nl_generic_observe_infer"]["p95_us"]),
    "har_mobile_p95_us": float(har["nl_mobile_observe_infer"]["p95_us"]),
    "har_speed_gate_1_25": g / m >= 1.25,
    "har_rss_generic_kb": rg,
    "har_rss_mobile_kb": rm,
    "har_rss_reduction": (rg - rm) / rg,
    "har_rss_gate_30pct": (rg - rm) / rg >= 0.30 and m <= g * 1.10,
    "har_mobile_max_abs": float(har["nl_mobile_observe_infer"]["max_abs_vs_generic"]),
    "har_mobile_correct": int(har["nl_mobile_observe_infer"]["correct"]),
    "har_generic_correct": int(har["nl_generic_observe_infer"]["correct"]),
    "har_accuracy": int(har["nl_mobile_observe_infer"]["correct"]) / 2947,
    "har_scalar_p50_us": float(har["cpp_scalar_kernel"]["p50_us"]),
    "har_ilp_p50_us": float(har["cpp_ilp_kernel"]["p50_us"]),
    "har_blas_p50_us": float(har["openblas_dgemv_kernel"]["p50_us"]),
    "har_blas_max_abs": float(har["openblas_dgemv_kernel"]["max_abs_vs_generic"]),
    "har_blas_correct": int(har["openblas_dgemv_kernel"]["correct"]),
    "har_mobile_allocs_per_sample": float(har["nl_mobile_observe_infer"]["allocs_per_sample"]),
    "har_generic_allocs_per_sample": float(har["nl_generic_observe_infer"]["allocs_per_sample"]),
    "har_plan_calls": int(har["nl_mobile_observe_infer"]["plan_calls"]),
    "cartpole_speedup_p50": cg / cm,
    "cartpole_generic_p50_us": cg,
    "cartpole_mobile_p50_us": cm,
    "cartpole_speed_gate_1_25": cg / cm >= 1.25,
    "cartpole_actions": int(cart["nl_mobile_observe_infer"]["correct"]),
    "cartpole_max_abs": float(cart["nl_mobile_observe_infer"]["max_abs_vs_generic"]),
    "cartpole_scalar_p50_us": float(cart["cpp_scalar_kernel"]["p50_us"]),
    "cartpole_ilp_p50_us": float(cart["cpp_ilp_kernel"]["p50_us"]),
    "cartpole_blas_p50_us": float(cart["openblas_dgemv_kernel"]["p50_us"]),
    "cartpole_rss_generic_kb": crg,
    "cartpole_rss_mobile_kb": crm,
    "cartpole_plan_calls": int(cart["nl_mobile_observe_infer"]["plan_calls"]),
    "har_predict_lines_identical": lines(log/"har_generic_stdout.txt", "DATA_PREDICT") == lines(log/"har_mobile_stdout.txt", "DATA_PREDICT"),
    "cartpole_predict_lines_identical": lines(log/"cartpole_generic_stdout.txt", "PREDICT ") == lines(log/"cartpole_mobile_stdout.txt", "PREDICT "),
    "har_mobile_report_plan": mobile_report["counts"].get("mobile_plan"),
    "har_generic_report_has_plan": "mobile_plan" in generic_report["counts"],
    "har_generic_bytes": (work/"har_generic/program").stat().st_size,
    "har_mobile_bytes": (work/"har_mobile/program").stat().st_size,
    "cartpole_generic_bytes": (work/"cartpole_generic/program").stat().st_size,
    "cartpole_mobile_bytes": (work/"cartpole_mobile/program").stat().st_size,
    "har_model_bytes": (work/"har_model.nlw").stat().st_size,
    "cartpole_model_bytes": (work/"cartpole_pretrained.nlw").stat().st_size,
    "generated_har_cpp_sha256": hashlib.sha256((work/"har_mobile/program.cpp").read_bytes()).hexdigest(),
}
(log/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
cp "$work/har_generic/har_report.json" "$logdir/har_generic_report.json"
cp "$work/har_mobile/har_report.json" "$logdir/har_mobile_report.json"
cp "$work/cartpole_generic/cartpole_report.json" "$logdir/cartpole_generic_report.json"
cp "$work/cartpole_mobile/cartpole_report.json" "$logdir/cartpole_mobile_report.json"
cp "$work/har_mobile/program.cpp" "$logdir/har_mobile_generated.cpp"
echo "logs=$logdir"

