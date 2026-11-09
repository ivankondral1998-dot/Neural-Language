#!/system/bin/sh
# Controlled same-device OLD-vs-DENSE Neural Language and plain C++ comparison.
# No training, networking, overwrites or deletions. Three A/B pairs.
set -eu
BASE=$(CDPATH= cd "$(dirname "$0")" && pwd)
RUN="$BASE/dense_ab_$(date +%Y%m%d_%H%M%S)_$$"
mkdir "$RUN"
cd "$BASE"
sha256sum -c SHA256SUMS > "$RUN/input_sha256.log" || {
    cat "$RUN/input_sha256.log"; exit 2;
}
{
    getprop ro.product.model || true
    getprop ro.build.version.release || true
    getprop ro.product.cpu.abi || true
} > "$RUN/device.txt"
chmod 700 bench_old bench_dense
for pair in 1 2 3; do
    if [ "$pair" -eq 2 ]; then
        order="dense old"
    else
        order="old dense"
    fi
    for variant in $order; do
        log="$RUN/pair_${pair}_${variant}.log"
        "./bench_$variant" cartpole_pretrained.nlw 4096 7 > "$log" 2> "$RUN/pair_${pair}_${variant}.err" || {
            cat "$RUN/pair_${pair}_${variant}.err"
            echo "BENCH_FAILED pair=$pair variant=$variant log=$log"
            exit 3
        }
        grep -Fq 'REFERENCE_EQUIVALENCE_PASS vectors=517 max_abs_error=0' "$log"
        grep -Fq 'BENCH_PASS samples=4096 trials=7' "$log"
        echo "DEVICE_SAMPLE pair=$pair variant=$variant"
        grep '^BENCH_PASS ' "$log"
    done
done
echo "DENSE_AB_ANDROID_PASS"
echo "LOG_DIR=$RUN"
