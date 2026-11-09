#!/system/bin/sh
# One command on a Termux Android arm64 device; no compilers or dependencies.
set -eu
BASE=$(CDPATH= cd "$(dirname "$0")" && pwd)
RUN="$BASE/bench_run_$(date +%Y%m%d_%H%M%S)_$$"
mkdir "$RUN"
cd "$BASE"
if [ -f "SHA256SUMS" ]; then
    sha256sum -c SHA256SUMS > "$RUN/integrity.log" || {
        cat "$RUN/integrity.log"; exit 2;
    }
fi
if command -v getprop >/dev/null 2>&1; then
    {
        getprop ro.product.model
        getprop ro.build.version.release
        getprop ro.product.cpu.abi
    } > "$RUN/device.txt"
fi
chmod 700 "./bench_cartpole_compare"
./bench_cartpole_compare cartpole_pretrained.nlw 4096 7 \
    > "$RUN/results.log" 2> "$RUN/errors.log" || {
    cat "$RUN/errors.log"
    echo "BENCH_FAILED; logs: $RUN"
    exit 3
}
grep -Fq "REFERENCE_EQUIVALENCE_PASS" "$RUN/results.log"
grep -Fq "BENCH_PASS" "$RUN/results.log"
grep "REFERENCE_EQUIVALENCE_PASS\|^TRIAL \|^BENCH_PASS" "$RUN/results.log"
echo "LOG_DIR=$RUN"
