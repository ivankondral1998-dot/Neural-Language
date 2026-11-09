#!/system/bin/sh
# Non-destructive device measurement: fresh process, model load + one prediction.
# NOT an isolated inference benchmark and NOT a baseline comparison.
set -eu

BASE=$(CDPATH= cd "$(dirname "$0")" && pwd)
TIME_BIN="${PREFIX:-/data/data/com.termux/files/usr}/bin/time"
if [ ! -x "$TIME_BIN" ]; then
    echo "GNU time not installed. In Termux run: pkg install time" >&2
    exit 2
fi
for file in pretrained_cartpole cartpole_pretrained.nlw; do
    if [ ! -f "$BASE/$file" ]; then
        echo "Missing $file in $BASE" >&2
        exit 2
    fi
done

RUN="$BASE/bench_cold_$(date +%Y%m%d_%H%M%S)_$$"
mkdir "$RUN"
printf 'iteration,wall_seconds,user_seconds,system_seconds,max_rss_kib\n' > "$RUN/samples.csv"
i=1
while [ "$i" -le 10 ]; do
    SAMPLE="$RUN/sample_$i"
    mkdir "$SAMPLE"
    cp "$BASE/pretrained_cartpole" "$BASE/cartpole_pretrained.nlw" "$SAMPLE/"
    chmod 700 "$SAMPLE/pretrained_cartpole"
    (
        cd "$SAMPLE"
        "$TIME_BIN" -f '%e,%U,%S,%M' -o timings.txt ./pretrained_cartpole > stdout.log 2> stderr.log
    )
    if ! grep -Fq 'PREDICT version=1 values=-0.3736812805,0.3745496992' "$SAMPLE/stdout.log"; then
        echo "FAILED iteration $i; inspect $SAMPLE" >&2
        exit 3
    fi
    printf '%s,%s\n' "$i" "$(cat "$SAMPLE/timings.txt")" >> "$RUN/samples.csv"
    i=$((i+1))
done

echo "COLD_START_10_PASS"
echo "MEASUREMENTS=$RUN/samples.csv"
cat "$RUN/samples.csv"
