#!/bin/sh
set -eu
cd "$(dirname "$0")"
export LC_ALL=C
sha256sum -c SHA256SUMS
chmod u+x ./har_infer
log="$(mktemp ./har-run.XXXXXX)"
if command -v getprop >/dev/null 2>&1; then
    printf 'model=%s android=%s abi=%s\n' "$(getprop ro.product.model)" \
        "$(getprop ro.build.version.release)" "$(getprop ro.product.cpu.abi)"
fi
printf 'log=%s\n' "$log"
./har_infer > "$log"
awk -f verify.awk expected.txt "$log"
