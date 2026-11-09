#!/system/bin/sh
# Offline, non-destructive, fresh-directory test of three REAL Android NDK ELF64 programs.
set -eu

BASE=$(CDPATH= cd "$(dirname "$0")" && pwd)
RUN="$BASE/run_$(date +%Y%m%d_%H%M%S)_$$"
mkdir "$RUN"

for name in pretrained_cartpole integrated inference cartpole_pretrained.nlw; do
    cp "$BASE/$name" "$RUN/$name"
done
cd "$RUN"
chmod 700 pretrained_cartpole integrated inference

echo "RUN_DIR=$RUN"
./pretrained_cartpole > cartpole.stdout.log 2> cartpole.stderr.log
grep -Fq 'MODEL_LOADED cartpole_pretrained.nlw' cartpole.stdout.log
grep -Fq 'PREDICT version=1 values=-0.3736812805,0.3745496992' cartpole.stdout.log
test -s cartpole_pilot_report.json
echo "CARTPOLE_DEVICE_PASS"

./integrated > trainer.stdout.log 2> trainer.stderr.log
grep -Fq 'MODEL_SAVED trained.nlw' trainer.stdout.log
grep -Fq 'VERIFY PASS' trainer.stdout.log
grep -Fq 'PREDICT version=1600 values=0.3' trainer.stdout.log
test -s trained.nlw
test -s integrated_result.json
echo "TRAIN_DEVICE_PASS"

./inference > inference.stdout.log 2> inference.stderr.log
grep -Fq 'MODEL_LOADED trained.nlw' inference.stdout.log
grep -Fq 'PREDICT version=1 values=0.3' inference.stdout.log
test -s inference_result.json
echo "INFERENCE_DEVICE_PASS"

echo "ANDROID_DEVICE_3_PROGRAMS_PASS"
echo "LOG_DIR=$RUN"
