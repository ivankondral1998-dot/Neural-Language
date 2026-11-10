#!/usr/bin/env python3
"""Package frozen baseline/current sources for same-device native testing."""
import argparse, hashlib, json, shutil, subprocess, sys, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = '32fc5be2a44a2b08dcf875880c7b1f7e7ce42f5a'
p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--ndk', type=Path, help='optional verified Android NDK root for API26 ARM64 binaries')
a = p.parse_args()
d = a.output.resolve()
d.mkdir(parents=True, exist_ok=False)
for variant in ('before', 'after'):
    (d / variant).mkdir()
    for name in ('neural_runtime.hpp', 'mobile_plan.hpp'):
        f = 'language/runtime/' + name
        data = (subprocess.check_output(['git', 'show', BASE + ':' + f], cwd=ROOT)
                if variant == 'before' else (ROOT / f).read_bytes())
        (d / variant / name).write_bytes(data)
for src, name in [
    ('research/android_ndk/bench_cartpole_compare.cpp', 'bench_cartpole_compare.cpp'),
    ('research/agent_pilot/models/cartpole_pretrained.nlw', 'cartpole_pretrained.nlw'),
    ('research/mobile_kernel_003/bench_int8.cpp', 'bench_int8.cpp'),
    ('research/mobile_kernel_003/run_compare.py', 'run_compare.py')]:
    shutil.copyfile(ROOT / src, d / name)
shutil.copyfile(ROOT / 'language/examples/mobile_int8_plan.nir', d / 'mobile_int8_plan.nir')
subprocess.run([sys.executable, str(ROOT / 'language/compiler.py'),
                str(d / 'mobile_int8_plan.nir'), '-o', str(d / 'packed_demo'),
                '--profile', 'mobile-cpu', '--emit-only'], check=True)
if a.ndk:
    cxx = a.ndk.resolve() / 'toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang++'
    binaries = d / 'binaries'
    binaries.mkdir()
    flags = ['-std=c++17', '-O3', '-ffp-contract=off', '-Wall', '-Wextra', '-Werror', '-static-libstdc++', '-fPIE', '-pie']
    for variant in ('before', 'after'):
        subprocess.run([str(cxx), *flags, '-I', str(d / variant),
                        str(d / 'bench_cartpole_compare.cpp'), '-o', str(binaries / variant)], check=True)
    subprocess.run([str(cxx), *flags, '-I', str(d / 'after'),
                    str(d / 'bench_int8.cpp'), '-o', str(binaries / 'int8')], check=True)
    subprocess.run([str(cxx), *flags, '-DNEURAL_MOBILE_PLAN', '-I', str(d / 'after'),
                    str(d / 'packed_demo.cpp'), '-o', str(binaries / 'packed-demo')], check=True)
    (binaries / 'build.json').write_text(json.dumps({
        'compiler': subprocess.check_output([str(cxx), '--version'], text=True).splitlines()[0],
        'flags': flags, 'target': 'Android API26 AArch64', 'device_executed': False,
        'sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                   for f in binaries.iterdir() if f.is_file()}}, indent=2) + '\n')
manifest = {'baseline': BASE, 'files': {
    str(f.relative_to(d)): hashlib.sha256(f.read_bytes()).hexdigest()
    for f in d.rglob('*') if f.is_file()}}
(d / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
(d / 'README.txt').write_text(
    'Same-device comparison; no weights updated. Requires Python3 and C++17 compiler.\n'
    'Run: python run_compare.py --compiler clang++ --output results_NEW\n'
    'If binaries/ exists: python run_compare.py --prebuilt --output results_NEW\n'
    'Run on ARM64 Android from a private executable directory (e.g. Termux home), not shared Download storage.\n'
    'For packed INT8 comparison run ./binaries/int8 2048 9 (when bundled).\n'
    'Actual compiled .nir mixed-storage/fallback example: ./binaries/packed-demo\n'
    'Output directory must be new. Keep environment.json and all paired logs.\n'
    'INT8: clang++ -std=c++17 -O3 -ffp-contract=off '
    '-Iafter bench_int8.cpp -o bench_int8\n./bench_int8\n'
    'No phone results or universal language speed advantage are asserted.\n')
archive = d.with_suffix('.tar.gz')
with tarfile.open(archive, 'x:gz') as t:
    t.add(d, arcname=d.name)
print(archive)
