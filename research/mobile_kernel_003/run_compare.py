#!/usr/bin/env python3
"""Run paired native benchmarks from the portable bundle; host is not Android."""
import argparse, hashlib, json, os, platform, re, statistics, subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--compiler', default='c++')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--pairs', type=int, default=5)
p.add_argument('--prebuilt', action='store_true', help='use bundled Android ARM64 executables')
a = p.parse_args()
if not 3 <= a.pairs <= 15:
    p.error('--pairs must be 3..15')
root = Path(__file__).resolve().parent
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
flags = ['-std=c++17', '-O3', '-ffp-contract=off', '-Wall', '-Wextra', '-Werror']
if a.prebuilt and platform.machine().lower() not in ('aarch64', 'arm64'):
    p.error('prebuilt executables require Android ARM64; host results are separate')
meta = {'platform': platform.platform(), 'machine': platform.machine(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'compiler': ('bundled Android NDK r30/API26; see build.json' if a.prebuilt else
                     subprocess.check_output([a.compiler, '--version'], text=True).splitlines()[0]),
        'flags': flags, 'pairs': a.pairs,
        'sources': {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in root.rglob('*') if f.is_file() and f.suffix in ('.hpp', '.cpp', '.nlw')}}
(out / 'environment.json').write_text(json.dumps(meta, indent=2) + '\n')
executables = {}
for variant in ('before', 'after'):
    if a.prebuilt:
        exe = root / 'binaries' / variant
        build = json.loads((root / 'binaries' / 'build.json').read_text())
        if hashlib.sha256(exe.read_bytes()).hexdigest() != build['sha256'][variant]:
            raise ValueError('bundled executable checksum mismatch')
    else:
        exe = out / variant
        subprocess.run([a.compiler, *flags, '-I', str(root / variant),
                        str(root / 'bench_cartpole_compare.cpp'), '-o', str(exe)], check=True)
    executables[variant] = exe
meta['executables_sha256'] = {k: hashlib.sha256(v.read_bytes()).hexdigest()
                              for k, v in executables.items()}
(out / 'environment.json').write_text(json.dumps(meta, indent=2) + '\n')
results = []
for pair in range(a.pairs):
    for variant in (('before', 'after') if pair % 2 == 0 else ('after', 'before')):
        # Model loader deliberately permits local filenames only.
        r = subprocess.run([str(executables[variant]), 'cartpole_pretrained.nlw', '4096', '7'],
                           cwd=root, text=True, capture_output=True)
        (out / f'pair_{pair}_{variant}.txt').write_text(r.stdout + r.stderr)
        r.check_returncode()
        last = next(l for l in r.stdout.splitlines() if l.startswith('BENCH_PASS '))
        fields = dict(re.findall(r'(\w+)=([^ ]+)', last))
        results.append({'pair': pair, 'variant': variant,
                        'neural_us': float(fields['neural_median_us_per_infer']),
                        'cpp_us': float(fields['plain_cpp_median_us_per_infer']),
                        'max_abs_error': float(fields['max_abs_error'])})
summary = {'samples': results,
           'before_us': statistics.median(x['neural_us'] for x in results if x['variant'] == 'before'),
           'after_us': statistics.median(x['neural_us'] for x in results if x['variant'] == 'after'),
           'cpp_us': statistics.median(x['cpp_us'] for x in results),
           'interpretation': 'Measured on this host only; no universal C++ or Android advantage.'}
summary['before_over_after'] = summary['before_us'] / summary['after_us']
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
