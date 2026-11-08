#!/usr/bin/env python3
"""NL-COMP-001B: compare *actual* Neural Language 0.3 Model with OpenBLAS."""
from __future__ import annotations
import argparse
import csv
import glob
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'runtime_compare.cpp'
REFERENCE=ROOT/'change_aware_001.cpp'

def locate_blases():
    import numpy, scipy
    lib32=glob.glob(str(Path(scipy.__file__).parent.parent/'scipy.libs'/'libscipy_openblas-*.so'))
    lib64=glob.glob(str(Path(numpy.__file__).parent.parent/'numpy.libs'/'libscipy_openblas64_*.so'))
    if not lib32 or not lib64:raise RuntimeError('Missing optimized OpenBLAS variants (SciPy LP64 and NumPy ILP64)')
    return Path(lib32[0]).resolve(),Path(lib64[0]).resolve()

def run(args,**kw):return subprocess.run(args,check=True,text=True,capture_output=True,**kw)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',default='g++')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--reps',type=int,default=5)
    parser.add_argument('--steps',type=int,default=250)
    parser.add_argument('--warmup',type=int,default=32)
    parser.add_argument('--source-root',type=Path,default=ROOT.parents[1],
                        help='git checkout root containing language/runtime/neural_runtime.hpp')
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lib32,lib64=locate_blases()
    header=args.source_root/'language/runtime/neural_runtime.hpp'
    body=header.read_bytes()
    sha=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
    expected='ee71cdab32b248814aa23f26318130bd155c824f'
    if sha!=expected:raise RuntimeError('Neural Language actual 0.3 runtime Git blob mismatch '+sha)
    target=out/'runtime_compare'
    command=[args.compiler,'-std=c++17','-O3','-march=native','-fno-fast-math',
             '-Wall','-Wextra','-Werror','-DNEURAL_RUNTIME_SCIPY_BLAS','-DSCIPY_BLAS',
             '-I',str(args.source_root),str(SOURCE),str(lib32),str(lib64),
             '-Wl,-rpath,'+str(lib32.parent),'-Wl,-rpath,'+str(lib64.parent),'-o',str(target)]
    result=run(command)
    (out/'build.log').write_text(result.stdout+'\n'+result.stderr)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    meta={'experiment':'NL-COMP-001B','timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
          'source_blobs':{'language/runtime/neural_runtime.hpp':sha},
          'sources_sha256':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in [('runtime_compare.cpp',SOURCE),('change_aware_001.cpp',REFERENCE)]},
          'compiler_version':run([args.compiler,'--version']).stdout.splitlines()[0],
          'compile_command':command,'OpenBLAS_LP64':str(lib32),'OpenBLAS_ILP64':str(lib64),
          'thread_count':1,'cpu':next((s for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),None),
          'platform':platform.platform(),'reps':args.reps,'steps':args.steps,'warmup':args.warmup,'quick':args.quick,
          'criteria':'runtime_AUTO median <= .8*runtime_FULL, paired repeats >=4/5; relerr <=1e-9',
          'limitations':'Linux CPU; direct runtime 0.3 source; JNI/Android not tested.'}
    measured=[];rows=[]
    for size in ((0,) if args.quick else (0,1)):
      for layer in (0,1):
       for edits in ((1,64) if args.quick else (1,8,32,64)):
        name=f'net{size}_layer{layer}_edits{edits}'
        proc=subprocess.run([str(target),str(size),str(layer),str(edits),str(args.reps),str(args.steps),str(args.warmup)],env=env,text=True,capture_output=True,timeout=120)
        (out/f'{name}.stdout.log').write_text(proc.stdout)
        (out/f'{name}.stderr.log').write_text(proc.stderr)
        if proc.returncode:raise RuntimeError(f'{name} failed: {proc.stderr}')
        by={k:[] for k in ['runtime_FULL','runtime_AUTO','runtime_DELTA','external_BLAS']};correct=[]
        for line in proc.stdout.splitlines():
            cols=line.split('\t');d=dict(p.split('=',1) for p in cols[1:])
            if cols[0]=='CHECK': correct.append(d)
            elif cols[0]=='SAMPLE':
                for k in ['size','layer','edits','trial','blas_layers','full','delta']:d[k]=int(d[k])
                for k in ['mean_us','p95_us','checksum']:d[k]=float(d[k])
                by[d['method']].append(d);measured.append(d)
            else:raise RuntimeError(f'{name}: unexpected {cols[0]}')
        if len(correct)!=1 or sum(map(len,by.values()))!=4*args.reps:raise RuntimeError(f'{name}: invalid output count')
        check=correct[0]
        if float(check['max_relative'])>1e-9:raise RuntimeError(f'{name}: numeric mismatch')
        def med(mode,col):return statistics.median(x[col] for x in by[mode])
        paired=sum(by['runtime_AUTO'][k]['mean_us']<=.8*by['runtime_FULL'][k]['mean_us'] for k in range(args.reps))
        full=med('runtime_FULL','mean_us');auto=med('runtime_AUTO','mean_us')
        bl=med('external_BLAS','mean_us');delta=med('runtime_DELTA','mean_us')
        row={'case':name,'size':'128-256-32' if size==0 else '256-512-128',
             'layer':layer,'edits':edits,'max_relative_error':float(check['max_relative']),
             'max_abs_error':float(check['max_abs']),'comparisons':int(check['comparisons']),
             'runtime_full_us':full,'runtime_auto_us':auto,'runtime_delta_us':delta,'external_blas_us':bl,
             'runtime_auto_over_full':auto/full,'runtime_auto_over_external_blas':auto/bl,
             'auto_paired_wins_20pct':paired,'auto_passes_pre_registered_gate':paired>=4 and auto<=.8*full,
             'auto_median_blas_layers':med('runtime_AUTO','blas_layers'),
             'auto_median_full_passes':med('runtime_AUTO','full'),
             'auto_median_delta_passes':med('runtime_AUTO','delta')}
        rows.append(row)
        print(f"{name:25} actual FULL {full:7.3f}us AUTO {auto:7.3f}us BLAS {bl:7.3f}us "
              f"ratio AUTO/FULL {auto/full:.3f} pass {row['auto_passes_pre_registered_gate']}",flush=True)
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out/'samples.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(measured[0]));w.writeheader();w.writerows(measured)
    (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f'COMPLETE {len(rows)}: {sum(x["auto_passes_pre_registered_gate"] for x in rows)} passed actual Model::infer 20% gate')

if __name__=='__main__':main()
