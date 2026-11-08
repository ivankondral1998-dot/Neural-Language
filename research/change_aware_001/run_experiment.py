#!/usr/bin/env python3
"""NL-COMP-001: run and preserve pre-registered OpenBLAS comparison."""
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

HERE=Path(__file__).resolve().parent
SRC=HERE/'change_aware_001.cpp'

def cmd(args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True, **kwargs)

def discover_openblas():
    import scipy
    pattern=str(Path(scipy.__file__).parent.parent/'scipy.libs'/'libscipy_openblas-*.so')
    found=glob.glob(pattern)
    if not found:
        raise RuntimeError('optimized SciPy/OpenBLAS shared library not available: '+pattern)
    return Path(found[0]).resolve()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler',default='g++')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reps',type=int,default=5)
    p.add_argument('--steps',type=int,default=250)
    p.add_argument('--warmup',type=int,default=32)
    p.add_argument('--quick',action='store_true',help='smoke testing; not the preregistered result')
    args=p.parse_args()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    lib=discover_openblas()
    binary=output/'change_aware_001'
    flags=['-std=c++17','-O3','-march=native','-fno-fast-math',
           '-Wall','-Wextra','-Werror','-DSCIPY_BLAS']
    command=[args.compiler,*flags,str(SRC),str(lib),f'-Wl,-rpath,{lib.parent}','-o',str(binary)]
    build=cmd(command)
    (output/'build.log').write_text(build.stdout+'\n'+build.stderr)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    try: cpu=Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0].strip(' :\t')
    except Exception: cpu='unknown'
    metadata={
      'experiment':'NL-COMP-001','run_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
      'status':'experimental kernel, NOT production Model::infer',
      'compiler_command':command,'compiler_version':cmd([args.compiler,'--version']).stdout.splitlines()[0],
      'openblas':str(lib),'openblas_sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),
      'openblas_threads':1,'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),
      'platform':platform.platform(),'architecture':platform.machine(),'cpu_model':cpu,
      'python':sys.version,'cases':[],'reps':args.reps,'steps':args.steps,'warmup':args.warmup,
      'acceptance':'median change_aware <= 0.80*median blas_full with 4/5 paired trials, relerr <=1e-9',
      'quick':args.quick,
    }
    sizes=(0,1) if not args.quick else (0,)
    batches=(1,8,32,64) if not args.quick else (1,64)
    all_samples=[];results=[]
    for size in sizes:
      for layer in (0,1):
       for edits in batches:
        case=f'net{size}_layer{layer}_edits{edits}'
        command=[str(binary),str(size),str(layer),str(edits),str(args.reps),str(args.steps),str(args.warmup)]
        process=subprocess.run(command,env=env,capture_output=True,text=True,timeout=120)
        (output/f'{case}.stdout.log').write_text(process.stdout)
        (output/f'{case}.stderr.log').write_text(process.stderr)
        if process.returncode:
            raise RuntimeError(f'{case} FAILED code {process.returncode}: '+process.stderr)
        measured={x:[] for x in ('scalar_full','blas_full','change_aware')}
        checks=[]
        for line in process.stdout.splitlines():
            cells=line.split('\t')
            if cells[0] not in ('SAMPLE','CHECK'):
                raise RuntimeError('unexpected output '+line)
            row=dict(x.split('=',1) for x in cells[1:])
            if cells[0]=='CHECK':checks.append(row)
            else:
                row.update(size=int(row['size']),layer=int(row['layer']),edits=int(row['edits']),trial=int(row['trial']),mean_us=float(row['mean_us']),p95_us=float(row['p95_us']),checksum=float(row['checksum']))
                measured[row['method']].append(row)
                all_samples.append(row)
        if len(checks)!=1 or sum(len(v) for v in measured.values())!=3*args.reps:
            raise RuntimeError('missing results: '+case)
        if float(checks[0]['max_relative'])>1e-9:
            raise RuntimeError('unacceptable numeric drift: '+case)
        def med(method,col):return statistics.median(x[col] for x in measured[method])
        paired=sum(measured['change_aware'][i]['mean_us']<=.8*measured['blas_full'][i]['mean_us'] for i in range(args.reps))
        result={
            'case':case,'shape':'128-256-32' if size==0 else '256-512-128',
            'layer':layer,'edits':edits,
            'max_relative_error':float(checks[0]['max_relative']),
            'max_abs_error':float(checks[0]['max_abs']),
            'comparisons':int(checks[0]['comparisons']),
            'scalar_us':med('scalar_full','mean_us'),
            'blas_us':med('blas_full','mean_us'),
            'delta_us':med('change_aware','mean_us'),
            'scalar_p95_us':med('scalar_full','p95_us'),
            'blas_p95_us':med('blas_full','p95_us'),
            'delta_p95_us':med('change_aware','p95_us'),
            'ratio_delta_to_blas':med('change_aware','mean_us')/med('blas_full','mean_us'),
            'paired_improvements_20pct':paired,
            'passes_20pct_gate':paired>=4 and med('change_aware','mean_us')<=.8*med('blas_full','mean_us'),
            'trials':args.reps,
        }
        results.append(result)
        metadata['cases'].append(case)
        print(f"{case:25} OpenBLAS {result['blas_us']:8.3f}us / changed {result['delta_us']:8.3f}us "
              f"ratio={result['ratio_delta_to_blas']:.3f} pass={result['passes_20pct_gate']}",flush=True)
    with (output/'samples.csv').open('w',newline='') as handle:
        wr=csv.DictWriter(handle,fieldnames=['size','layer','edits','trial','method','mean_us','p95_us','checksum']);wr.writeheader();wr.writerows(all_samples)
    with (output/'results.csv').open('w',newline='') as handle:
        wr=csv.DictWriter(handle,fieldnames=list(results[0]));wr.writeheader();wr.writerows(results)
    (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'COMPLETE {len(results)} scenarios; {sum(x["passes_20pct_gate"] for x in results)} meet preregistered win criterion')

if __name__=='__main__':main()
