#!/usr/bin/env python3
"""Offline, independent rerun for NL-COMP-002. Never modifies original sources."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def digest(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def call(args, logfile:Path, timeout=120, env=None):
    p=subprocess.run(args,cwd=ROOT,text=True,capture_output=True,timeout=timeout,env=env)
    logfile.write_text('CMD: '+' '.join(map(str,args))+'\n'+p.stdout+'\nSTDERR:\n'+p.stderr)
    if p.returncode:
        raise RuntimeError('command failed (exit '+str(p.returncode)+'): '+' '.join(map(str,args))+'; see '+str(logfile))
    return p

def openblas_path()->Path:
    try:
        import numpy
    except ImportError as e:
        raise RuntimeError('NumPy OpenBLAS is required for independent optimized baseline') from e
    lib_dir=Path(numpy.__file__).resolve().parent.parent/'numpy.libs'
    candidates=sorted(lib_dir.glob('libscipy_openblas64*.so'))
    if not candidates:
        raise RuntimeError('Cannot find SciPy OpenBLAS in '+str(lib_dir))
    return candidates[0]

def summarize(csv_path:Path):
    with csv_path.open(newline='') as stream:
        records=list(csv.DictReader(stream))
    grouped={}
    for row in records:
        key=f"network={row['network']} edits={row['edits']}"
        grouped.setdefault(key,[]).append(row)
    results={}
    for key,rows in grouped.items():
        ratio=[float(r['ratio']) for r in rows]
        results[key]={
            'median_ratio':statistics.median(ratio),
            'max_scaled_error':max(float(r['max_scaled_error']) for r in rows),
            'median_auto_us':statistics.median(float(r['auto_us']) for r in rows),
            'median_full_us':statistics.median(float(r['full_us']) for r in rows),
            'trials':len(rows),'pairs_20_percent_faster':sum(x<=0.8 for x in ratio),
            'auto_used_incremental':any(int(r['auto_delta'])>0 for r in rows),
            'auto_median_regression_gt_20pct':statistics.median(ratio)>1.20,
        }
    return results

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,help='new output directory, default temp under runs/change_input/')
    ap.add_argument('--phase',choices=['holdout','calibration','both'],default='holdout')
    args=ap.parse_args()
    if args.out:
        folder=args.out.absolute()
        folder.mkdir(parents=True,exist_ok=False)
    else:
        out_root=ROOT/'runs'/'change_input'
        out_root.mkdir(parents=True,exist_ok=True)
        folder=Path(tempfile.mkdtemp(prefix='run_',dir=out_root))
    blas=openblas_path()
    env=os.environ.copy()
    env.update({'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    metadata={'platform':platform.platform(), 'machine':platform.machine(), 'python':sys.version,
              'openblas_library':str(blas),'openblas_sha256':digest(blas),
              'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [
                  ROOT/'language/compiler.py',ROOT/'language/runtime/neural_runtime.hpp',
                  HERE/'bench_input_change.cpp',ROOT/'language/tests/change_input_stress.cpp']},
              'phase':args.phase,'result_dir':str(folder)}
    (folder/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
    commands=[]
    for compiler in ('g++','clang++'):
        if not shutil.which(compiler):raise RuntimeError(compiler+' unavailable')
        tag='gcc' if compiler=='g++' else 'clang'
        binary=folder/f'bench_{tag}'
        cmd=[compiler,'-std=c++17','-O2','-Wall','-Wextra','-Werror',
             '-DNEURAL_RUNTIME_SCIPY_BLAS','-I',str(ROOT/'language/runtime'),
             str(HERE/'bench_input_change.cpp'),str(blas),'-Wl,-rpath,'+str(blas.parent),
             '-o',str(binary)]
        call(cmd,folder/f'build_{tag}.log',timeout=90,env=env)
        phases=['holdout','calibration'] if args.phase=='both' else [args.phase]
        for phase in phases:
            argv=[str(binary)]+(['--holdout'] if phase=='holdout' else [])
            run=call(argv,folder/f'run_{tag}_{phase}.log',timeout=90,env=env)
            data=folder/f'{tag}_{phase}.csv'
            data.write_text(run.stdout)
            result=summarize(data)
            json_path=folder/f'{tag}_{phase}_summary.json'
            json_path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
            if any(v['max_scaled_error']>1e-9 for v in result.values()):
                raise RuntimeError('numerical oracle mismatch')
            commands.append({'compiler':tag,'phase':phase,
                             'scenarios':len(result),'wins_20pct':sum(v['median_ratio']<=.8 for v in result.values()),
                             'regressions_20pct':sum(v['auto_median_regression_gt_20pct'] for v in result.values()),
                             'max_scaled_error':max(v['max_scaled_error'] for v in result.values())})
    (folder/'summary.json').write_text(json.dumps({'runs':commands},indent=2)+'\n')
    print(json.dumps({'output':str(folder),'runs':commands},ensure_ascii=False))
if __name__=='__main__':main()
