#!/usr/bin/env python3
"""Independent (stdlib-only) mathematical check of the saved simple linear model.
This checks one explicit held-out example family; not a general proof of backprop.
"""
import argparse
import math
from pathlib import Path


def run(model_path: Path) -> None:
    tokens=model_path.read_text(encoding='utf-8').split()
    assert tokens[0]=='NLMODEL2', 'unsupported format'
    assert tokens[1:3]==['2','1'], 'expected two inputs and one layer'
    assert tokens[3] in ('0','1'), 'unexpected optimized execution policy'
    assert tokens[4:9]==['answer','2','1','0','0'], 'unexpected layer description'
    bias=float(tokens[9]); w1=float(tokens[10]); w2=float(tokens[11]); a,b=map(int,tokens[12:14])
    assert tokens[14]=='END' and len(tokens)==15
    assert a==1 and b==1
    tests=[((0.25,0.75),0.3),((0.75,0.35),0.52)]
    for inputs,target in tests:
        y=bias+w1*inputs[0]+w2*inputs[1]
        assert math.isfinite(y) and abs(y-target)<=0.02,(inputs,y,target)
        print(f'ORACLE PASS input={inputs} expected={target:.6f} got={y:.12f}')
    print('ORACLE PASS independently computed from saved weights, no C++ runtime')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',type=Path);args=p.parse_args()
    run(args.model)
