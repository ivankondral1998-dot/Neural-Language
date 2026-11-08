#!/usr/bin/env python3
"""Independent, stdlib-only NLMODEL3 evaluator; does not import compiler/runtime.
The reference computes graph edges from serialized values, not C++ activations.
"""
from __future__ import annotations
import math
from pathlib import Path
import sys

def read_model(path):
    lines=iter(Path(path).read_text(encoding='utf-8').splitlines())
    if next(lines)!='NLMODEL3': raise ValueError('not a graph model')
    ni,nlayer=map(int,next(lines).split())
    mode=int(next(lines))
    if mode not in (0,1,2):raise ValueError('invalid mode')
    layers=[]
    for _ in range(nlayer):
        parts=next(lines).split()
        name=parts[0]
        width,rows,activation,storage,nparents=map(int,parts[1:6])
        parents=list(map(int,parts[6:]))
        if len(parents)!=nparents or storage not in (0,1) or activation not in (0,1,2):raise ValueError('invalid layer header')
        bias=list(map(float,next(lines).split()))
        weights=list(map(float,next(lines).split()))
        gates=list(map(int,next(lines).split()))
        if len(bias)!=rows or len(weights)!=width*rows or len(gates)!=len(weights):raise ValueError('bad weight layout')
        if any(g not in (0,1) for g in gates):raise ValueError('bad mask')
        layers.append((name,width,rows,activation,parents,bias,weights,gates))
    if next(lines)!='END':raise ValueError('missing end')
    if any(line.strip() for line in lines):raise ValueError('trailing data')
    return ni,layers

def predict(model, x):
    ninput,layers=model
    if len(x)!=ninput:raise ValueError('input shape')
    values=[]
    for name,width,rows,activation,parents,bias,weights,gates in layers:
        inputs=[]
        for index in parents:
            inputs.extend(x if index==-1 else values[index])
        if len(inputs)!=width:raise ValueError('parent shape')
        y=[]
        for r in range(rows):
            z=bias[r]+sum(weights[r*width+c]*inputs[c]*gates[r*width+c] for c in range(width))
            if activation==1:z=max(0,z)
            elif activation==2:z=1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
            y.append(z)
        values.append(y)
    return values[-1]

def main():
    model=read_model(sys.argv[1]);assert len(model[1])==3
    targets=[([0.25,0.75],[0.475]),([0.75,0.35],[0.56])]
    for x,expected in targets:
        predicted=predict(model,x)
        if any(abs(a-b)>2e-6 for a,b in zip(predicted,expected)):
            raise AssertionError(f'independent oracle mismatch {x}: {predicted} != {expected}')
        print('GRAPH_ORACLE_PASS',x,predicted)
if __name__=='__main__':main()
