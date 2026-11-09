#!/usr/bin/env python3
"""Convert the existing Hwihwa-Lab CartPole PPO policy JSON to Neural Language.

This does not train, change, quantize, or invent weights. Accepts only the known
4->64(tanh)->64(tanh)->2(linear) forward path and copies all float32 numbers as
binary-exact Python floats formatted with 17 significant decimal digits.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

LAYER_SPECS = (('hidden1', 'w1', 'b1', 64, 4, 3),
               ('hidden2', 'w2', 'b2', 64, 64, 3),
               ('actions', 'w_out', 'b_out', 2, 64, 0))


def validate_policy(p: dict) -> None:
    if not isinstance(p, dict) or set(p) != {'w1','b1','w2','b2','w_out','b_out','activation'}:
        raise ValueError('Unexpected pretrained policy JSON schema')
    if p['activation'] != 'tanh':
        raise ValueError('Policy is not the expected tanh architecture')
    for name, wname, bname, outs, ins, _ in LAYER_SPECS:
        weights, biases = p[wname], p[bname]
        if not isinstance(weights,list) or len(weights)!=outs or not isinstance(biases,list) or len(biases)!=outs:
            raise ValueError(f'{name}: unexpected layer shape')
        for row in weights:
            if not isinstance(row,list) or len(row)!=ins:
                raise ValueError(f'{name}: unexpected row shape')
            for x in row:
                if type(x) not in (float,int) or not math.isfinite(x) or abs(x)>=1e9:
                    raise ValueError(f'{name}: invalid weight')
        for x in biases:
            if type(x) not in (float,int) or not math.isfinite(x) or abs(x)>=1e9:
                raise ValueError(f'{name}: invalid bias')


def to_nlw(p: dict) -> str:
    validate_policy(p)
    lines=['NLMODEL2','4 3','0']
    for name,wname,bname,outs,ins,activation in LAYER_SPECS:
        lines.append(f'{name} {ins} {outs} {activation} 0')
        lines.append(' '.join(format(v,'.17g') for v in p[bname]))
        lines.append(' '.join(format(v,'.17g') for row in p[wname] for v in row))
        lines.append(' '.join('1' for _ in range(ins*outs)))
    lines.append('END')
    return '\n'.join(lines)+'\n'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('input_json',type=Path)
    ap.add_argument('output_nlw',type=Path)
    args=ap.parse_args()
    data=args.input_json.read_bytes()
    p=json.loads(data)
    result=to_nlw(p)
    args.output_nlw.write_text(result,encoding='utf-8')
    print('EXPORTED_PARAMS 4610')
    print('SOURCE_SHA256',hashlib.sha256(data).hexdigest())
    print('OUTPUT_SHA256',hashlib.sha256(result.encode()).hexdigest())

if __name__=='__main__':main()
