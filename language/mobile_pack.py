#!/usr/bin/env python3
"""Lossless offline conversion: trusted Neural Language NLMODEL2 -> NLMOBIN1.

The output stores binary64 floating weights, native INT8 codes and packed edge
masks, preserving inference semantics exactly. This tool runs on the host only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import struct
import zlib

MAGIC = b'NLMOBIN1'
NAME = re.compile(r'[A-Za-z_][A-Za-z_0-9]*\Z')
MAX_WEIGHTS = 4_000_000
MAX_SOURCE_BYTES = 256 * 1024 * 1024


class PackingError(ValueError):
    pass


def fail(message: str):
    raise PackingError(message)


def int_tokens(text: str, expected: int, context: str):
    parts = text.split()
    if len(parts) != expected:
        fail(f'{context}: expected {expected} integers')
    try:
        if any(not re.fullmatch(r'[+-]?\d+', token) for token in parts):
            fail(f'{context}: invalid integer')
        return [int(token) for token in parts]
    except ValueError as exc:
        raise PackingError(f'{context}: invalid integer') from exc


def real_tokens(text: str, expected: int, context: str):
    parts = text.split()
    if len(parts) != expected:
        fail(f'{context}: expected {expected} numbers')
    try:
        result = [float(token) for token in parts]
    except ValueError as exc:
        raise PackingError(f'{context}: invalid number') from exc
    if any(not math.isfinite(v) or abs(v) >= 1e9 for v in result):
        fail(f'{context}: number outside supported range')
    return result


def serialize(source: Path) -> tuple[bytes, dict]:
    if source.stat().st_size > MAX_SOURCE_BYTES:
        fail('source checkpoint is too large')
    lines = source.read_text(encoding='utf-8').splitlines()
    if len(lines) < 4 or lines[0] != 'NLMODEL2':
        fail('only sequential NLMODEL2 is supported')
    inputs, layer_count = int_tokens(lines[1], 2, 'header')
    (mode,) = int_tokens(lines[2], 1, 'mode')
    if not 1 <= inputs <= 2048 or not 1 <= layer_count <= 8 or mode not in (0, 1, 2):
        fail('invalid network bounds')
    cursor = 3
    previous = inputs
    total_weights = 0
    payload = bytearray()
    int8_count = 0
    fp64_count = 0
    names = set()
    for k in range(layer_count):
        if cursor + 4 > len(lines):
            fail(f'layer {k}: incomplete checkpoint')
        header = lines[cursor].split()
        if len(header) != 5 or NAME.fullmatch(header[0]) is None or header[0] in names:
            fail(f'layer {k}: invalid name or header')
        names.add(header[0])
        in_size, out_size, activation, storage = int_tokens(' '.join(header[1:]), 4, f'layer {k}')
        if in_size != previous or not 1 <= out_size <= 4096 or activation not in (0,1,2) or storage not in (0,1):
            fail(f'layer {k}: invalid shape or operator')
        count = in_size * out_size
        total_weights += count
        if total_weights > MAX_WEIGHTS:
            fail('model exceeds 4 million weights')
        biases = real_tokens(lines[cursor+1], out_size, f'layer {k} bias')
        weights = real_tokens(lines[cursor+2], count, f'layer {k} weights')
        gates = int_tokens(lines[cursor+3], count, f'layer {k} masks')
        if any(g not in (0,1) for g in gates):
            fail(f'layer {k}: invalid mask bit')
        payload.extend(struct.pack('<IIII',in_size,out_size,activation,storage))
        payload.extend(struct.pack('<'+'d'*out_size,*biases))
        if storage:
            q=[]
            for w in weights:
                code = round(w*127.0)
                if not -127 <= code <= 127 or abs(code / 127.0 - w) > 1e-14:
                    fail(f'layer {k}: nonrepresentable INT8 weight')
                q.append(code)
            payload.extend(struct.pack('<'+'b'*count,*q))
            int8_count += count
        else:
            payload.extend(struct.pack('<'+'d'*count,*weights))
            fp64_count += count
        packed = bytearray((count + 7)//8)
        for j, g in enumerate(gates):
            if g: packed[j//8] |= 1<<(j%8)
        payload.extend(packed)
        previous = out_size
        cursor += 4
    if cursor != len(lines)-1 or lines[cursor] != 'END':
        fail('unexpected trailing content or missing END')
    checksum = zlib.crc32(payload)
    header = struct.pack('<8sIIIIQII',MAGIC,1,inputs,layer_count,total_weights,len(payload),checksum,0)
    result = bytes(header) + payload
    assert len(header)==40
    return result, {'inputs': inputs, 'outputs':previous, 'layers':layer_count,
                    'weights':total_weights, 'int8_weights':int8_count,
                    'fp64_weights':fp64_count, 'source_bytes':source.stat().st_size,
                    'mobile_bytes':len(result), 'crc32':f'{checksum:08x}'}


def convert(source: Path, output: Path) -> dict:
    # Never overwrite an existing destination (including symlinks), or an alias of the source.
    if output.exists() or output.is_symlink() or source.resolve() == output.resolve():
        fail('destination already exists or aliases source')
    result, meta = serialize(source)
    # O_EXCL guarantees the output is new even under concurrent writers.
    fd = os.open(output, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as file:
            file.write(result)
            file.flush()
            os.fsync(file.fileno())
    except BaseException:
        # Never delete a file implicitly. A partial new output may remain after an I/O failure.
        raise
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source',type=Path,help='trusted NLMODEL2 checkpoint')
    ap.add_argument('-o','--output',type=Path,required=True,help='new NLMOBIN1 file; no overwrite')
    ns=ap.parse_args(argv)
    try:
        print(json.dumps(convert(ns.source,ns.output),indent=2))
        return 0
    except (OSError,UnicodeError,PackingError,struct.error) as exc:
        ap.exit(2,f'NEURAL PACK ERROR: {exc}\n')


if __name__=='__main__':
    raise SystemExit(main())
