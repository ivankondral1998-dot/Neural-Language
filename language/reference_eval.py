"""Independent FP64 forward oracle for NLMODEL2 and NLMODEL3.

No import of the language compiler or C++ runtime; weights are read from
native checkpoint after training. This checks forward semantics, NOT gradients.
Only local checkpoints produced in an isolated experiment should be supplied.
"""
from __future__ import annotations
import math
from pathlib import Path


class ReferenceError(ValueError):
    pass


def read_model(path):
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    if len(lines) < 5 or lines[0] not in ('NLMODEL2', 'NLMODEL3'):
        raise ReferenceError('unsupported reference checkpoint')
    graph = lines[0] == 'NLMODEL3'
    try:
        senses, layer_count = map(int, lines[1].split())
        mode = int(lines[2])
    except ValueError as exc:
        raise ReferenceError('invalid model header') from exc
    if not 1 <= senses <= 2048 or not 1 <= layer_count <= 8 or mode not in (0, 1, 2):
        raise ReferenceError('out of range model header')
    cursor = 3
    layers = []
    for i in range(layer_count):
        try:
            header = lines[cursor].split(); cursor += 1
            if len(header) < 5:
                raise ReferenceError('short layer header')
            name = header[0]
            width, rows, activation, storage = map(int, header[1:5])
            if graph:
                number_of_parents = int(header[5]); parents = [int(x) for x in header[6:]]
                if len(parents) != number_of_parents or not 1 <= number_of_parents <= 8:
                    raise ReferenceError('invalid model graph parents')
            else:
                if len(header) != 5: raise ReferenceError('unexpected sequential topology')
                parents = [i - 1]
            if not 1 <= width <= 32768 or not 1 <= rows <= 4096 or storage not in (0, 1) or activation not in (0, 1, 2):
                raise ReferenceError('invalid layer dimensions or kind')
            if any(parent >= i or parent < -1 for parent in parents):
                raise ReferenceError('model parent is not topologically earlier')
            expected_width = sum(senses if p == -1 else layers[p]['rows'] for p in parents)
            if width != expected_width: raise ReferenceError('graph input shape mismatch')
            bias = [float(x) for x in lines[cursor].split()]; cursor += 1
            weights = [float(x) for x in lines[cursor].split()]; cursor += 1
            gates = [int(x) for x in lines[cursor].split()]; cursor += 1
            if len(bias) != rows or len(weights) != rows * width or len(gates) != rows * width:
                raise ReferenceError('model payload shape mismatch')
            if any(g not in (0,1) for g in gates) or any(not math.isfinite(x) for x in bias + weights):
                raise ReferenceError('non-finite model parameters or invalid gates')
            layers.append({'name':name,'width':width,'rows':rows,'activation':activation,'storage':storage,
                           'parents':parents,'bias':bias,'weights':weights,'gates':gates})
        except (IndexError, ValueError) as exc:
            raise ReferenceError(f'invalid model data for layer {i}') from exc
    if cursor != len(lines)-1 or lines[cursor] != 'END':
        raise ReferenceError('trailing data or invalid model terminator')
    return senses, layers


def predict(model, data):
    senses, layers = model
    if len(data) != senses or any(not math.isfinite(v) for v in data):
        raise ReferenceError('reference input shape/finite mismatch')
    activations = []
    for layer in layers:
        inp = []
        for parent in layer['parents']:
            inp.extend(data if parent == -1 else activations[parent])
        outputs = []
        for r in range(layer['rows']):
            offset = r*layer['width']
            z = layer['bias'][r]
            for k, value in enumerate(inp):
                z += layer['weights'][offset+k] * layer['gates'][offset+k] * value
            if layer['activation'] == 1:
                z = max(0.0, z)
            elif layer['activation'] == 2:
                z = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))
            if not math.isfinite(z):raise ReferenceError('non-finite reference output')
            outputs.append(z)
        activations.append(outputs)
    return activations[-1]
