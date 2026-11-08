#!/usr/bin/env python3
"""Neural Language 0.2: source -> validated Neural IR -> C++17 -> native binary.
This is a bootstrap compiler, not the language runtime. No eval or arbitrary code execution.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unicodedata

IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
VEC = re.compile(r"\[([^\]]*)\]")

class CompileError(Exception):
    pass

def fail(line, text):
    raise CompileError(f"line {line}: {text}")

def num(src, line):
    if not re.fullmatch(NUMBER, src): fail(line, f"invalid number {src!r}")
    x = float(src)
    if not math.isfinite(x) or abs(x) >= 1e9: fail(line, "number outside supported range")
    return x

def integer(src, line, lo, hi):
    if not re.fullmatch(r"\d+", src): fail(line, "expected nonnegative integer")
    # Every accepted integer bound fits within 32 bits; reject huge literals
    # before Python's integer conversion limit can raise an uncaught ValueError.
    if len(src)>20: fail(line, "integer literal is too long")
    x = int(src)
    if not lo <= x <= hi: fail(line, f"integer outside {lo}..{hi}")
    return x

def ident(src, line):
    if not IDENT.fullmatch(src): fail(line, f"invalid identifier {src!r}")
    return src

def vector(src, line, size):
    m=VEC.fullmatch(src.strip())
    if not m: fail(line, "invalid vector syntax")
    values=[num(s.strip(),line) for s in m.group(1).split(',')]
    if len(values) != size: fail(line,f"expected {size} values, got {len(values)}")
    return values

def quoted(src, line):
    try: s=json.loads(src)
    except ValueError: fail(line,"expected quoted string")
    if not isinstance(s,str) or not s:
        fail(line,"invalid report name")
    try: encoded=s.encode('utf-8')
    except UnicodeError: fail(line,"report name contains invalid Unicode")
    if len(encoded)>80:
        fail(line,"invalid report name (maximum 80 UTF-8 bytes)")
    # C0/C1 controls, DEL, bidi format controls and other non-printables
    # are not portable/safe components of report filenames.
    if s in ('.', '..') or any(unicodedata.category(c).startswith('C') or unicodedata.category(c) in ('Zl','Zp') for c in s):
        fail(line,"report name contains forbidden control characters")
    return s

def parse(text):
    lines = [(i, s.strip()) for i,s in enumerate(text.splitlines(),1) if s.strip() and not s.lstrip().startswith('#')]
    if len(lines)>5000: raise CompileError("source exceeds 5000 lines")
    model=None; i=0
    def take():
        nonlocal i
        if i>=len(lines):raise CompileError("missing brain declaration")
        v=lines[i];i+=1;return v
    lineno,s=take()
    m=re.fullmatch(r"brain\s+(\w+)\s*\{",s)
    if not m: fail(lineno,"source must start with brain Name {")
    name=ident(m.group(1),lineno)
    senses=None; neurons=[]; seed=11; init_declared=False; seen=set(); budget=None
    while i<len(lines):
        line,s=take()
        if s=='}':break
        t=s.split()
        if t[0]=='senses' and len(t)==2:
            if senses is not None:fail(line,"duplicate senses")
            senses=integer(t[1],line,1,2048)
        elif t[0]=='neurons' and len(t) in (4,5):
            nm=ident(t[1],line)
            if nm in seen:fail(line,"duplicate neuron layer")
            seen.add(nm)
            storage=t[4] if len(t)==5 else 'float'
            if storage not in ('float','int8'): fail(line,"storage must be float or int8")
            neurons.append({'name':nm,'size':integer(t[2],line,1,4096),'activation':t[3],'storage':storage})
            if t[3] not in ('linear','relu','sigmoid'): fail(line,"activation must be linear, relu or sigmoid")
            if len(neurons)>8:fail(line,"maximum eight neuron layers")
        elif t[0]=='budget' and len(t)==3:
            if budget is not None: fail(line,"duplicate budget")
            budget=(integer(t[1],line,1,1<<30), integer(t[2],line,1,1<<30))
        elif t[0]=='init' and len(t)==2:
            if init_declared:fail(line,"duplicate init")
            seed=integer(t[1],line,0,4294967295)
            init_declared=True
        else:fail(line,"invalid brain declaration: "+s)
    else:raise CompileError("brain block missing }")
    if senses is None or not neurons: raise CompileError("brain needs senses and at least one neurons layer")
    shapes={};prev=senses;total=0
    for n in neurons:
        total+=prev*n['size']; shapes[n['name']]=(n['size'],prev);prev=n['size']
    if total>4000000:raise CompileError("brain exceeds total weight budget")
    # Static resident numeric tensors only: one owning weight array per layer,
    # gate bytes, biases, cached preactivations/activations, and input.
    # This excludes vectors/maps/allocator overhead, checkpoints, advise clones,
    # pending change queues and temporary training gradients. Not a process RSS cap.
    weight_bytes=0; storage_bytes=8*senses; macs=0; prev=senses
    for n in neurons:
        width=1 if n['storage']=='int8' else 8
        weight_bytes += prev*n['size']*width
        storage_bytes += prev*n['size']*(width+1) + n['size']*(8+16)
        macs += prev*n['size']
        prev=n['size']
    if budget is not None and (storage_bytes>budget[0] or macs>budget[1]):
        raise CompileError(f"budget exceeded: {storage_bytes} resident tensor bytes / {macs} forward MACs > {budget[0]} / {budget[1]}")
    model={'name':name,'senses':senses,'neurons':neurons,'init':seed,'weights':total,
           'weight_bytes':weight_bytes,'storage_bytes':storage_bytes,
           'macs':macs,'budget':budget,'budget_scope':'resident numeric tensors and forward MACs only'}
    def ops(nested=False,depth=0):
        nonlocal i
        if depth>8:raise CompileError("cycle nesting limit exceeded")
        out=[]
        while i<len(lines):
            line,s=take()
            if s=='}':
                if nested:return out
                fail(line,"unexpected }")
            if s.startswith('cycle '):
                cm=re.fullmatch(r"cycle\s+(\d+)\s*\{",s)
                if not cm:fail(line,"invalid cycle syntax")
                count=integer(cm.group(1),line,1,100000)
                child=ops(True,depth+1)
                out.append({'op':'cycle','count':count,'body':child,'line':line});continue
            mo=re.fullmatch(r"observe\s+(\[[^\]]*\])",s)
            if mo:out.append({'op':'observe','values':vector(mo.group(1),line,senses),'line':line});continue
            if s=='predict':out.append({'op':'predict','line':line});continue
            mo=re.fullmatch(r"learn\s+(\[[^\]]*\])\s+rate\s+(\S+)",s)
            if mo:
                rate=num(mo.group(2),line)
                if rate<=0 or rate>1:fail(line,"learning rate must be (0,1]")
                out.append({'op':'learn','values':vector(mo.group(1),line,prev),'rate':rate,'line':line});continue
            mo=re.fullmatch(r"tune\s+(auto|full|incremental)",s)
            if mo:out.append({'op':'tune','mode':mo.group(1),'line':line});continue
            mo=re.fullmatch(r"weight\s+(\w+)\s+(\d+)\s+(\d+)\s+add\s+(\S+)",s)
            if mo:
                layer=ident(mo.group(1),line)
                if layer not in shapes:fail(line,"unknown neuron layer")
                rows,cols=shapes[layer]
                out.append({'op':'weight','layer':layer,'row':integer(mo.group(2),line,0,rows-1),'col':integer(mo.group(3),line,0,cols-1),'delta':num(mo.group(4),line),'line':line});continue
            mo=re.fullmatch(r"edge\s+(\w+)\s+(\d+)\s+(\d+)\s+(on|off)",s)
            if mo:
                layer=ident(mo.group(1),line)
                if layer not in shapes:fail(line,"unknown neuron layer")
                rows,cols=shapes[layer]
                out.append({'op':'edge','layer':layer,'row':integer(mo.group(2),line,0,rows-1),'col':integer(mo.group(3),line,0,cols-1),'enabled':mo.group(4)=='on','line':line});continue
            mo=re.fullmatch(r"remember\s+(\w+)",s)
            if mo:out.append({'op':'remember','name':ident(mo.group(1),line),'line':line});continue
            mo=re.fullmatch(r"rewind\s+(\w+)",s)
            if mo:out.append({'op':'rewind','name':ident(mo.group(1),line),'line':line});continue
            if s=='inspect':out.append({'op':'inspect','line':line});continue
            mo=re.fullmatch(r"advise\s+(\d+)",s)
            if mo:out.append({'op':'advise','rounds':integer(mo.group(1),line,10,100000),'line':line});continue
            mo=re.fullmatch(r"optimize\s+(\d+)",s)
            if mo:out.append({'op':'optimize','rounds':integer(mo.group(1),line,20,2000),'line':line});continue
            mo=re.fullmatch(r"validate\s+(\[[^\]]*\])\s+target\s+(\[[^\]]*\])\s+tolerance\s+(\S+)",s)
            if mo:
                tol=num(mo.group(3),line)
                if tol<0:fail(line,"tolerance cannot be negative")
                out.append({'op':'validate','input':vector(mo.group(1),line,senses),'target':vector(mo.group(2),line,prev),'tolerance':tol,'line':line});continue
            mo=re.fullmatch(r"(save_model|load_model)\s+(\".*\")",s)
            if mo:
                fn=quoted(mo.group(2),line)
                if '/' in fn or '\\' in fn or '..' in fn:fail(line,"model file must have a simple filename")
                out.append({'op':mo.group(1),'filename':fn,'line':line});continue
            mo=re.fullmatch(r"verify\s+(\d+)",s)
            if mo:out.append({'op':'verify','rounds':integer(mo.group(1),line,2,2000),'line':line});continue
            mo=re.fullmatch(r"expect\s+(\[[^\]]*\])\s+tolerance\s+(\S+)",s)
            if mo:
                tol=num(mo.group(2),line)
                if tol<0:fail(line,"tolerance cannot be negative")
                out.append({'op':'expect','values':vector(mo.group(1),line,prev),'tolerance':tol,'line':line});continue
            mo=re.fullmatch(r"report\s+(\".*\")",s)
            if mo:
                fn=quoted(mo.group(1),line)
                if '/' in fn or '\\' in fn or '..' in fn: fail(line,"report must have a simple filename")
                out.append({'op':'report','filename':fn,'line':line});continue
            fail(line,"unknown instruction: "+s)
        if nested:raise CompileError("cycle block missing }")
        return out
    program=ops()
    def cost(arr):
        n=0
        for x in arr:
            n+=x['count']*cost(x['body']) if x['op']=='cycle' else 1
            if n>1000000:raise CompileError("program exceeds one million operations")
        return n
    steps=cost(program)
    if not program: raise CompileError("brain needs executable instructions")
    contract=validate_semantics(program)
    ir = {'ir_version':'NeuralIR/0.2','brain':model,'operations':program,
          'estimated_max_operations':steps,'semantic_contract':contract}
    ir['shape_contract'] = verify_typed_ir(ir)
    return ir



# Conservative state/effect vocabulary for the operations supported by NeuralIR/0.2.
# This metadata is an additive IR contract, NOT a proof that native execution
# cannot fail (e.g. numeric overflow, filesystem failures or external inputs).
# "writes" means "can change on success", not "must change".
_EFFECTS = {
    'observe':    ((), ('current_input', 'cache'), None),
    'predict':    (('current_input', 'model_parameters'), ('cache', 'counters', 'stdout'), None),
    'learn':      (('current_input', 'model_parameters'), ('model_parameters', 'cache', 'model_version', 'counters'), None),
    'tune':       ((), ('runtime_mode',), None),
    'weight':     (('model_parameters',), ('model_parameters', 'model_version', 'cache'), None),
    'edge':       (('model_parameters',), ('model_parameters', 'model_version', 'cache'), None),
    'remember':   (('current_input', 'model_parameters'), ('snapshots', 'stdout'), None),
    'rewind':     (('snapshots',), ('current_input', 'model_parameters', 'model_version', 'cache', 'stdout'), None),
    'inspect':    (('model_parameters',), ('stdout',), None),
    'advise':     (('current_input', 'model_parameters'), ('cache', 'counters', 'event_log', 'stdout'), None),
    'optimize':   (('current_input', 'model_parameters'), ('runtime_mode', 'cache', 'counters', 'event_log', 'stdout'), None),
    'validate':   (('model_parameters',), ('event_log', 'stdout'), None),
    'save_model': (('model_parameters',), ('stdout',), 'write'),
    'load_model': ((), ('model_parameters', 'model_version', 'runtime_mode', 'cache', 'stdout'), 'read'),
    'verify':     (('current_input', 'model_parameters'), ('event_log', 'stdout'), None),
    'expect':     (('current_input', 'model_parameters'), ('cache', 'counters', 'stdout'), None),
    'report':     (('model_parameters',), ('stdout',), 'write'),
}
_NEEDS_INPUT = frozenset({'predict', 'learn', 'expect', 'advise', 'verify', 'optimize'})


def validate_semantics(program):
    """Statically check deterministic snapshot/input lifetime and publish IR effects.

    The language has no branches and cycles have positive, constant counts.
    Checkpoints persist after rewind, are single-assignment, and are bounded to
    64 per native Model.  A loop containing a checkpoint may execute only once.
    A second abstract loop iteration is required: first iteration may remove
    input and cause a precondition failure only on a subsequent iteration.
    Since the input lattice is boolean and the body is deterministic, its
    transfer function is identity or a constant and stabilizes after 1 pass.
    """
    snapshots = {}
    counts = {}

    def walk(operations, observed, multiplier=1):
        for operation in operations:
            kind = operation['op']
            if kind == 'cycle':
                count = operation['count']
                # Reject guaranteed duplicate snapshots before processing a loop.
                observed = walk(operation['body'], observed, multiplier*count)
                if count > 1:
                    # Validate the second iteration's input preconditions.
                    # Do not count it again; counts represent dynamic operations.
                    observed = check_second_pass(operation['body'], observed)
                continue
            if kind not in _EFFECTS:
                raise CompileError(f"unsupported IR operation {kind!r}")
            counts[kind] = counts.get(kind, 0) + multiplier
            if kind in _NEEDS_INPUT and not observed:
                fail(operation['line'], f"{kind} requires observe before use (input state absent)")
            if kind == 'observe':
                observed = True
            elif kind == 'remember':
                if multiplier > 1:
                    fail(operation['line'], 'checkpoint in repeating cycle would be created more than once')
                label = operation['name']
                if label in snapshots:
                    fail(operation['line'], f"checkpoint already exists: {label}")
                if len(snapshots) >= 64:
                    fail(operation['line'], 'checkpoint limit exceeded (64)')
                snapshots[label] = observed
            elif kind == 'rewind':
                label = operation['name']
                if label not in snapshots:
                    fail(operation['line'], f"unknown checkpoint: {label}")
                observed = snapshots[label]
        return observed

    def check_second_pass(operations, observed):
        # No checkpoints can be defined inside a repeating cycle; walk the
        # already-validated body's input transitions without mutating metadata.
        for operation in operations:
            kind = operation['op']
            if kind == 'cycle':
                observed = check_second_pass(operation['body'], observed)
                # A nested positive-count cycle has the same boolean fixed point.
            elif kind in _NEEDS_INPUT and not observed:
                fail(operation['line'], f"{kind} requires observe before use (input state absent)")
            elif kind == 'observe':
                observed = True
            elif kind == 'rewind':
                observed = snapshots[operation['name']]
        return observed

    observed = walk(program, False)
    return {
        'version': 'NeuralEffects/1',
        'initial_state': {'input_bound': False},
        'final_state': {'input_bound': observed},
        'snapshot_count': len(snapshots),
        'checkpoints': {name: {'input_bound': state} for name, state in snapshots.items()},
        'operation_counts': dict(sorted(counts.items())),
        'effects': {
            kind: {'requires': ['current_input'] if kind in _NEEDS_INPUT else [],
                   'reads': list(_EFFECTS[kind][0]),
                   'writes': list(_EFFECTS[kind][1]),
                   'filesystem': _EFFECTS[kind][2]}
            for kind in sorted(counts)
        },
        'scope': 'static success-path model; runtime checks remain mandatory',
    }


def validate_input_state(program):
    """Compatibility wrapper for callers of the earlier precondition checker."""
    return validate_semantics(program)


# The .nir parser checks vector dimensions while reading source.  An IR may also
# be passed directly to emit(), or mutated after parse(), so the backend must
# independently validate the complete shape-bearing subset before codegen.
# NeuralShapes/1 covers the existing sequential dense layers and rank-1 f64
# operands; it is NOT a general tensor type system or a DAG verifier.
def verify_typed_ir(ir):
    """Validate NeuralIR/0.2 for the native sequential backend, derive type facts.

    Return the canonical, deterministic NeuralShapes/1 declaration.  Legacy
    in-memory IR without a shape_contract is accepted, but never trusted.
    """
    def require(condition, reason):
        if not condition:
            raise CompileError('Neural IR type/shape: ' + reason)

    def uint(value, lo, hi, label):
        require(type(value) is int and lo <= value <= hi, label + ' must be integer within bounds')
        return value

    def real(value, label, lo=None, hi=None, hi_exclusive=False):
        require(type(value) in (int, float) and abs(value) < 1e9 and math.isfinite(value),
                label + ' invalid numeric value')
        if lo is not None:
            require(value >= lo, label + ' below valid range')
        if hi is not None:
            require(value < hi if hi_exclusive else value <= hi,
                    label + ' above valid range')
        return value

    def vec(values, expected, label):
        require(type(values) is list and len(values) == expected,
                f'{label} vector shape mismatch (expected [{expected}])')
        for index, v in enumerate(values):
            real(v, f'{label} numeric component {index}')

    def identifier(value, label):
        require(type(value) is str and IDENT.fullmatch(value) is not None,
                label + ' invalid identifier')
        return value

    def filename(value, label):
        require(type(value) is str and value and value not in ('.', '..') and
                '/' not in value and '\\' not in value and '..' not in value,
                label + ' invalid filename')
        try:
            encoded = value.encode('utf-8')
        except UnicodeError:
            raise CompileError('Neural IR type/shape: ' + label + ' invalid filename Unicode')
        require(len(encoded) <= 80 and not any(
            unicodedata.category(c).startswith('C') or
            unicodedata.category(c) in ('Zl', 'Zp') for c in value),
            label + ' invalid filename characters')

    require(type(ir) is dict and ir.get('ir_version') == 'NeuralIR/0.2',
            'unsupported IR version (expected NeuralIR/0.2)')
    brain = ir.get('brain')
    require(type(brain) is dict, 'brain must be an object')
    identifier(brain.get('name'), 'brain name')
    senses = uint(brain.get('senses'), 1, 2048, 'senses')
    seed = uint(brain.get('init'), 0, 4294967295, 'init')
    layers = brain.get('neurons')
    require(type(layers) is list and 1 <= len(layers) <= 8,
            'neurons must contain 1..8 sequential layers')
    shapes = {}
    declarations = []
    prev = senses
    total = 0
    weight_bytes = 0
    storage_bytes = 8 * senses
    for index, layer in enumerate(layers):
        require(type(layer) is dict, f'layer {index} must be an object')
        name = identifier(layer.get('name'), f'layer {index} name')
        require(name not in shapes, f'duplicate neuron layer: {name}')
        size = uint(layer.get('size'), 1, 4096, f'layer {index} size')
        activation = layer.get('activation')
        require(type(activation) is str and activation in ('linear', 'relu', 'sigmoid'),
                f'layer {index} invalid activation')
        storage = layer.get('storage', 'float')
        require(type(storage) is str and storage in ('float', 'int8'),
                f'layer {index} invalid storage')
        shapes[name] = (size, prev)
        total += prev * size
        width = 1 if storage == 'int8' else 8
        weight_bytes += prev * size * width
        storage_bytes += prev * size * (width + 1) + size * (8 + 16)
        declarations.append({
            'name': name,
            'input': {'dtype': 'f64', 'shape': [prev]},
            'output': {'dtype': 'f64', 'shape': [size]},
            'weight_shape': [size, prev],
            'weight_storage': storage,
        })
        prev = size
    require(total <= 4000000, 'model exceeds maximum weights')
    for label, actual, calculated in (
        ('weights', brain.get('weights'), total),
        ('macs', brain.get('macs'), total),
        ('weight_bytes', brain.get('weight_bytes'), weight_bytes),
        ('storage_bytes', brain.get('storage_bytes'), storage_bytes),
    ):
        require(type(actual) is int and actual == calculated,
                f'{label} inconsistent with declared tensor shapes')
    require(brain.get('budget_scope') == 'resident numeric tensors and forward MACs only',
            'budget_scope mismatch')
    budget = brain.get('budget')
    if budget is not None:
        require(type(budget) in (tuple, list) and len(budget) == 2,
                'budget must be two positive bounds')
        resident_bound = uint(budget[0], 1, 1 << 30, 'budget resident bytes')
        mac_bound = uint(budget[1], 1, 1 << 30, 'budget forward MACs')
        require(storage_bytes <= resident_bound and total <= mac_bound,
                'budget exceeded for declared model')

    operations = ir.get('operations')
    require(type(operations) is list and operations, 'operations must be a non-empty list')
    signatures = {}

    def check_ops(ops, depth=0):
        require(type(ops) is list, 'cycle body must be an array of operations')
        require(depth <= 8, 'cycle nesting limit exceeded')
        cost = 0
        for op in ops:
            require(type(op) is dict, 'operation must be an object')
            kind = op.get('op')
            require(type(kind) is str and kind in _EFFECTS or kind == 'cycle',
                    'unsupported IR operation')
            uint(op.get('line'), 1, 5000, f'{kind} line')
            if kind == 'cycle':
                count = uint(op.get('count'), 1, 100000, 'cycle count')
                body = op.get('body')
                require(type(body) is list, 'cycle body must be an array of operations')
                inner_cost = check_ops(body, depth + 1)
                cost += inner_cost * count
            else:
                cost += 1
                if kind == 'observe':
                    vec(op.get('values'), senses, 'observe')
                    signatures[kind] = {'values': {'dtype': 'f64', 'shape': [senses]}}
                elif kind == 'learn':
                    vec(op.get('values'), prev, 'learn')
                    real(op.get('rate'), 'learn rate', 0, 1)
                    require(op['rate'] > 0, 'learn rate must be positive')
                    signatures[kind] = {'values': {'dtype': 'f64', 'shape': [prev]}, 'rate': 'f64'}
                elif kind == 'expect':
                    vec(op.get('values'), prev, 'expect')
                    real(op.get('tolerance'), 'expect tolerance', 0)
                    signatures[kind] = {'values': {'dtype': 'f64', 'shape': [prev]}, 'tolerance': 'f64'}
                elif kind == 'validate':
                    vec(op.get('input'), senses, 'validate input')
                    vec(op.get('target'), prev, 'validate target')
                    real(op.get('tolerance'), 'validate tolerance', 0)
                    signatures[kind] = {
                        'input': {'dtype': 'f64', 'shape': [senses]},
                        'target': {'dtype': 'f64', 'shape': [prev]},
                        'tolerance': 'f64',
                    }
                elif kind in ('weight', 'edge'):
                    layer = op.get('layer')
                    require(type(layer) is str and layer in shapes,
                            f'{kind} invalid layer')
                    rows, columns = shapes[layer]
                    require(type(op.get('row')) is int and 0 <= op['row'] < rows and
                            type(op.get('col')) is int and 0 <= op['col'] < columns,
                            f'{kind} index outside tensor shape')
                    if kind == 'weight':
                        real(op.get('delta'), 'weight delta')
                    else:
                        require(type(op.get('enabled')) is bool, 'edge enabled must be boolean')
                    signatures[kind] = {'layer': 'dense_layer', 'row': 'index', 'col': 'index'}
                elif kind == 'tune':
                    require(op.get('mode') in ('auto', 'full', 'incremental'), 'tune invalid mode')
                elif kind == 'advise':
                    uint(op.get('rounds'), 10, 100000, 'advise rounds')
                elif kind == 'verify':
                    uint(op.get('rounds'), 2, 2000, 'verify rounds')
                elif kind == 'optimize':
                    uint(op.get('rounds'), 20, 2000, 'optimize rounds')
                elif kind in ('remember', 'rewind'):
                    identifier(op.get('name'), kind + ' checkpoint')
                elif kind in ('save_model', 'load_model', 'report'):
                    filename(op.get('filename'), kind)
                # predict and inspect do not carry operands.
            require(cost <= 1000000, 'program exceeds one million operations')
        return cost

    total_ops = check_ops(operations)
    require(type(ir.get('estimated_max_operations')) is int and
            ir['estimated_max_operations'] == total_ops,
            'estimated_max_operations inconsistent with operations')
    shapes_contract = {
        'version': 'NeuralShapes/1',
        'input': {'dtype': 'f64', 'shape': [senses]},
        'output': {'dtype': 'f64', 'shape': [prev]},
        'layers': declarations,
        'operation_signatures': {k: signatures[k] for k in sorted(signatures)},
        'scope': 'sequential dense layers; rank-1 f64 input/output; no general tensor or DAG support',
    }
    if 'shape_contract' in ir:
        require(ir['shape_contract'] == shapes_contract, 'shape contract mismatch')
    return shapes_contract



def verify_semantic_contract(ir):
    """Verify effect facts against actual operations before native code emission.

    This is intentionally limited to state, snapshot lifetime and effect
    metadata; it is not a general verifier of all Neural IR shapes/types.
    Older IR dictionaries without metadata remain accepted but are checked.
    """
    contract = validate_semantics(ir['operations'])
    declared = ir.get('semantic_contract')
    if declared is not None and declared != contract:
        raise CompileError('Neural IR semantic contract mismatch')
    return contract


def analyze_ir(ir):
    """Deterministic compiler-side developer assistant; no external model."""
    verify_typed_ir(ir)
    verify_semantic_contract(ir)
    b=ir['brain']
    n=sum(x['size'] for x in b['neurons'])
    weights=b['weights']
    counts={k:0 for k in ('observe','predict','learn','weight','edge','remember','rewind','advise','verify','optimize','validate','save_model','load_model','inspect','report')}
    def scan(ops,scale=1):
        for op in ops:
            if op['op']=='cycle':scan(op['body'],scale*op['count'])
            elif op['op'] in counts:counts[op['op']]+=scale
    scan(ir['operations'])
    messages=[]
    if any(layer.get('storage')=='int8' for layer in b['neurons']):
        messages.append('INT8 uses fixed scale 1/127; small training steps can round to zero; validate convergence separately.')
    if counts['learn'] and not counts['predict']:
        messages.append('Training has no explicit final prediction; add a held-out observation and predict to evaluate it.')
    if counts['learn']:
        messages.append('Learning uses local SGD on squared error; no evidence of generalization or resistance to forgetting.')
    if counts['weight'] or counts['edge']:
        messages.append('Sparse mutation is supported; measure before assuming a runtime advantage.')
    if counts['verify']:
        messages.append('Verify: deterministic full-vs-incremental differential tests on isolated state copies; does not prove correctness against an independent oracle.')
    if counts['optimize']:
        messages.append('Optimize is an opt-in native runtime selection (AUTO vs FULL) on synthetic single-layer edits; it changes only mode, not weights or source. Not a general speed claim.')
    if counts['validate']:
        messages.append('Validate runs declared examples on isolated state copies. Held-out provenance remains the author responsibility.')
    if counts['save_model'] or counts['load_model']:
        messages.append('Model weights can be saved and loaded through a local versioned text format; only trusted local model files should be loaded.')
    if not counts['advise'] and not counts['optimize']:
        messages.append('No native advise command; strategy performance has not been measured.')
    if counts['advise']:
        messages.append('Advise compares runtime full and delta paths on synthetic edits; full may use BLAS with --blas.')
    return {'operations':counts,'weights':weights,
            'minimum_weight_storage_bytes_fp64':weights*8,
            'actual_weight_bytes':b['weight_bytes'],
            'estimated_resident_tensor_bytes':b['storage_bytes'],
            'estimated_forward_macs':b['macs'],
            'budget_scope':b['budget_scope'],
            'minimum_mask_storage_bytes':weights,
            'minimum_activation_storage_bytes_fp64':n*8,
            'advice':messages}

def number(v):
    return format(float(v),'.17g')
def vec(v):
    return 'std::vector<double>{'+','.join(number(x) for x in v)+'}'
def string(s):
    return json.dumps(s,ensure_ascii=False)

def emit(ir):
    verify_typed_ir(ir)
    verify_semantic_contract(ir)
    b=ir['brain']; lines=['#include "neural_runtime.hpp"','#include <exception>', 'int main() {','try {',f'neural::Model brain({string(b["name"])},{b["senses"]});']
    for k,l in enumerate(b['neurons']):
        typ={'linear':0,'relu':1,'sigmoid':2}[l['activation']]
        store=1 if l.get('storage')=='int8' else 0
        lines.append(f'brain.layer({string(l["name"])},{l["size"]},{typ},static_cast<uint32_t>({b["init"]}ULL+{k}*31337ULL),{store});')
    def block(ops,depth=0):
        for j,x in enumerate(ops):
            o=x['op']
            if o=='cycle':
                v=f'n_{depth}_{j}';lines.append(f'for(int {v}=0;{v}<{x["count"]};++{v}) {{');block(x['body'],depth+1);lines.append('}')
            elif o=='observe':lines.append(f'brain.observe({vec(x["values"])});')
            elif o=='predict':lines.append('brain.infer();')
            elif o=='learn':lines.append(f'brain.learn({vec(x["values"])},{number(x["rate"])});')
            elif o=='tune':lines.append('brain.mode=neural::Model::'+{'auto':'AUTO','full':'FULL','incremental':'DELTA'}[x['mode']]+';')
            elif o=='weight':lines.append(f'brain.change({string(x["layer"])},{x["row"]},{x["col"]},{number(x["delta"])});')
            elif o=='edge':lines.append(f'brain.gate({string(x["layer"])},{x["row"]},{x["col"]},{str(x["enabled"]).lower()});')
            elif o=='remember':lines.append(f'brain.checkpoint({string(x["name"])});')
            elif o=='rewind':lines.append(f'brain.restore({string(x["name"])});')
            elif o=='inspect':lines.append('brain.inspect();')
            elif o=='advise':lines.append(f'brain.advise({x["rounds"]});')
            elif o=='verify':lines.append(f'brain.verify({x["rounds"]});')
            elif o=='optimize':lines.append(f'brain.optimize({x["rounds"]});')
            elif o=='validate':lines.append(f'brain.validate({vec(x["input"])},{vec(x["target"])},{number(x["tolerance"])});')
            elif o=='save_model':lines.append(f'brain.save_model({string(x["filename"])});')
            elif o=='load_model':lines.append(f'brain.load_model({string(x["filename"])});')
            elif o=='expect':lines.append(f'brain.expect({vec(x["values"])},{number(x["tolerance"])});')
            elif o=='report':lines.append(f'brain.report({string(x["filename"])});')
            else:raise AssertionError(o)
    block(ir['operations'])
    lines+=['return 0;','} catch(const std::exception& e) {std::cerr<<"NEURAL ERROR: "<<e.what()<<"\\n";return 1;}','}']
    return '\n'.join(lines)+'\n'

def main(argv=None):
    ap=argparse.ArgumentParser(description='Compile .nir to native C++17 executable')
    ap.add_argument('source',type=Path)
    ap.add_argument('-o','--output',type=Path,default=Path('build/neural_program'))
    ap.add_argument('--emit-only',action='store_true',help='generate C++ and IR, do not compile')
    ap.add_argument('--run',action='store_true',help='run the compiled executable')
    ap.add_argument('--compiler',default='g++')
    ap.add_argument('--blas',action='store_true',help='link optional optimized BLAS backend (NumPy OpenBLAS or system CBLAS)')
    args=ap.parse_args(argv)
    try:
        src=args.source.read_text(encoding='utf-8')
        ir=parse(src);ir['source_sha256']=hashlib.sha256(src.encode('utf-8')).hexdigest()
        ir['compiler_advice']=analyze_ir(ir)
        out=args.output.resolve();out.parent.mkdir(parents=True,exist_ok=True)
        cpp=out.with_suffix('.cpp');cpp.write_text(emit(ir),encoding='utf-8')
        out.with_suffix('.nir.json').write_text(json.dumps(ir,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('COMPILED IR:',out.with_suffix('.nir.json'))
        print('COMPILER ADVICE:', '; '.join(ir['compiler_advice']['advice']))
        if args.emit_only:return 0
        cxx=shutil.which(args.compiler)
        if not cxx:raise CompileError('C++17 compiler not found: '+args.compiler)
        runtime=Path(__file__).resolve().parent/'runtime'
        link=[]
        if args.blas:
            import glob
            try:
                import numpy
                candidates=glob.glob(str(Path(numpy.__file__).resolve().parent.parent/'numpy.libs'/'libscipy_openblas64*.so'))
            except ImportError:
                candidates=[]
            if candidates:
                so=Path(candidates[0]).resolve()
                link=['-DNEURAL_RUNTIME_SCIPY_BLAS',str(so),f'-Wl,-rpath,{so.parent}']
            else:
                link=['-DNEURAL_RUNTIME_CBLAS','-lblas']
        subprocess.run([cxx,'-std=c++17','-O2','-Wall','-Wextra','-Werror','-I',str(runtime),str(cpp),*link,'-o',str(out)],check=True)
        print('BUILT:',out)
        if args.run:
            r=subprocess.run([str(out)],cwd=out.parent)
            return r.returncode
        return 0
    except (OSError,CompileError,subprocess.CalledProcessError) as e:
        print('NEURAL COMPILER ERROR:',e,file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
