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
_DATA_OPS = frozenset({"train_data", "predict_data"})
_MAX_DATA_SAMPLES = 100_000
_MAX_DATA_ELEMENTS = 8_000_000


def primitive_work(op):
    if op["op"] == "train_data":
        return op["samples"] * op["epochs"]
    if op["op"] == "predict_data":
        return op["samples"]
    return 1

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
    senses=None; neurons=[]; seed=11; seen=set(); budget=None
    while i<len(lines):
        line,s=take()
        if s=='}':break
        t=s.split()
        if t[0]=='senses' and len(t)==2:
            if senses is not None:fail(line,"duplicate senses")
            senses=integer(t[1],line,1,2048)
        elif t[0]=='neurons' and len(t)>=4:
            nm=ident(t[1],line)
            if nm=='input' or nm in seen:fail(line,"duplicate or reserved neuron layer")
            size=integer(t[2],line,1,4096)
            if t[3] not in ('linear','relu','sigmoid','tanh'):fail(line,"activation must be linear, relu, sigmoid or tanh")
            rest=t[4:]
            storage='float'
            if rest and rest[0] in ('float','int8'):storage=rest.pop(0)
            if rest and rest[0]!='from':fail(line,"expected storage or 'from earlier_layer ...'")
            implicit=[neurons[-1]['name']] if neurons else ['input']
            parents=implicit
            if rest:
                parents=rest[1:]
                if not 1<=len(parents)<=8:fail(line,"parent count must be 1..8")
                if len(set(parents))!=len(parents):fail(line,"duplicate graph parent")
                for parent in parents:
                    ident(parent,line)
                    if parent!='input' and parent not in seen:
                        fail(line,"graph parent must be input or an earlier layer: "+parent)
            layer={'name':nm,'size':size,'activation':t[3],'storage':storage}
            if parents!=implicit:layer['sources']=parents
            neurons.append(layer)
            seen.add(nm)
            if len(neurons)>8:fail(line,"maximum eight neuron layers")
        elif t[0]=='budget' and len(t)==3:
            if budget is not None: fail(line,"duplicate budget")
            budget=(integer(t[1],line,1,1<<30), integer(t[2],line,1,1<<30))
        elif t[0]=='init' and len(t)==2:seed=integer(t[1],line,0,4294967295)
        else:fail(line,"invalid brain declaration: "+s)
    else:raise CompileError("brain block missing }")
    if senses is None or not neurons: raise CompileError("brain needs senses and at least one neurons layer")
    # Acyclic feed-forward graph: parents must precede consumers.
    # Preserve legacy IR shape for sequential-only programs byte-for-byte.
    dimensions={'input':senses};shapes={};prev=senses;total=0
    weight_bytes=0;storage_bytes=8*senses;macs=0;graph=False
    for index,n in enumerate(neurons):
        implicit=[neurons[index-1]['name']] if index else ['input']
        parents=n.get('sources',implicit)
        if parents!=implicit:graph=True
        width=sum(dimensions[parent] for parent in parents)
        total+=width*n['size'];shapes[n['name']]=(n['size'],width)
        weight_width=1 if n['storage']=='int8' else 8
        weight_bytes+=width*n['size']*weight_width
        storage_bytes+=width*n['size']*(weight_width+1)+n['size']*(8+16)
        macs+=width*n['size']
        dimensions[n['name']]=n['size'];prev=n['size']
    if total>4000000:raise CompileError("brain exceeds total weight budget")
    if graph:
        reachable=set()
        layers_by_name={n['name']:n for n in neurons}
        def reach(name):
            if name=='input' or name in reachable:return
            reachable.add(name)
            i=next(i for i,n in enumerate(neurons) if n['name']==name)
            defaults=[neurons[i-1]['name']] if i else ['input']
            for source in layers_by_name[name].get('sources',defaults):reach(source)
        reach(neurons[-1]['name'])
        dead=[n['name'] for n in neurons if n['name'] not in reachable]
        if dead:raise CompileError('graph layers disconnected from output: '+', '.join(dead))
    if budget is not None and (storage_bytes>budget[0] or macs>budget[1]):
        raise CompileError(f"budget exceeded: {storage_bytes} resident tensor bytes / {macs} forward MACs > {budget[0]} / {budget[1]}")
    model={'name':name,'senses':senses,'neurons':neurons,'init':seed,'weights':total,
           'weight_bytes':weight_bytes,'storage_bytes':storage_bytes,
           'macs':macs,'budget':budget,'budget_scope':'resident numeric tensors and forward MACs only'}
    if graph:model['graph']=True
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
            mo=re.fullmatch(r"observe_delta\s+(\d+)\s+add\s+(\S+)",s)
            if mo:
                out.append({'op':'observe_delta','index':integer(mo.group(1),line,0,senses-1),
                            'delta':num(mo.group(2),line),'line':line});continue
            mo=re.fullmatch(r'(train_data|predict_data)\s+(".*")\s+samples\s+(\d+)'
                            r'(?:\s+epochs\s+(\d+)\s+rate\s+(\S+)\s+seed\s+(\d+))?',s)
            if mo:
                kind, source, samples, epochs, rate, shuffle_seed = mo.groups()
                if (kind == 'train_data') != (epochs is not None):
                    fail(line, 'train_data requires epochs, rate and seed; predict_data takes only samples')
                op={'op':kind,'filename':quoted(source,line),
                    'samples':integer(samples,line,1,_MAX_DATA_SAMPLES),'line':line}
                if epochs is not None:
                    op.update(epochs=integer(epochs,line,1,1000),rate=num(rate,line),
                              seed=integer(shuffle_seed,line,0,4294967295))
                out.append(op)
                continue
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
            n+=x['count']*cost(x['body']) if x['op']=='cycle' else primitive_work(x)
            if n>1000000:raise CompileError("program exceeds one million operations")
        return n
    steps=cost(program)
    if not program: raise CompileError("brain needs executable instructions")
    # Shared state/effect semantics, including restored input in snapshots.
    # The NeuralEffects/1 metadata is additive and does not change native code.
    effects=validate_semantics(program)
    result={'ir_version':'NeuralIR/0.4-graph' if graph else 'NeuralIR/0.2','brain':model,'operations':program,
            'estimated_max_operations':steps,'semantic_contract':effects}
    # Additive semantic IR metadata; never silently claim general DAG support.
    def collect_deltas(items):
        found=[]
        for op in items:
            if op['op']=='observe_delta':found.append(op)
            elif op['op']=='cycle':found.extend(collect_deltas(op['body']))
        return found
    deltas=collect_deltas(program)
    if deltas:
        result['neural_changes']={
            'contract':'NeuralChanges/1','topology':'dag_dense' if graph else 'sequential_dense_only',
            'input_width':senses,
            'dependencies':[{'source':parent,'target':n['name']}
                            for k,n in enumerate(neurons)
                            for parent in n.get('sources',[neurons[k-1]['name']] if k else ['input'])],
            'operation':'observe_delta','effect':{'reads':['input','weights','cache'],
                'writes':['input','cache'],'requires_observe':True},
            'occurrences':[{'line':x['line'],'index':x['index']} for x in deltas],
            'semantic':'input[index] += delta, then mathematically full forward at next use; '
                       'incremental evaluation allowed within numerical tolerance'
        }
    if graph:
        result['graph_contract']={
            'version':'NeuralGraph/1','output':neurons[-1]['name'],
            'edges':[{'source':src,'target':n['name']} for k,n in enumerate(neurons)
                     for src in n.get('sources',[neurons[k-1]['name']] if k else ['input'])],
            'scope':'ordered acyclic dense layers; inputs concatenated in listed parent order',
        }
    result['shape_contract']=verify_typed_ir(result)
    verify_neural_changes(result)
    return result



# The existing NeuralEffects/1 vocabulary from the typed-IR research branch,
# extended only for the already-implemented observe_delta operator. Do not
# confuse this with a full tensor/DAG effect system.
_EFFECTS = {
    'observe':       ((), ('current_input', 'cache'), None),
    'observe_delta': (('current_input',), ('current_input', 'cache'), None),
    'predict':       (('current_input', 'model_parameters'), ('cache', 'counters', 'stdout'), None),
    'train_data':   (('model_parameters',), ('current_input', 'model_parameters', 'cache', 'model_version', 'counters', 'stdout'), 'read'),
    'predict_data': (('model_parameters', 'runtime_mode'), ('current_input', 'cache', 'counters', 'stdout'), 'read'),
    'learn':         (('current_input', 'model_parameters'), ('model_parameters', 'cache', 'model_version', 'counters'), None),
    'tune':          ((), ('runtime_mode',), None),
    'weight':        (('model_parameters',), ('model_parameters', 'model_version', 'cache'), None),
    'edge':          (('model_parameters',), ('model_parameters', 'model_version', 'cache'), None),
    'remember':      (('current_input', 'model_parameters'), ('snapshots', 'stdout'), None),
    'rewind':        (('snapshots',), ('current_input', 'model_parameters', 'model_version', 'cache', 'stdout'), None),
    'inspect':       (('model_parameters',), ('stdout',), None),
    'advise':        (('current_input', 'model_parameters'), ('cache', 'counters', 'event_log', 'stdout'), None),
    'optimize':      (('current_input', 'model_parameters'), ('runtime_mode', 'cache', 'counters', 'event_log', 'stdout'), None),
    'validate':      (('model_parameters',), ('event_log', 'stdout'), None),
    'save_model':    (('model_parameters',), ('stdout',), 'write'),
    'load_model':    ((), ('model_parameters', 'model_version', 'runtime_mode', 'cache', 'stdout'), 'read'),
    'verify':        (('current_input', 'model_parameters'), ('event_log', 'stdout'), None),
    'expect':        (('current_input', 'model_parameters'), ('cache', 'counters', 'stdout'), None),
    'report':        (('model_parameters',), ('stdout',), 'write'),
}
_NEEDS_INPUT = frozenset({'observe_delta','predict','learn','expect','advise','verify','optimize'})


def validate_semantics(program):
    """Check input and snapshot lifetime without unrolling up to 10^6 cycles.

    Exact for current branch-free language with positive, statically bounded
    cycle counts and checkpoint values recording whether input was bound.
    The only transitions of the boolean input-state are identity and constants.
    """
    snapshots={}
    counts={}

    def require_input(op, observed):
        if op['op'] in _NEEDS_INPUT and not observed:
            if op['op']=='observe_delta':
                fail(op['line'],'observe_delta requires a preceding observe or restoring an observed checkpoint')
            fail(op['line'],f"{op['op']} requires observe before use (input state absent)")

    def next_epoch(ops, observed):
        """Second iteration / fixed-point check, without duplicating counts."""
        for op in ops:
            kind=op['op']
            if kind=='cycle':
                observed=next_epoch(op['body'],observed)
            else:
                require_input(op,observed)
                if kind=='observe' or kind in _DATA_OPS:observed=True
                elif kind=='rewind':observed=snapshots[op['name']]
        return observed

    def walk(ops,observed,multiplier=1):
        if not isinstance(ops,list):raise CompileError('NeuralEffects: operations must be a list')
        for op in ops:
            if not isinstance(op,dict):raise CompileError('NeuralEffects: operation must be an object')
            kind=op.get('op')
            if kind=='cycle':
                count=op.get('count')
                if type(count) is not int or not 1<=count<=100000:
                    raise CompileError('NeuralEffects: cycle count invalid')
                observed=walk(op.get('body'),observed,multiplier*count)
                if count>1:observed=next_epoch(op['body'],observed)
                continue
            if kind not in _EFFECTS:raise CompileError(f'NeuralEffects: unsupported operation {kind!r}')
            counts[kind]=counts.get(kind,0)+multiplier
            require_input(op,observed)
            if kind=='observe' or kind in _DATA_OPS:observed=True
            elif kind=='remember':
                label=op.get('name')
                if type(label) is not str:raise CompileError('NeuralEffects: checkpoint name invalid')
                if multiplier>1:fail(op['line'],'checkpoint in repeating cycle would be created more than once')
                if label in snapshots:fail(op['line'],f'checkpoint already exists: {label}')
                if len(snapshots)>=64:fail(op['line'],'checkpoint limit exceeded (64)')
                snapshots[label]=observed
            elif kind=='rewind':
                label=op.get('name')
                if type(label) is not str:
                    raise CompileError('NeuralEffects: checkpoint name invalid')
                if label not in snapshots:fail(op['line'],f'unknown checkpoint: {label}')
                observed=snapshots[label]
        return observed

    observed=walk(program,False)
    return {
        'version':'NeuralEffects/1',
        'initial_state':{'input_bound':False},
        'final_state':{'input_bound':observed},
        'snapshot_count':len(snapshots),
        'checkpoints':{name:{'input_bound':state} for name,state in snapshots.items()},
        'operation_counts':dict(sorted(counts.items())),
        'effects':{
            kind:{'requires':['current_input'] if kind in _NEEDS_INPUT else [],
                  'reads':list(_EFFECTS[kind][0]),
                  'writes':list(_EFFECTS[kind][1]),
                  'filesystem':_EFFECTS[kind][2]}
            for kind in sorted(counts)
        },
        'scope':'static success-path model; runtime checks remain mandatory',
    }


def verify_semantic_contract(ir):
    """Recheck effect metadata after any IR transform and before native emit."""
    contract=validate_semantics(ir['operations'])
    declared=ir.get('semantic_contract')
    if declared is not None and declared!=contract:
        raise CompileError('Neural IR semantic contract mismatch')
    return contract


def verify_typed_ir(ir):
    """Validate NeuralIR/0.2 dense types/shapes before emitting native code.

    Ported from the project's existing NeuralShapes/1 research contract, with
    a typed scalar input index and finite delta for observe_delta. This is
    intentionally NOT a general DAG or tensor type system.
    """
    def require(condition, reason):
        if not condition:
            raise CompileError('Neural IR type/shape: ' + reason)

    def uint(value, lo, hi, label):
        require(type(value) is int and lo <= value <= hi, label + ' must be integer within bounds')
        return value

    def real(value, label, lo=None, hi=None):
        require(type(value) in (int, float) and abs(value) < 1e9,
                label + ' invalid numeric value')
        if lo is not None: require(value >= lo, label + ' below valid range')
        if hi is not None: require(value <= hi, label + ' above valid range')
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
        require(type(value) is str and value and value not in ('.','..') and
                '/' not in value and chr(92) not in value and '..' not in value,
                label + ' invalid filename')
        try: encoded=value.encode('utf-8')
        except UnicodeError: raise CompileError('Neural IR type/shape: ' + label + ' invalid filename Unicode')
        require(len(encoded) <= 80 and not any(
            unicodedata.category(c).startswith('C') or
            unicodedata.category(c) in ('Zl','Zp') for c in value),
            label + ' invalid filename characters')

    require(type(ir) is dict and ir.get('ir_version') in ('NeuralIR/0.2','NeuralIR/0.4-graph'),
            'unsupported IR version')
    brain=ir.get('brain')
    require(type(brain) is dict, 'brain must be an object')
    identifier(brain.get('name'), 'brain name')
    senses=uint(brain.get('senses'),1,2048,'senses')
    uint(brain.get('init'),0,4294967295,'init')
    layers=brain.get('neurons')
    require(type(layers) is list and 1 <= len(layers) <= 8,
            'neurons must contain 1..8 sequential layers')
    shapes={};dimensions={'input':senses};declarations=[];prev=senses;total=0;weight_bytes=0;storage_bytes=8*senses;graph=False
    for index,layer in enumerate(layers):
        require(type(layer) is dict, f'layer {index} must be an object')
        name=identifier(layer.get('name'), f'layer {index} name')
        require(name != 'input' and name not in shapes, f'duplicate or reserved neuron layer: {name}')
        size=uint(layer.get('size'),1,4096,f'layer {index} size')
        activation=layer.get('activation')
        require(type(activation) is str and activation in ('linear','relu','sigmoid','tanh'),
                f'layer {index} invalid activation')
        storage=layer.get('storage','float')
        require(type(storage) is str and storage in ('float','int8'),
                f'layer {index} invalid storage')
        implicit=[layers[index-1]['name']] if index else ['input']
        parents=layer.get('sources',implicit)
        require(type(parents) is list and 1<=len(parents)<=8,'graph parents invalid')
        require(len(set(str(x) for x in parents))==len(parents),'duplicate graph parent')
        for parent in parents:
            identifier(parent,'graph parent')
            require(parent in dimensions,'graph parent must precede consumer')
        if parents!=implicit:graph=True
        width_in=sum(dimensions[parent] for parent in parents)
        shapes[name]=(size,width_in)
        total+=width_in*size
        width=1 if storage=='int8' else 8
        weight_bytes+=width_in*size*width
        storage_bytes+=width_in*size*(width+1)+size*(8+16)
        declaration={
            'name':name,
            'input':{'dtype':'f64','shape':[width_in]},
            'output':{'dtype':'f64','shape':[size]},
            'weight_shape':[size,width_in],
            'weight_storage':storage,
        }
        if graph:declaration['sources']=list(parents)
        declarations.append(declaration)
        dimensions[name]=size
        prev=size
    if graph:
        for i,d in enumerate(declarations):
            if 'sources' not in d:
                d['sources']=list(layers[i].get('sources',[layers[i-1]['name']] if i else ['input']))
        reachable=set()
        def visit(name):
            if name=='input' or name in reachable:return
            reachable.add(name)
            idx=next(i for i,d in enumerate(declarations) if d['name']==name)
            for parent in declarations[idx]['sources']:visit(parent)
        visit(layers[-1]['name'])
        require(len(reachable)==len(layers),'unreachable graph layer')
    require(ir['ir_version']==('NeuralIR/0.4-graph' if graph else 'NeuralIR/0.2'),
            'graph version mismatch')
    require(brain.get('graph',False) is graph,'graph flag mismatch')
    require(total <= 4000000,'model exceeds maximum weights')
    for label,actual,calculated in (
        ('weights',brain.get('weights'),total),
        ('macs',brain.get('macs'),total),
        ('weight_bytes',brain.get('weight_bytes'),weight_bytes),
        ('storage_bytes',brain.get('storage_bytes'),storage_bytes),
    ):
        require(type(actual) is int and actual == calculated,
                f'{label} inconsistent with declared tensor shapes')
    require(brain.get('budget_scope') == 'resident numeric tensors and forward MACs only',
            'budget_scope mismatch')
    budget=brain.get('budget')
    if budget is not None:
        require(type(budget) in (tuple,list) and len(budget)==2,
                'budget must be two positive bounds')
        resident_bound=uint(budget[0],1,1 << 30,'budget resident bytes')
        mac_bound=uint(budget[1],1,1 << 30,'budget forward MACs')
        require(storage_bytes <= resident_bound and total <= mac_bound,
                'budget exceeded for declared model')

    operations=ir.get('operations')
    require(type(operations) is list and operations, 'operations must be a non-empty list')
    signatures={}
    def check_ops(ops,depth=0):
        require(type(ops) is list,'cycle body must be an array of operations')
        require(depth <= 8, 'cycle nesting limit exceeded')
        cost=0
        for op in ops:
            require(type(op) is dict,'operation must be an object')
            kind=op.get('op')
            require(type(kind) is str and (kind in _EFFECTS or kind=='cycle'),
                    'unsupported IR operation')
            uint(op.get('line'),1,5000,f'{kind} line')
            if graph and kind in ('advise','verify','optimize'):
                fail(op['line'],kind+' not implemented for DAG graphs')
            if kind=='cycle':
                count=uint(op.get('count'),1,100000,'cycle count')
                body=op.get('body')
                require(type(body) is list,'cycle body must be an array of operations')
                cost+=check_ops(body,depth+1)*count
            else:
                if kind=='observe':
                    vec(op.get('values'),senses,'observe')
                    signatures[kind]={'values':{'dtype':'f64','shape':[senses]}}
                elif kind=='observe_delta':
                    uint(op.get('index'),0,senses-1,'observe_delta index')
                    real(op.get('delta'),'observe_delta delta')
                    signatures[kind]={'index':'input_index','delta':'f64'}
                elif kind=='learn':
                    vec(op.get('values'),prev,'learn')
                    real(op.get('rate'),'learn rate',0,1)
                    require(op['rate']>0,'learn rate must be positive')
                    signatures[kind]={'values':{'dtype':'f64','shape':[prev]},'rate':'f64'}
                elif kind in _DATA_OPS:
                    filename(op.get('filename'),kind)
                    samples=uint(op.get('samples'),1,_MAX_DATA_SAMPLES,kind+' samples')
                    require(samples*senses <= _MAX_DATA_ELEMENTS,kind+' dataset element limit exceeded')
                    signature={'features':{'dtype':'f64','rank':2,'columns':senses}}
                    if kind=='train_data':
                        require(prev>=2,'train_data requires at least two output classes')
                        uint(op.get('epochs'),1,1000,'train_data epochs')
                        uint(op.get('seed'),0,4294967295,'train_data seed')
                        real(op.get('rate'),'train_data rate',0,1)
                        require(op['rate']>0,'train_data rate must be positive')
                        signature.update(label={'dtype':'i32','classes':prev},rate='f64',
                                         epochs='u32',seed='u32')
                    else:
                        require(not {'epochs','rate','seed'}.intersection(op),
                                'predict_data cannot carry training operands')
                        signature['logits']={'dtype':'f64','columns':prev}
                    signatures[kind]=signature
                elif kind=='expect':
                    vec(op.get('values'),prev,'expect')
                    real(op.get('tolerance'),'expect tolerance',0)
                    signatures[kind]={'values':{'dtype':'f64','shape':[prev]},'tolerance':'f64'}
                elif kind=='validate':
                    vec(op.get('input'),senses,'validate input')
                    vec(op.get('target'),prev,'validate target')
                    real(op.get('tolerance'),'validate tolerance',0)
                    signatures[kind]={
                        'input':{'dtype':'f64','shape':[senses]},
                        'target':{'dtype':'f64','shape':[prev]},
                        'tolerance':'f64',
                    }
                elif kind in ('weight','edge'):
                    layer=op.get('layer')
                    require(type(layer) is str and layer in shapes, f'{kind} invalid layer')
                    rows,columns=shapes[layer]
                    require(type(op.get('row')) is int and 0 <= op['row'] < rows and
                            type(op.get('col')) is int and 0 <= op['col'] < columns,
                            f'{kind} index outside tensor shape')
                    if kind=='weight':real(op.get('delta'),'weight delta')
                    else:require(type(op.get('enabled')) is bool,'edge enabled must be boolean')
                    signatures[kind]={'layer':'dense_layer','row':'index','col':'index'}
                elif kind=='tune':
                    require(type(op.get('mode')) is str and op['mode'] in ('auto','full','incremental'),
                            'tune invalid mode')
                elif kind=='advise':uint(op.get('rounds'),10,100000,'advise rounds')
                elif kind=='verify':uint(op.get('rounds'),2,2000,'verify rounds')
                elif kind=='optimize':uint(op.get('rounds'),20,2000,'optimize rounds')
                elif kind in ('remember','rewind'):
                    label=op.get('name')
                    require(type(label) is str and IDENT.fullmatch(label) is not None,
                            'checkpoint name invalid')
                elif kind in ('save_model','load_model','report'):filename(op.get('filename'),kind)
                # predict and inspect carry no operands
                cost+=primitive_work(op)
            require(cost <= 1000000,'program exceeds one million operations')
        return cost
    total_ops=check_ops(operations)
    require(type(ir.get('estimated_max_operations')) is int and
            ir['estimated_max_operations']==total_ops,
            'estimated_max_operations inconsistent with operations')
    shape_contract={
        'version':'NeuralShapes/1',
        'input':{'dtype':'f64','shape':[senses]},
        'output':{'dtype':'f64','shape':[prev]},
        'layers':declarations,
        'operation_signatures':{k:signatures[k] for k in sorted(signatures)},
        'scope':'topologically ordered dense DAG; rank-1 f64; no general tensor support' if graph
                else 'sequential dense layers; rank-1 f64 input/output; no general tensor or DAG support',
    }
    if 'shape_contract' in ir:
        require(ir['shape_contract']==shape_contract,'shape contract mismatch')
    if graph:
        graph_contract={
            'version':'NeuralGraph/1',
            'output':layers[-1]['name'],
            'edges':[{'source':src,'target':d['name']} for d in declarations for src in d['sources']],
            'scope':'ordered acyclic dense layers; inputs concatenated in listed parent order',
        }
        if ir.get('graph_contract')!=graph_contract:
            require(False,'graph contract mismatch')
    else:
        require('graph_contract' not in ir,'sequential IR cannot carry graph contract')
    return shape_contract


def verify_neural_changes(ir):
    """Validate the additive NeuralChanges/1 contract before analysis or codegen.

    This is deliberately narrow; it is not a general-purpose typed DAG verifier.
    """
    verify_typed_ir(ir)
    verify_semantic_contract(ir)
    brain=ir.get('brain')
    if not isinstance(brain,dict):raise CompileError('NeuralChanges: missing brain')
    senses=brain.get('senses')
    if type(senses) is not int or not (1<=senses<=2048):
        raise CompileError('NeuralChanges: input width invalid')
    neurons=brain.get('neurons')
    if not isinstance(neurons,list) or not neurons:
        raise CompileError('NeuralChanges: neuron structure invalid')
    # Lifecycle rules are checked by verify_semantic_contract above, using
    # checkpoint-aware NeuralEffects. Only delta payload/occurrences are checked
    # here; the old naive rewind=False walk incorrectly rejected valid restores.
    occurrences=[]
    def visit(items):
        if not isinstance(items,list):raise CompileError('NeuralChanges: operations invalid')
        for op in items:
            if not isinstance(op,dict):raise CompileError('NeuralChanges: operation invalid')
            kind=op.get('op')
            if kind=='observe_delta':
                idx=op.get('index')
                delta=op.get('delta')
                lineno=op.get('line')
                if type(idx) is not int or not 0<=idx<senses:
                    raise CompileError('NeuralChanges: observe_delta index invalid')
                if type(delta) not in (int,float) or not math.isfinite(delta) or abs(delta)>=1e9:
                    raise CompileError('NeuralChanges: observe_delta amount invalid')
                if type(lineno) is not int or lineno<1:
                    raise CompileError('NeuralChanges: physical line invalid')
                occurrences.append({'line':lineno,'index':idx})
            elif kind=='cycle':visit(op.get('body'))
    visit(ir.get('operations'))
    meta=ir.get('neural_changes')
    if not occurrences:
        if meta is not None:raise CompileError('NeuralChanges: orphan metadata')
        return
    if not isinstance(meta,dict) or meta.get('contract')!='NeuralChanges/1' or meta.get('topology')!=('dag_dense' if brain.get('graph') else 'sequential_dense_only'):
        raise CompileError('NeuralChanges: contract missing or invalid')
    deps=[]
    for k,layer in enumerate(neurons):
        if not isinstance(layer,dict) or type(layer.get('name')) is not str:
            raise CompileError('NeuralChanges: invalid layer')
        for parent in layer.get('sources',[neurons[k-1]['name']] if k else ['input']):
            deps.append({'source':parent,'target':layer['name']})
    if meta.get('dependencies')!=deps or meta.get('occurrences')!=occurrences or meta.get('input_width')!=senses:
        raise CompileError('NeuralChanges: corrupted dependencies or occurrences')
    if meta.get('operation')!='observe_delta' or meta.get('effect')!={
        'reads':['input','weights','cache'],
        'writes':['input','cache'],'requires_observe':True}:
        raise CompileError('NeuralChanges: effect mismatch')

def analyze_ir(ir):
    """Deterministic compiler-side developer assistant; no external model."""
    verify_neural_changes(ir)
    b=ir['brain']
    n=sum(x['size'] for x in b['neurons'])
    weights=b['weights']
    counts={k:0 for k in ('observe','observe_delta','predict','learn','weight','edge','remember','rewind','advise','verify','optimize','validate','save_model','load_model','inspect','report')}
    def scan(ops,scale=1):
        for op in ops:
            if op['op']=='cycle':scan(op['body'],scale*op['count'])
            elif op['op'] in _DATA_OPS:counts[op['op']]=counts.get(op['op'],0)+scale*primitive_work(op)
            elif op['op'] in counts:counts[op['op']]+=scale
    scan(ir['operations'])
    messages=[]
    if any(layer.get('storage')=='int8' for layer in b['neurons']):
        messages.append('INT8 uses fixed scale 1/127; small training steps can round to zero; validate convergence separately.')
    if (counts['learn'] or counts.get('train_data')) and not (counts['predict'] or counts.get('predict_data')):
        messages.append('Training has no explicit final prediction; add a held-out observation and predict to evaluate it.')
    if counts['learn'] or counts.get('train_data'):
        messages.append('Learning uses local SGD on squared error; no evidence of generalization or resistance to forgetting.')
    if counts['observe_delta']:
        messages.append('Observe_delta is a stateful input patch; AUTO may use delta propagation or a full recompute; no guaranteed mobile win.')
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
    if ir.get('compute_plan'):
        if any(l.get('storage') == 'int8' for l in b['neurons']):
            messages.append('mobile-cpu plan specializes sequential dense FP64 arithmetic with packed INT8 storage dequantized per term: four independent rows, same left-to-right reduction as FULL. Not native INT8 dot products and not a phone measurement.')
        else:
            messages.append('mobile-cpu plan specializes sequential dense FP64 inference: four independent rows, same left-to-right reduction as FULL. Not an INT8 kernel and not a phone measurement.')
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

_ACT_CODE = {'linear': 0, 'relu': 1, 'sigmoid': 2, 'tanh': 3}

def make_mobile_plan(ir):
    """Sequential dense FP64 arithmetic, with FP64 or packed INT8 weights."""
    brain = ir.get('brain')
    if not isinstance(brain, dict):
        raise CompileError('mobile-cpu profile requires a brain')
    if brain.get('graph'):
        raise CompileError('mobile-cpu profile supports sequential dense layers only, not a DAG')
    layers_in = brain.get('neurons')
    if not isinstance(layers_in, list) or not layers_in:
        raise CompileError('mobile-cpu profile requires at least one layer')
    width = brain.get('senses')
    layers = []
    for layer in layers_in:
        storage = layer.get('storage', 'float')
        if storage not in ('float', 'int8'):
            raise CompileError('mobile-cpu profile requires float or int8 storage')
        if 'sources' in layer:
            raise CompileError('mobile-cpu profile supports sequential layers only')
        activation = layer.get('activation')
        if activation not in _ACT_CODE:
            raise CompileError('mobile-cpu profile has an unsupported activation')
        out = layer.get('size')
        planned_layer = {
            'name': layer['name'],
            'in': width,
            'out': out,
            'activation': activation,
            'fusion': 'gemv_bias_activation',
            'weight_order': 'row-major-int8-dequant-fp64' if storage == 'int8' else 'row-major-fp64',
            'reduction': 'left-to-right per row; up to 4 independent rows',
        }
        if storage == 'int8':
            planned_layer['dequantization'] = 'double(q) * scale per term, then multiply by FP64 input'
        layers.append(planned_layer)
        width = out
    plan = {
        'contract': 'NeuralPlan/1',
        'profile': 'mobile-cpu',
        'numeric': 'fp64-strict-row-order',
        'fallback': 'generic Model::forward when any edge is disabled or storage is not fp64',
        'blocking_rows': 4,
        'lifetime': 'reuses Model.z and Model.a; no per-inference temporary vector',
        'eliminated_from_hot_path': [
            'storage and mask dispatch inside the row loop',
            'runtime input width in the inner loop',
            'per-inference temporary vectors',
        ],
        'not_claimed': [
            'not INT8 inference',
            'not a new matrix-multiplication algorithm',
            'not a phone measurement',
            'not an FMA contraction',
        ],
        'layers': layers,
    }
    if any(l.get('storage') == 'int8' for l in layers_in):
        plan['fallback'] = 'generic Model::forward when any edge is disabled or shape/storage differs from the compiled plan'
        plan['not_claimed'][0] = 'not native INT8 dot products'
    return plan

def verify_mobile_plan(ir):
    plan = ir.get('compute_plan')
    if plan is None:
        return None
    expected = make_mobile_plan(ir)
    if plan != expected:
        raise CompileError('NeuralPlan/1 does not match the brain')
    return plan

def mobile_function_source(ir):
    brain = ir['brain']
    senses = brain['senses']
    layers = brain['neurons']
    lines = [
        'static bool nl_mobile_try(neural::Model& m){',
        f'if(m.graph||m.layers.size()!={len(layers)}||m.ninput!={senses})return false;',
    ]
    width = senses
    calls = []
    for index, layer in enumerate(layers):
        out = layer['size']
        act = _ACT_CODE[layer['activation']]
        quantized = layer.get('storage') == 'int8'
        storage = 1 if quantized else 0
        lines.append(
            f'if(m.layers[{index}].in!={width}||m.layers[{index}].out!={out}||'
            f'm.layers[{index}].type!={act}||m.layers[{index}].storage!={storage}||'
            f'm.layers[{index}].disabled!=0)return false;'
        )
        source = 'm.input.data()' if index == 0 else f'm.a[{index - 1}].data()'
        calls.append(f'if((int)m.z[{index}].size()!={out}){{m.z[{index}].assign({out},0);m.a[{index}].assign({out},0);}}')
        kernel = 'mobile_dense_int8_ilp4' if quantized else 'mobile_dense_ilp4'
        weights = 'q' if quantized else 'w'
        scale = f',m.layers[{index}].scale' if quantized else ''
        calls.append(
            f'neural::{kernel}<{width},{out},{act}>('
            f'm.layers[{index}].{weights}.data(),m.layers[{index}].b.data(),{source},'
            f'm.z[{index}].data(),m.a[{index}].data(){scale});'
        )
        width = out
    lines.extend(calls)
    lines.append('return true;')
    lines.append('}')
    return '\n'.join(lines)

def emit(ir):
    verify_neural_changes(ir)
    plan=verify_mobile_plan(ir)
    b=ir['brain']; lines=['#include "neural_runtime.hpp"','#include <exception>', 'int main() {','try {',f'neural::Model brain({string(b["name"])},{b["senses"]});']
    for k,l in enumerate(b['neurons']):
        typ={'linear':0,'relu':1,'sigmoid':2,'tanh':3}[l['activation']]
        store=1 if l.get('storage')=='int8' else 0
        if b.get('graph'):
            sources=l.get('sources',[b['neurons'][k-1]['name']] if k else ['input'])
            srcs='std::vector<std::string>{'+','.join(string(p) for p in sources)+'}'
            lines.append(f'brain.layer_sources({string(l["name"])},{l["size"]},{typ},static_cast<uint32_t>({b["init"]}ULL+{k}*31337ULL),{store},{srcs});')
        else:
            lines.append(f'brain.layer({string(l["name"])},{l["size"]},{typ},static_cast<uint32_t>({b["init"]}ULL+{k}*31337ULL),{store});')
    if plan is not None:
        lines.append('brain.mobile_forward=&nl_mobile_try;')
    def block(ops,depth=0):
        for j,x in enumerate(ops):
            o=x['op']
            if o=='cycle':
                v=f'n_{depth}_{j}';lines.append(f'for(int {v}=0;{v}<{x["count"]};++{v}) {{');block(x['body'],depth+1);lines.append('}')
            elif o=='observe':lines.append(f'brain.observe({vec(x["values"])});')
            elif o=='observe_delta':lines.append(f'brain.change_input({x["index"]},{number(x["delta"])});')
            elif o=='predict':lines.append('brain.infer();')
            elif o=='train_data':
                lines.append(f'neural::train_data(brain,{string(x["filename"])},{x["samples"]},{x["epochs"]},{number(x["rate"])},{x["seed"]}u);')
            elif o=='predict_data':
                # Generic emission stays byte-for-byte compatible. The mobile
                # profile can reuse the bound input buffer in FULL mode; AUTO
                # and DELTA retain their existing observe semantics.
                reuse=',true' if plan is not None else ''
                lines.append(f'neural::predict_data(brain,{string(x["filename"])},{x["samples"]}{reuse});')
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
    if any(line.startswith(('neural::train_data(', 'neural::predict_data(')) for line in lines):
        lines.insert(1,'#include "neural_dataset.hpp"')
    if plan is not None:
        lines.insert(1,'#include "mobile_plan.hpp"')
        main_at=lines.index('int main() {')
        lines.insert(main_at, mobile_function_source(ir))
    lines+=['return 0;','} catch(const std::exception& e) {std::cerr<<"NEURAL ERROR: "<<e.what()<<"\\n";return 1;}','}']
    return '\n'.join(lines)+'\n'

def main(argv=None):
    ap=argparse.ArgumentParser(description='Compile .nir to native C++17 executable')
    ap.add_argument('source',type=Path)
    ap.add_argument('-o','--output',type=Path,default=Path('build/neural_program'))
    ap.add_argument('--emit-only',action='store_true',help='generate C++ and IR, do not compile')
    ap.add_argument('--run',action='store_true',help='run the compiled executable')
    ap.add_argument('--compiler',default='g++')
    ap.add_argument('--profile',choices=('generic','mobile-cpu'),default='generic',
                    help='generic keeps the previous C++ path; mobile-cpu emits NeuralPlan/1')
    ap.add_argument('--blas',action='store_true',help='link optional optimized BLAS backend (NumPy OpenBLAS or system CBLAS)')
    args=ap.parse_args(argv)
    try:
        src=args.source.read_text(encoding='utf-8')
        ir=parse(src);ir['source_sha256']=hashlib.sha256(src.encode('utf-8')).hexdigest()
        if args.profile=='mobile-cpu':
            ir['compute_plan']=make_mobile_plan(ir)
        ir['compiler_advice']=analyze_ir(ir)
        out=args.output.resolve()
        cpp=out.with_suffix('.cpp')
        ir_file=out.with_suffix('.nir.json')
        if args.source.resolve() in (out,cpp,ir_file):
            raise CompileError('output would overwrite source; choose another --output path')
        out.parent.mkdir(parents=True,exist_ok=True)
        cpp.write_text(emit(ir),encoding='utf-8')
        ir_file.write_text(json.dumps(ir,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
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
        defines=[]
        if ir.get('compute_plan'):
            defines.append('-DNEURAL_MOBILE_PLAN')
        subprocess.run([cxx,'-std=c++17','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror',*defines,'-I',str(runtime),str(cpp),*link,'-o',str(out)],check=True)
        print('BUILT:',out)
        if args.run:
            r=subprocess.run([str(out)],cwd=out.parent)
            return r.returncode
        return 0
    except (OSError,CompileError,subprocess.CalledProcessError) as e:
        print('NEURAL COMPILER ERROR:',e,file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
