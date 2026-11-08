#!/usr/bin/env python3
"""Neural Language 0.4: source -> validated Neural IR -> C++17 -> native binary.
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
            if nm in seen or nm=='input':fail(line,"duplicate or reserved neuron layer")
            if t[3] not in ('linear','relu','sigmoid'):fail(line,"activation must be linear, relu or sigmoid")
            size=integer(t[2],line,1,4096)
            rest=t[4:]
            storage='float'
            if rest and rest[0] in ('float','int8'):
                storage=rest.pop(0)
            implicit=[neurons[-1]['name']] if neurons else ['input']
            parents=implicit
            if rest:
                if rest[0]!='from' or len(rest)<2:fail(line,"expected 'from input' or 'from earlier_layer ...'")
                parents=rest[1:]
                if len(parents)>8:fail(line,"too many parent sources")
                if len(set(parents))!=len(parents):fail(line,"duplicate parent source")
                for parent in parents:
                    ident(parent,line)
                    if parent!='input' and parent not in seen:
                        fail(line,"parent must be an earlier neuron layer or input: "+parent)
            neurons.append({'name':nm,'size':size,'activation':t[3],'storage':storage,'sources':parents})
            seen.add(nm)
            if len(neurons)>8:fail(line,"maximum eight neuron layers")
        elif t[0]=='budget' and len(t)==3:
            if budget is not None: fail(line,"duplicate budget")
            budget=(integer(t[1],line,1,1<<30), integer(t[2],line,1,1<<30))
        elif t[0]=='init' and len(t)==2:seed=integer(t[1],line,0,4294967295)
        else:fail(line,"invalid brain declaration: "+s)
    else:raise CompileError("brain block missing }")
    if senses is None or not neurons: raise CompileError("brain needs senses and at least one neurons layer")
    # A topologically ordered, feed-forward DAG. Parent values are concatenated
    # in the order written in the source. Output is the last declared layer.
    dimensions={'input':senses}
    shapes={};total=0;macs=0;weight_bytes=0;storage_bytes=8*senses
    prev=senses
    graph=False
    for k,n in enumerate(neurons):
        expected=[neurons[k-1]['name']] if k else ['input']
        if n['sources']!=expected:graph=True
        width=sum(dimensions[p] for p in n['sources'])
        total+=width*n['size'];macs+=width*n['size']
        width_bytes=1 if n['storage']=='int8' else 8
        weight_bytes+=width*n['size']*width_bytes
        storage_bytes+=width*n['size']*(width_bytes+1)+n['size']*(8+16)
        shapes[n['name']]=(n['size'],width)
        dimensions[n['name']]=n['size']
        prev=n['size']
    if total>4000000:raise CompileError("brain exceeds total weight budget")
    # A declared branch that cannot reach the sole output has no trainable
    # purpose in this version. Reject silent dead ends, not merely cycles.
    if graph:
        by_name={n['name']:n for n in neurons}
        connected=set()
        def visit(name):
            if name=='input' or name in connected:return
            connected.add(name)
            for parent in by_name[name]['sources']:visit(parent)
        visit(neurons[-1]['name'])
        dead=[n['name'] for n in neurons if n['name'] not in connected]
        if dead:raise CompileError("neuron layers not connected to output: "+', '.join(dead))
    # Static resident tensor estimate, not a hard RSS bound; the graph's
    # temporary concatenation vectors and training workspaces are excluded.
    if budget is not None and (storage_bytes>budget[0] or macs>budget[1]):
        raise CompileError(f"budget exceeded: {storage_bytes} resident tensor bytes / {macs} forward MACs > {budget[0]} / {budget[1]}")
    model={'name':name,'senses':senses,'neurons':neurons,'init':seed,'weights':total,
           'weight_bytes':weight_bytes,'storage_bytes':storage_bytes,
           'macs':macs,'budget':budget,'budget_scope':'resident numeric tensors and forward MACs only', 'graph':graph}
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
    if graph:
        def check_graph_ops(xs):
            for x in xs:
                if x['op']=='cycle':check_graph_ops(x['body'])
                elif x['op'] in ('advise','verify','optimize') or (x['op']=='tune' and x['mode']=='incremental'):
                    fail(x['line'],"incremental/delta benchmarking is not implemented for branching graphs")
        check_graph_ops(program)
    if not program: raise CompileError("brain needs executable instructions")
    return {'ir_version':'NeuralIR/0.4-graph' if graph else 'NeuralIR/0.2','brain':model,'operations':program,'estimated_max_operations':steps}

def analyze_ir(ir):
    """Deterministic compiler-side developer assistant; no external model."""
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
    if b.get('graph'):
        messages.append('DAG branch/merge graph: ordered parent activations concatenate at each layer; native full recomputation only. No graph delta speed claim.')
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
    b=ir['brain']; lines=['#include "neural_runtime.hpp"','#include <exception>', 'int main() {','try {',f'neural::Model brain({string(b["name"])},{b["senses"]});']
    for k,l in enumerate(b['neurons']):
        typ={'linear':0,'relu':1,'sigmoid':2}[l['activation']]
        store=1 if l.get('storage')=='int8' else 0
        if b.get('graph'):
            srcs='std::vector<std::string>{'+','.join(string(p) for p in l['sources'])+'}'
            lines.append(f'brain.layer_sources({string(l["name"])},{l["size"]},{typ},static_cast<uint32_t>({b["init"]}ULL+{k}*31337ULL),{store},{srcs});')
        else:
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
