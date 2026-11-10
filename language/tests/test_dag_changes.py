"""NL-COMP-006: end-to-end graph shape/effect/delta integration.

The C++ stress program contains an independent long-double oracle. Every
negative case below must be rejected before the generated native code executes.
"""
from __future__ import annotations
import copy
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from compiler import CompileError,parse,emit,analyze_ir

GRAPH=("brain Graph {\n senses 4\n neurons left 5 relu from input\n"
       " neurons right 3 sigmoid from input\n neurons joined 2 linear from left right input\n init 33\n}\n")
OPS="observe [0.1,0.2,0.3,-0.4]\npredict\nobserve_delta 2 add 0.02\npredict\n"

class DAGShapeAndChanges(unittest.TestCase):
    def test_graph_shapes_and_edges(self):
        ir=parse(GRAPH+OPS)
        self.assertEqual(ir['ir_version'],'NeuralIR/0.4-graph')
        self.assertEqual(ir['brain']['weights'],4*5+4*3+(5+3+4)*2)
        self.assertEqual(ir['shape_contract']['layers'][2]['weight_shape'],[2,12])
        self.assertEqual(ir['shape_contract']['layers'][2]['sources'],['left','right','input'])
        self.assertEqual(ir['graph_contract']['output'],'joined')
        self.assertEqual(len(ir['graph_contract']['edges']),5)
        self.assertEqual(ir['neural_changes']['topology'],'dag_dense')
        self.assertEqual(ir['neural_changes']['dependencies'],ir['graph_contract']['edges'])
        self.assertIn('brain.layer_sources(',emit(ir))
        self.assertIn('brain.change_input(',emit(ir))

    def test_bad_graphs_fail_before_compilation(self):
        with self.assertRaisesRegex(CompileError,'parent'):
            parse(GRAPH.replace('from left right input','from later')+OPS)
        with self.assertRaisesRegex(CompileError,'duplicate graph parent'):
            parse(GRAPH.replace('from left right input','from left left input')+OPS)
        with self.assertRaisesRegex(CompileError,'disconnected'):
            parse(GRAPH.replace('from left right input','from left input')+OPS)
        with self.assertRaisesRegex(CompileError,'reserved'):
            parse(GRAPH.replace('neurons right','neurons input')+OPS)
        with self.assertRaisesRegex(CompileError,'parent'):
            parse(GRAPH.replace('from left right input','from joined right')+OPS)
        with self.assertRaisesRegex(CompileError,'parent count'):
            parse(GRAPH.replace('from left right input','from')+OPS)

    def test_mutated_ir_fails_controlled(self):
        ir=parse(GRAPH+OPS)
        variants=[]
        def varied(transform):
            x=copy.deepcopy(ir);transform(x);variants.append(x)
        varied(lambda x:x['brain']['neurons'][2].update(sources=['joined']))
        varied(lambda x:x['brain']['neurons'][2].update(sources=['left','left']))
        varied(lambda x:x['brain']['neurons'][2].update(sources=['missing']))
        varied(lambda x:x['brain']['neurons'][2].update(sources=[['left']]))
        varied(lambda x:x['brain']['neurons'][2].update(size=3))
        varied(lambda x:x['brain']['neurons'][2].update(sources=['input']))
        varied(lambda x:x['graph_contract']['edges'][0].update(source='right'))
        varied(lambda x:x['shape_contract']['layers'][2]['weight_shape'].__setitem__(1,2))
        varied(lambda x:x['neural_changes'].update(topology='sequential_dense_only'))
        varied(lambda x:x['neural_changes'].update(dependencies=[]))
        varied(lambda x:x['operations'][2].update(index=8))
        varied(lambda x:x['operations'][2].update(delta=float('nan')))
        varied(lambda x:x['operations'][2].update(delta=True))
        varied(lambda x:x['brain'].update(weights=42))
        varied(lambda x:x.update(ir_version='NeuralIR/0.2'))
        for item in variants:
            with self.subTest(case=variants.index(item)),self.assertRaises(CompileError):
                emit(item)
            with self.assertRaises(CompileError):analyze_ir(item)

    def test_codegen_rejects_dag_operations_even_with_recomputed_effects(self):
        from compiler import validate_semantics

        for name, rounds in (("verify", 20), ("advise", 20), ("optimize", 90)):
            ir = parse(GRAPH + "observe [0,0,0,0]\ncycle 2 {\npredict\n}\n")
            ir["operations"][1]["body"][0].update(op=name, rounds=rounds)
            ir["semantic_contract"] = validate_semantics(ir["operations"])
            with self.subTest(name=name), self.assertRaisesRegex(CompileError, "not implemented"):
                emit(ir)

    def test_codegen_reports_malformed_ir_as_compile_error(self):
        ir = parse(GRAPH + OPS)
        bad_cases = (
            None, {},
            dict(ir, operations=[{"op": []}]),
            dict(ir, operations=[{"op": "observe", "line": 1, "values": [10**1000, 0, 0, 0]}]),
        )
        for index, bad in enumerate(bad_cases):
            with self.subTest(index=index), self.assertRaises(CompileError):
                emit(bad)

    def test_snapshot_and_effect_lifetimes(self):
        program=GRAPH+('observe [0.1,0.2,0.3,-0.4]\nremember x\n'
                       'observe_delta 2 add 0.02\nrewind x\nobserve_delta 2 add 0.03\npredict\n')
        ir=parse(program)
        self.assertTrue(ir['semantic_contract']['checkpoints']['x']['input_bound'])
        self.assertEqual(ir['semantic_contract']['operation_counts']['observe_delta'],2)
        with self.assertRaisesRegex(CompileError,'observe_delta requires'):
            parse(GRAPH+'remember x\nobserve [0.1,0.2,0.3,-0.4]\nrewind x\nobserve_delta 1 add 0.1\n')

    def test_sequential_explicit_parent_remains_legacy(self):
        data=('brain Legacy {\n senses 2\n neurons x 4 relu from input\n'
              ' neurons y 1 linear from x\n}\nobserve [0.1,0.2]\npredict\n')
        ir=parse(data)
        self.assertEqual(ir['ir_version'],'NeuralIR/0.2')
        self.assertNotIn('graph_contract',ir)

    def test_invalid_graph_benchmark_commands_do_not_run(self):
        for cmd in ('optimize 90','verify 20','advise 20'):
            with self.subTest(cmd=cmd),self.assertRaisesRegex(CompileError,'not implemented'):
                parse(GRAPH+'observe [0,0,0,0]\n'+cmd+'\n')

    @unittest.skipUnless(shutil.which('g++'), 'requires GCC')
    def test_checkpoint_reload_and_topology_guard(self):
        with tempfile.TemporaryDirectory(prefix='nl_dag_model_') as tmp:
            root=Path(tmp)
            base=GRAPH+'observe [0.1,0.2,0.3,-0.4]\npredict\nsave_model "saved.nlw"\n'
            reload=GRAPH+'load_model "saved.nlw"\nobserve [0.1,0.2,0.3,-0.4]\npredict\n'
            changed=GRAPH.replace('from left right input','from right left input')+(
                'load_model "saved.nlw"\nobserve [0.1,0.2,0.3,-0.4]\npredict\n')
            outputs=[]
            for label,src in (('save',base),('reload',reload),('mismatch',changed)):
                program=root/(label+'.nir');program.write_text(src)
                binary=root/label
                build=subprocess.run([sys.executable,str(ROOT/'compiler.py'),str(program),
                                      '-o',str(binary)],cwd=root,capture_output=True,text=True,timeout=60)
                self.assertEqual(build.returncode,0,build.stderr)
                proc=subprocess.run([str(binary)],cwd=root,capture_output=True,text=True,timeout=10)
                if label=='mismatch':
                    self.assertNotEqual(proc.returncode,0)
                    self.assertIn('topology mismatch',proc.stderr)
                else:
                    self.assertEqual(proc.returncode,0,proc.stderr)
                    outputs.append(re.findall(r'PREDICT version=\d+ values=([^\n]+)',proc.stdout,re.M)[0])
            self.assertEqual(outputs[0],outputs[1])
    @unittest.skipUnless(shutil.which('g++') and shutil.which('clang++'),'requires GCC and Clang')
    def test_native_graph_execution_and_checkpoints(self):
        with tempfile.TemporaryDirectory(prefix='nl_dag_') as d:
            d=Path(d)
            sample=GRAPH+('tune incremental\nobserve [0.1,0.2,0.3,-0.4]\npredict\n'
                          'remember prior\nobserve_delta 2 add 0.01\npredict\n'
                          'rewind prior\npredict\nlearn [0.1,0.2] rate 0.01\npredict\n'
                          'save_model "graph.weights"\n')
            for cc in ('g++','clang++'):
                output=d/('graph_gcc' if cc=='g++' else 'graph_clang')
                program=d/(cc.replace('+','p')+'.nir');program.write_text(sample,encoding='utf-8')
                built=subprocess.run([sys.executable,str(ROOT/'compiler.py'),str(program),'-o',str(output),
                                      '--compiler',cc],cwd=d,capture_output=True,text=True,timeout=65)
                self.assertEqual(built.returncode,0,built.stderr)
                result=subprocess.run([str(output)],cwd=d,capture_output=True,text=True,timeout=15)
                self.assertEqual(result.returncode,0,result.stderr)
                values=re.findall(r'^PREDICT version=\d+ values=([^\n]+)',result.stdout,re.M)
                self.assertEqual(len(values),4)
                self.assertEqual(values[0],values[2])
                checkpoint=(d/'graph.weights')
                self.assertTrue(checkpoint.read_text().startswith('NLMODEL3\n'))
                checkpoint.unlink()

    @unittest.skipUnless(shutil.which('g++') and shutil.which('clang++'),'requires GCC and Clang')
    def test_native_independent_dag_oracle_stress(self):
        with tempfile.TemporaryDirectory(prefix='nl_dag_cpp_') as d:
            for cc in ('g++','clang++'):
                output=Path(d)/(cc.replace('+','p'))
                built=subprocess.run([cc,'-O2','-std=c++17','-Wall','-Wextra','-Werror',
                                      '-I',str(ROOT/'runtime'),str(ROOT/'tests/graph_change_stress.cpp'),
                                      '-o',str(output)],capture_output=True,text=True,timeout=65)
                self.assertEqual(built.returncode,0,built.stderr)
                result=subprocess.run([str(output)],capture_output=True,text=True,timeout=35)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('DAG_CHANGE_PASS',result.stdout)

if __name__=='__main__':unittest.main()
