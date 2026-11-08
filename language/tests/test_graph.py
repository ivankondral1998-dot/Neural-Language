"""Graph-0.4: DAG parse, compile, train, checkpoint and finite-difference tests."""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from compiler import parse,emit,CompileError
TRAIN=(ROOT/'examples/graph_train.nir').read_text(encoding='utf-8')
INFER=(ROOT/'examples/graph_inference.nir').read_text(encoding='utf-8')

def build(src: str, folder: Path, name: str, cc='g++'):
    path=folder/(name+'.nir');path.write_text(src,encoding='utf-8')
    result=subprocess.run([sys.executable,str(ROOT/'compiler.py'),str(path),'-o',str(folder/name),'--compiler',cc],text=True,capture_output=True,timeout=60)
    if result.returncode:raise AssertionError(result.stderr+'\n'+result.stdout)
    return folder/name

class GraphStaticTests(unittest.TestCase):
    def test_ir_and_generated_code(self):
        ir=parse(TRAIN)
        self.assertTrue(ir['brain']['graph'])
        self.assertEqual(ir['brain']['weights'],12)
        self.assertEqual(ir['brain']['macs'],12)
        self.assertEqual([x['sources'] for x in ir['brain']['neurons']],[['input'],['input'],['left','right']])
        self.assertIn('brain.layer_sources(',emit(ir))
    def test_legacy_chain_stays_legacy(self):
        x=parse('brain A {\n senses 2\n neurons l 3 linear\n neurons r 1 sigmoid\n}\nobserve [1,0]\npredict')
        self.assertFalse(x['brain']['graph'])
        self.assertEqual(x['brain']['weights'],9)
        self.assertIn('brain.layer(',emit(x))
    def test_forward_parent_rejected(self):
        with self.assertRaisesRegex(CompileError,'earlier'):
            parse('brain A {\n senses 2\n neurons l 2 linear from r\n neurons r 2 linear from input\n}\npredict')
    def test_unknown_parent_rejected(self):
        with self.assertRaises(CompileError):
            parse('brain A {\n senses 2\n neurons l 2 linear from missing\n}\npredict')
    def test_duplicate_source_rejected(self):
        with self.assertRaisesRegex(CompileError,'duplicate parent'):
            parse('brain A {\n senses 2\n neurons l 2 linear from input input\n}\npredict')
    def test_dead_branch_rejected(self):
        with self.assertRaisesRegex(CompileError,'not connected'):
            parse('brain A {\n senses 2\n neurons abandoned 1 linear from input\n neurons output 1 linear from input\n}\npredict')
    def test_too_large_graph_budget_rejected(self):
        with self.assertRaisesRegex(CompileError,'budget exceeded'):
            parse('brain A {\n senses 2\n neurons a 4 linear from input\n neurons b 4 linear from input\n neurons out 1 linear from a b\n budget 1 1\n}\npredict')
    def test_int8_with_graph_is_accepted(self):
        ir=parse('brain A {\n senses 2\n neurons a 3 relu int8 from input\n neurons b 2 relu from input\n neurons out 1 sigmoid from a b\n}\nobserve [1,1]\npredict')
        self.assertEqual(ir['brain']['weights'],15)
    def test_graph_delta_profiling_rejected(self):
        base='brain A {\n senses 2\n neurons a 1 linear from input\n neurons b 1 linear from input\n neurons c 1 linear from a b\n}\nobserve [1,1]\n'
        for operation in ('tune incremental','advise 100','verify 40','optimize 40'):
            with self.subTest(operation=operation),self.assertRaisesRegex(CompileError,'not implemented'):
                parse(base+operation)

class GraphRuntimeTests(unittest.TestCase):
    def test_training_and_second_binary_gcc(self):
        with tempfile.TemporaryDirectory(prefix='nl04_graph_') as td:
            d=Path(td);t=build(TRAIN,d,'trainer');i=build(INFER,d,'infer')
            tr=subprocess.run([str(t)],cwd=d,capture_output=True,text=True,timeout=20)
            self.assertEqual(tr.returncode,0,tr.stderr)
            self.assertIn('PREDICT version=2000 values=0.475',tr.stdout)
            self.assertEqual((d/'graph_trained.nlw').read_text().splitlines()[0],'NLMODEL3')
            inf=subprocess.run([str(i)],cwd=d,capture_output=True,text=True,timeout=20)
            self.assertEqual(inf.returncode,0,inf.stderr)
            self.assertEqual(re.findall(r'PREDICT version=\d+ values=([^\n]+)',tr.stdout),
                             re.findall(r'PREDICT version=\d+ values=([^\n]+)',inf.stdout))
            self.assertEqual(inf.stdout.count('VALIDATE PASS'),2)
            report=json.loads((d/'graph_infer_report.json').read_text())
            self.assertEqual(report['counts']['train'],0)
    @unittest.skipUnless(shutil.which('clang++'),'clang++ unavailable')
    def test_graph_clang(self):
        with tempfile.TemporaryDirectory(prefix='nl04_clang_') as td:
            d=Path(td);t=build(TRAIN,d,'trainer','clang++')
            run=subprocess.run([str(t)],cwd=d,capture_output=True,text=True,timeout=25)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertIn('VALIDATE PASS',run.stdout)
    def test_topology_mismatch_cannot_load(self):
        altered=INFER.replace('from left right','from right left')
        with tempfile.TemporaryDirectory(prefix='nl04_mismatch_') as td:
            d=Path(td);tr=build(TRAIN,d,'tr');ir=build(altered,d,'infer')
            self.assertEqual(subprocess.run([str(tr)],cwd=d,capture_output=True,timeout=20).returncode,0)
            result=subprocess.run([str(ir)],cwd=d,text=True,capture_output=True,timeout=15)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('model graph topology mismatch',result.stderr)
    def test_graph_snapshot_and_mask(self):
        src='''brain A {
 senses 2
 neurons left 1 linear from input
 neurons right 1 linear from input
 neurons out 1 linear from left right
}
observe [0.5,0.8]
predict
remember before
edge left 0 0 off
weight right 0 0 add 0.2
predict
rewind before
predict
'''
        with tempfile.TemporaryDirectory(prefix='nl04_snap_') as td:
            d=Path(td);exe=build(src,d,'test');run=subprocess.run([str(exe)],cwd=d,capture_output=True,text=True,timeout=15)
            self.assertEqual(run.returncode,0,run.stderr)
            vals=re.findall(r'PREDICT version=\d+ values=([^\n]+)',run.stdout)
            self.assertEqual(len(vals),3)
            self.assertEqual(vals[0],vals[2])
    def test_finite_difference_oracle(self):
        with tempfile.TemporaryDirectory(prefix='nl04_grad_') as td:
            d=Path(td);src=ROOT/'tests/graph_gradient_oracle.cpp';exe=d/'grad'
            c=subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT/'runtime'),str(src),'-o',str(exe)],capture_output=True,text=True,timeout=60)
            self.assertEqual(c.returncode,0,c.stderr)
            r=subprocess.run([str(exe)],capture_output=True,text=True,timeout=15)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertIn('GRADIENT_ORACLE_PASS',r.stdout)

if __name__=='__main__':unittest.main(verbosity=2)
