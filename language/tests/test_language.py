import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from compiler import parse,emit,CompileError

PREAMBLE='''brain TestNet {
  senses 2
  neurons hidden 4 relu
  neurons middle 3 relu
  neurons out 2 sigmoid
  init 41
}
'''

def run_native(src, td, filename='test_program'):
    srcpath=td/(filename+'.nir')
    srcpath.write_text(src,encoding='utf-8')
    cmd=[sys.executable,str(ROOT/'compiler.py'),str(srcpath),'-o',str(td/filename)]
    compile_result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if compile_result.returncode:
        raise AssertionError('compilation failed: '+compile_result.stderr)
    return subprocess.run([str(td/filename)],cwd=td,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)

def outputs(text):
    return [[float(x) for x in m.group(1).split(',')] for m in re.finditer(r'^PREDICT version=\d+ values=([^\n]+)',text,re.M)]

class CompilerTests(unittest.TestCase):
    def test_parse_minimal(self):
        x=parse('brain A {\n senses 2\n neurons B 1 sigmoid\n init 42\n}\nobserve [0, 1]\npredict')
        self.assertEqual(x['brain']['weights'],2)
        self.assertEqual(x['operations'][1]['op'],'predict')
        self.assertIn('brain.infer();',emit(x))
    def test_unknown_instruction_rejected(self):
        with self.assertRaises(CompileError):parse(PREAMBLE+'execute bash')
    def test_no_python_expression_in_vectors(self):
        with self.assertRaises(CompileError):parse(PREAMBLE+'observe [__import__("os"), 2]')
    def test_shape_rejected_at_compile(self):
        with self.assertRaisesRegex(CompileError,'expected 2 values'):parse(PREAMBLE+'observe [1]')
    def test_unknown_weight_rejected(self):
        with self.assertRaisesRegex(CompileError,'unknown neuron layer'):parse(PREAMBLE+'weight missing 0 0 add 1')
    def test_bad_index_rejected(self):
        with self.assertRaises(CompileError):parse(PREAMBLE+'weight hidden 900 0 add 0.1')
    def test_learning_rate_rejected(self):
        with self.assertRaisesRegex(CompileError,'learning rate'):parse(PREAMBLE+'learn [0, 1] rate 9')
    def test_unclosed_cycle_rejected(self):
        with self.assertRaisesRegex(CompileError,'cycle block missing'):parse(PREAMBLE+'cycle 4 {\n predict')
    def test_enormous_loop_rejected(self):
        with self.assertRaises(CompileError):parse(PREAMBLE+'cycle 100000 {\n cycle 100000 {\n predict\n}\n}')
    def test_path_traversal_rejected(self):
        with self.assertRaises(CompileError):parse(PREAMBLE+'report "../escape.json"')
    def test_target_shape_rejected(self):
        with self.assertRaises(CompileError):parse(PREAMBLE+'learn [1] rate 0.2')
    def test_duplicate_layer_rejected(self):
        with self.assertRaises(CompileError):parse('brain A {\n senses 1\n neurons same 2 relu\n neurons same 1 sigmoid\n}\npredict')
    def test_bounded_resources(self):
        with self.assertRaises(CompileError):parse('brain A {\n senses 2048\n neurons a 4096 relu\n}\npredict')
    def test_cpp_not_scripting_layer(self):
        cpp=emit(parse(PREAMBLE+'observe [1, 2]\npredict'))
        self.assertNotIn('Python.h',cpp)
        self.assertIn('brain.layer(',cpp)

class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='neural_language_tests_')
        cls.td=Path(cls.temp.name)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_sparse_vs_full_three_layer(self):
        common='''observe [0.25, -0.3]
predict
weight hidden 0 0 add 0.15
predict
weight middle 1 0 add -0.13
predict
edge out 0 0 off
predict
edge out 0 0 on
predict
weight out 0 1 add 0.01
predict
'''
        full=run_native(PREAMBLE+'tune full\n'+common,self.td,'full')
        delta=run_native(PREAMBLE+'tune incremental\n'+common,self.td,'delta')
        self.assertEqual(full.returncode,0,full.stderr)
        self.assertEqual(delta.returncode,0,delta.stderr)
        aa,bb=outputs(full.stdout),outputs(delta.stdout)
        self.assertEqual(len(aa),6)
        for x,y in zip(aa,bb):
            for a,b in zip(x,y):self.assertAlmostEqual(a,b,places=9)
    def test_relu_crossing_and_snapshot(self):
        source='''brain Small {
 senses 1
 neurons hidden 1 relu
 neurons out 1 linear
 init 27
}
observe [1]
tune incremental
predict
remember before
weight hidden 0 0 add -1
predict
weight hidden 0 0 add 2
predict
rewind before
predict
'''
        r=run_native(source,self.td,'relu')
        self.assertEqual(r.returncode,0,r.stderr)
        vals=outputs(r.stdout)
        self.assertEqual(len(vals),4)
        self.assertAlmostEqual(vals[0][0],vals[3][0],places=10)
    def test_real_learning_and_output(self):
        source=(ROOT/'examples/learning.nir').read_text(encoding='utf-8')
        r=run_native(source,self.td,'learning')
        self.assertEqual(r.returncode,0,r.stderr)
        preds=outputs(r.stdout)
        self.assertEqual(len(preds),7)
        self.assertLess(preds[0][0],0.1)
        self.assertLess(preds[1][0],0.1)
        self.assertLess(preds[2][0],0.1)
        self.assertGreater(preds[3][0],0.9)
        self.assertAlmostEqual(preds[3][0],preds[-1][0],places=9)
        self.assertIn('ADVICE',r.stdout)
        report=json.loads((self.td/'learning_report.json').read_text())
        self.assertEqual(report['counts']['train'],2000)
        self.assertEqual(report['counts']['changes'],2)
    def test_native_advisor_does_not_mutate_model(self):
        source='''brain Demo {
 senses 2
 neurons output 1 sigmoid
 init 3
}
observe [1, 1]
predict
advise 200
predict
report "audit.json"
'''
        r=run_native(source,self.td,'adviser')
        self.assertEqual(r.returncode,0,r.stderr)
        a=outputs(r.stdout)
        self.assertEqual(a[0],a[1])
        report=json.loads((self.td/'audit.json').read_text())
        self.assertEqual(report['version'],0)
        self.assertEqual(report['counts']['changes'],0)
        self.assertIn('equal=PASS',r.stdout)
    def test_expect_checks_result(self):
        src='''brain Test {
 senses 1
 neurons output 1 linear
 init 2
}
observe [0]
expect [0] tolerance 1e-12
'''
        r=run_native(src,self.td,'expect')
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('EXPECT PASS',r.stdout)
        bad=run_native(src.replace('expect [0]','expect [1]'),self.td,'expectbad')
        self.assertNotEqual(bad.returncode,0)
    def test_failure_does_not_change_original_source(self):
        src=PREAMBLE+'predict'
        p=self.td/'orig.nir';p.write_text(src)
        old=p.read_bytes()
        r=run_native(src,self.td,'runtimefail')
        self.assertNotEqual(r.returncode,0)
        self.assertIn('observe',r.stderr)
        self.assertEqual(p.read_bytes(),old)

if __name__=='__main__':unittest.main(verbosity=2)

class PreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='neural_language_preservation_')
        cls.td=Path(cls.temp.name)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_repeated_run_keeps_first_report(self):
        src='''brain Test {\n senses 1\n neurons result 1 linear\n init 5\n}\nobserve [0]\nreport "saved.json"'''
        r1=run_native(src,self.td,'repeat_report')
        self.assertEqual(r1.returncode,0,r1.stderr)
        a=(self.td/'saved.json').read_bytes()
        r2=subprocess.run([str(self.td/'repeat_report')],cwd=self.td,capture_output=True,text=True)
        self.assertEqual(r2.returncode,0,r2.stderr)
        self.assertEqual((self.td/'saved.json').read_bytes(),a)
        self.assertTrue((self.td/'saved_1.json').exists())
    def test_checkpoint_duplicate_rejected(self):
        src='''brain Test {\n senses 1\n neurons result 1 linear\n init 5\n}\nremember start\nremember start'''
        r=run_native(src,self.td,'dup_checkpoint')
        self.assertNotEqual(r.returncode,0)
        self.assertIn('checkpoint already exists',r.stderr)
