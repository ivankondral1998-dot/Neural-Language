"""V0.3 integration regression: compiler/IR/native/validation/advisor loop."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from compiler import CompileError,emit,parse,analyze_ir

PROGRAM=(ROOT/'examples/integrated.nir').read_text(encoding='utf-8')

class IntegratedSyntax(unittest.TestCase):
    def test_ops_present_in_ir_and_native_cpp(self):
        ir=parse(PROGRAM)
        names=[x['op'] for x in ir['operations']]
        self.assertIn('validate',names)
        self.assertIn('optimize',names)
        self.assertIn('verify',names)
        self.assertIn('brain.optimize(90);',emit(ir))
        self.assertIn('brain.validate(',emit(ir))
        self.assertEqual(analyze_ir(ir)['operations']['validate'],4)
    def test_bad_validation_input_rejected(self):
        with self.assertRaisesRegex(CompileError,'expected 2 values'):
            parse('brain Test {\n senses 2\n neurons o 1 linear\n}\nvalidate [1] target [0] tolerance 0.1')
    def test_bad_validation_target_rejected(self):
        with self.assertRaisesRegex(CompileError,'expected 1 values'):
            parse('brain Test {\n senses 2\n neurons o 1 linear\n}\nvalidate [1,2] target [0,1] tolerance 0.1')
    def test_bad_optimize_range_rejected(self):
        with self.assertRaises(CompileError):
            parse('brain Test {\n senses 1\n neurons o 1 linear\n}\noptimize 2')
    def test_negative_tolerance_rejected(self):
        with self.assertRaises(CompileError):
            parse('brain Test {\n senses 1\n neurons o 1 linear\n}\nvalidate [1] target [0] tolerance -0.1')

class IntegrationRuntime(unittest.TestCase):
    def native(self,source, compiler='g++', name='test'):
        with tempfile.TemporaryDirectory(prefix='nl03_') as td:
            d=Path(td); path=d/(name+'.nir');path.write_text(source,encoding='utf-8')
            exe=d/name
            built=subprocess.run([sys.executable,str(ROOT/'compiler.py'),str(path),'-o',str(exe),'--compiler',compiler],text=True,capture_output=True,timeout=80)
            self.assertEqual(built.returncode,0,built.stderr)
            run=subprocess.run([str(exe)],cwd=d,capture_output=True,text=True,timeout=35)
            reports={p.name:json.loads(p.read_text(encoding='utf-8')) for p in d.glob('*.json') if p.name not in [name+'.nir.json']}
            return run,reports
    def test_integrated_native_gcc(self):
        r,reports=self.native(PROGRAM,name='gcc')
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(r.stdout.count('VALIDATE PASS'),4)
        self.assertIn('VERIFY PASS',r.stdout)
        self.assertIn('OPTIMIZE selected=',r.stdout)
        self.assertRegex(r.stdout,r'PREDICT version=1600 values=0\.3(?:\b|\n)')
        self.assertIn('integrated_result.json',reports)
        self.assertEqual(reports['integrated_result.json']['counts']['train'],1600)
        self.assertEqual(reports['integrated_result.json']['counts']['changes'],0)
        self.assertEqual(reports['integrated_result.json']['version'],1600)
    @unittest.skipUnless(shutil.which('clang++'),'clang unavailable')
    def test_integrated_native_clang(self):
        r,_=self.native(PROGRAM,compiler='clang++',name='clang')
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(r.stdout.count('VALIDATE PASS'),4)
    def test_failed_validation_stops_pipeline(self):
        bad=PROGRAM.replace('target [0.3] tolerance 0.02','target [0.9] tolerance 0.02',1)
        r,reports=self.native(bad,name='invalid')
        self.assertNotEqual(r.returncode,0)
        self.assertIn('validation failed',r.stderr)
        self.assertNotIn('OPTIMIZE selected=',r.stdout)
        self.assertNotIn('integrated_result.json',reports)
    def test_optimizer_does_not_change_weights(self):
        source='''brain C {\n senses 2\n neurons hidden 32 relu\n neurons out 1 sigmoid\n init 7\n}\nobserve [1,1]\npredict\noptimize 50\npredict\nreport "same.json"'''
        r,reports=self.native(source,name='same')
        self.assertEqual(r.returncode,0,r.stderr)
        values=re.findall(r'^PREDICT version=\d+ values=([^\n]+)',r.stdout,re.M)
        self.assertEqual(len(values),2)
        self.assertEqual(values[0],values[1])
        self.assertEqual(reports['same.json']['version'],0)
        self.assertEqual(reports['same.json']['counts']['changes'],0)
    def test_evaluation_not_training(self):
        src='''brain C {\n senses 1\n neurons out 1 linear\n init 7\n}\nobserve [0]\nvalidate [0] target [0] tolerance 0\nreport "eval.json"'''
        r,reports=self.native(src,name='eval')
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(reports['eval.json']['counts']['train'],0)
        self.assertEqual(reports['eval.json']['version'],0)

if __name__=='__main__':unittest.main(verbosity=2)

class PersistentWeightsTests(unittest.TestCase):
    def test_roundtrip_two_native_executables(self):
        with tempfile.TemporaryDirectory(prefix='nl03_artifact_') as td:
            d=Path(td)
            for name,source in [('trainer',PROGRAM),('inference',(ROOT/'examples/inference.nir').read_text())]:
                f=d/(name+'.nir');f.write_text(source,encoding='utf-8')
                build=subprocess.run([sys.executable,str(ROOT/'compiler.py'),str(f),'-o',str(d/name)],capture_output=True,text=True,timeout=50)
                self.assertEqual(build.returncode,0,build.stderr)
            t=subprocess.run([str(d/'trainer')],cwd=d,capture_output=True,text=True,timeout=20)
            self.assertEqual(t.returncode,0,t.stderr)
            self.assertTrue((d/'trained.nlw').exists())
            self.assertIn('MODEL_SAVED',t.stdout)
            i=subprocess.run([str(d/'inference')],cwd=d,capture_output=True,text=True,timeout=20)
            self.assertEqual(i.returncode,0,i.stderr)
            self.assertIn('MODEL_LOADED',i.stdout)
            self.assertEqual(i.stdout.count('VALIDATE PASS'),2)
            report=json.loads((d/'inference_result.json').read_text())
            self.assertEqual(report['counts']['train'],0)
            self.assertEqual(report['version'],1)
            self.assertEqual(report['mode'],'FULL')
    def test_missing_and_truncated_model_rejected(self):
        src=(ROOT/'examples/inference.nir').read_text()
        with tempfile.TemporaryDirectory(prefix='nl03_badmodel_') as td:
            d=Path(td);f=d/'i.nir';f.write_text(src)
            b=subprocess.run([sys.executable,str(ROOT/'compiler.py'),str(f),'-o',str(d/'i')],capture_output=True,text=True,timeout=50)
            self.assertEqual(b.returncode,0,b.stderr)
            a=subprocess.run([str(d/'i')],cwd=d,capture_output=True,text=True)
            self.assertNotEqual(a.returncode,0)
            self.assertIn('model input must be a regular',a.stderr)
            (d/'trained.nlw').write_text('NLMODEL2\n2 1\n0\nanswer 2 1 0 0\n')
            a=subprocess.run([str(d/'i')],cwd=d,capture_output=True,text=True)
            self.assertNotEqual(a.returncode,0)
            self.assertIn('truncated model bias',a.stderr)
    def test_compiler_rejects_model_traversal(self):
        s='brain X {\n senses 1\n neurons out 1 linear\n}\nload_model "../secret.nlw"'
        with self.assertRaises(CompileError):parse(s)
    def test_save_and_load_masks_biases_int8(self):
        # Direct C++ exercise checks an int8-gated multi-layer network roundtrip.
        code=r'''#include "neural_runtime.hpp"
#include <cmath>
int main(){
  neural::Model a("A",2);a.layer("h",4,1,17,1);a.layer("o",1,2,19,0);
  a.observe({0.3,-0.5});a.learn({0.6},0.2);a.gate("h",0,0,false);
  a.infer(false);const auto expected=a.infer(false)[0];
  a.save_model("checkpoint.nlw");
  neural::Model b("A",2);b.layer("h",4,1,170,1);b.layer("o",1,2,190,0);
  b.load_model("checkpoint.nlw");b.observe({0.3,-0.5});
  if(b.layers[0].disabled!=1 || b.layers[0].enabled[0]!=0)return 1;
  return std::abs(expected-b.infer(false)[0])<1e-12 ? 0 : 2;
}'''
        with tempfile.TemporaryDirectory(prefix='nl03_int8save_') as td:
            d=Path(td);(d/'q.cpp').write_text(code)
            b=subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT/'runtime'),str(d/'q.cpp'),'-o',str(d/'q')],capture_output=True,text=True,timeout=50)
            self.assertEqual(b.returncode,0,b.stderr)
            a=subprocess.run([str(d/'q')],cwd=d,capture_output=True,text=True,timeout=20)
            self.assertEqual(a.returncode,0,a.stdout+a.stderr)
