"""Neural Language .nir verifier end-to-end and release safety regressions."""
import importlib.util
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('neural_compiler_verifier',ROOT/'compiler.py')
comp=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comp)

SAMPLE='''brain Example {
 senses 3
 neurons h 6 relu int8
 neurons out 2 sigmoid
 budget 16384 1000
 init 17
}
observe [1,0.5,-0.5]
remember saved
verify 180
predict
rewind saved
predict
'''


class VerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cxx=shutil.which('g++') or shutil.which('clang++')

    def test_parser_and_ir(self):
        ir=comp.parse(SAMPLE)
        self.assertEqual([x['op'] for x in ir['operations']],['observe','remember','verify','predict','rewind','predict'])
        self.assertIn('brain.verify(180);',comp.emit(ir))
        self.assertEqual(comp.analyze_ir(ir)['operations']['verify'],1)

    def test_reject_invalid_verify_limits(self):
        for value in ['0','1','2001','-1','xx']:
            with self.subTest(value=value):
                with self.assertRaises(comp.CompileError):
                    comp.parse(SAMPLE.replace('verify 180',f'verify {value}'))

    def test_nir_executes_and_preserves_outputs(self):
        if not self.cxx:self.skipTest('C++ compiler absent')
        with tempfile.TemporaryDirectory() as td:
            t=pathlib.Path(td);src=t/'verify.nir';exe=t/'verify'
            src.write_text(SAMPLE)
            cp=subprocess.run([sys.executable,str(ROOT/'compiler.py'),str(src),'-o',str(exe)],capture_output=True,text=True,timeout=45)
            self.assertEqual(cp.returncode,0,cp.stderr)
            done=subprocess.run([str(exe)],capture_output=True,text=True,timeout=25)
            self.assertEqual(done.returncode,0,done.stderr)
            self.assertIn('VERIFY PASS rounds=180',done.stdout)
            self.assertIn('delta_paths=',done.stdout)
            prediction=[line for line in done.stdout.splitlines() if line.startswith('PREDICT')]
            self.assertEqual(len(prediction),2)
            self.assertEqual(prediction[0].split('values=')[1],prediction[1].split('values=')[1])

    def test_release_build_rejects_nan_and_infinity(self):
        if not self.cxx:self.skipTest('C++ compiler absent')
        source=r'''#include "neural_runtime.hpp"
#include <limits>
#include <stdexcept>
int main(){
 neural::Model m("safety",1);
 m.layer("o",1,0,19,1);
 m.observe({1});m.infer(false);
 for(double value:{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()}){
   bool rejected=false;
   try {m.change("o",0,0,value);}catch(const std::runtime_error&) {rejected=true;}
   if(!rejected)return 1;
 }
 bool rejected=false;
 try {m.observe({std::numeric_limits<double>::quiet_NaN()});}catch(const std::runtime_error&) {rejected=true;}
 return rejected?0:2;
}'''
        with tempfile.TemporaryDirectory() as td:
            p=pathlib.Path(td);(p/'s.cpp').write_text(source)
            build=subprocess.run([self.cxx,'-std=c++17','-O2','-DNDEBUG','-Wall','-Wextra','-Werror','-I',str(ROOT/'runtime'),str(p/'s.cpp'),'-o',str(p/'s')],capture_output=True,text=True,timeout=45)
            self.assertEqual(build.returncode,0,build.stderr)
            ran=subprocess.run([str(p/'s')],capture_output=True,text=True,timeout=10)
            self.assertEqual(ran.returncode,0,ran.stderr)

    def test_mutation_detects_old_int8_rounding_defect(self):
        if not self.cxx:self.skipTest('C++ compiler absent')
        h=(ROOT/'runtime/neural_runtime.hpp').read_text()
        bug='const double updated=L.effective(ix);'
        self.assertEqual(h.count(bug),1)
        h=h.replace(bug,'const double updated=new_gate ? new_weight : 0.0;')
        cxx=r'''#include "neural_runtime.hpp"
#include <exception>
#include <iostream>
int main() {
 neural::Model m("corrupted",3);
 m.layer("h",6,1,31,1);m.layer("out",2,2,19,1);
 // The injected mutation affects only experimental raw-algebraic deltas.
 m.strict_numeric=false;
 m.observe({0.4,0.2,0.9});
 try {m.verify(250);}
 catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 10;}
 return 0;
}'''
        with tempfile.TemporaryDirectory() as td:
            p=pathlib.Path(td);(p/'neural_runtime.hpp').write_text(h);(p/'bug.cpp').write_text(cxx)
            build=subprocess.run([self.cxx,'-std=c++17','-O2','-Wall','-Wextra','-Werror','-I',str(p),str(p/'bug.cpp'),'-o',str(p/'bug')],capture_output=True,text=True,timeout=45)
            self.assertEqual(build.returncode,0,build.stderr)
            ran=subprocess.run([str(p/'bug')],capture_output=True,text=True,timeout=15)
            self.assertEqual(ran.returncode,10,ran.stdout+'\n'+ran.stderr)
            self.assertIn('verify numerical mismatch',ran.stderr)


if __name__=='__main__':
    unittest.main()
