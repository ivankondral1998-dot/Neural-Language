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

    def test_int8_delta_follows_stored_quantized_weight(self):
        """Sub-quantum requests must not move DELTA off the stored INT8 weight.

        The previous source-mutation canary corrupted a local `updated` value and
        expected verify() to fail. That local no longer feeds the numeric path:
        both FULL and the incremental suffix recompute from stored effective
        weights. A direct oracle now guards the same contract.
        """
        if not self.cxx:self.skipTest('C++ compiler absent')
        header=(ROOT/'runtime/neural_runtime.hpp').read_text()
        self.assertEqual(header.count('const double updated=L.effective(ix);'),1)
        cxx=r'''#include "neural_runtime.hpp"
#include <cstdint>
#include <cstring>
#include <iostream>
int bits_differ(double x,double y){
 uint64_t a,b; static_assert(sizeof(double)==8,"double");
 std::memcpy(&a,&x,8); std::memcpy(&b,&y,8); return a!=b;
}
int main(){
 neural::Model d("q",1); d.mode=neural::Model::DELTA;
 d.layer("o",1,0,1,1);
 d.layers[0].q[0]=10;
 d.observe({0.5}); d.infer(false);
 const double scale=d.layers[0].scale;
 const auto q_before=d.layers[0].q;
 d.change("o",0,0,scale*0.1);
 if(d.layers[0].q!=q_before){std::cerr<<"subquantum changed code\n";return 2;}
 double y=d.infer(false)[0];
 neural::Model f("f",1); f.mode=neural::Model::FULL;
 f.layer("o",1,0,1,1);
 f.layers[0].q=q_before; f.layers[0].enabled=d.layers[0].enabled; f.layers[0].b=d.layers[0].b;
 f.observe({0.5});
 if(bits_differ(y,f.infer(false)[0])){std::cerr<<"subquantum delta diverged\n";return 3;}
 d.change("o",0,0,scale*3.0);
 const double stored=d.layers[0].effective(0);
 double oracle=d.layers[0].b[0]+stored*0.5;
 if(bits_differ(d.infer(false)[0],oracle)){std::cerr<<"quantized oracle mismatch\n";return 4;}
 if(d.counters.delta==0){std::cerr<<"delta path not used\n";return 5;}
 return 0;
}'''
        with tempfile.TemporaryDirectory() as td:
            p=pathlib.Path(td);(p/'q.cpp').write_text(cxx)
            build=subprocess.run([self.cxx,'-std=c++17','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT/'runtime'),str(p/'q.cpp'),'-o',str(p/'q')],capture_output=True,text=True,timeout=45)
            self.assertEqual(build.returncode,0,build.stderr)
            ran=subprocess.run([str(p/'q')],capture_output=True,text=True,timeout=15)
            self.assertEqual(ran.returncode,0,ran.stdout+'\n'+ran.stderr)


if __name__=='__main__':
    unittest.main()
