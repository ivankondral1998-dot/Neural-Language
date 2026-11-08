"""Independent regression tests from Grok's 0.2 handoff review.

Requires a C++17 compiler; no external network or ML libraries.
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compiler import CompileError, parse

CXX = r'''#include "neural_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
int main() {
    neural::Model m("int8",2);
    m.layer("h",4,1,37,1);m.layer("out",1,0,19,1);
    if(!m.layers[0].w.empty()||!m.layers[1].w.empty())return 10;
    if(m.layers[0].q.size()!=8||m.layers[1].q.size()!=4)return 11;
    m.observe({0.7,-0.1});m.mode=neural::Model::DELTA;
    m.infer(false);
    for(int i=0;i<50;i++) {
        auto& l=m.layers[i%2]; int r=i%l.out,c=i%l.in;
        m.change(l.name,r,c,(i%3 ? .001 : .035));
        double got=m.infer(false)[0];
        m.cache=false;double full=m.infer(false)[0];
        if(std::abs(got-full)>1e-10) {std::cerr<<"Divergence "<<i<<" "<<got<<" "<<full<<"\n";return 12;}
    }
    m.checkpoint("saved");
    double before=m.infer(false)[0];
    m.change("out",0,0,.2);m.infer(false);
    m.restore("saved");
    if(std::abs(m.infer(false)[0]-before)>1e-10)return 13;
    m.learn({0.8},0.5);
    double trained=m.infer(false)[0];
    if(!std::isfinite(trained))return 14;
    m.advise(30);
    for(const auto& l:m.layers) if(!l.w.empty())return 15;
    // Saturating int8 changes must stay defined (no floating-to-int overflow).
    m.mode=neural::Model::DELTA;
    m.change("out",0,0,10000.0);
    double saturated_delta=m.infer(false)[0];
    m.cache=false;double saturated_full=m.infer(false)[0];
    if(std::abs(saturated_delta-saturated_full)>1e-10)return 17;
    std::cout<<"INT8_SHADOW_FREE_AND_DELTA_EQUAL_PASS\n";
    // A single quantization-null edit must not fake a delta in a one-layer net.
    neural::Model n("single",1);n.layer("out",1,0,17,1);
    n.observe({1.0});n.mode=neural::Model::DELTA;n.infer(false);
    n.change("out",0,0,.001);
    double inc=n.infer(false)[0];n.cache=false;double fresh=n.infer(false)[0];
    if(std::abs(inc-fresh)>1e-12)return 16;
    std::cout<<"QUANTIZATION_ROUNDING_DELTA_EQUAL_PASS\n";
}'''


class FixedCompilerTests(unittest.TestCase):
    def test_budget_includes_gate_and_cache(self):
        src='''brain Test {\n senses 2048\n neurons h 16 relu int8\n budget 65536 200000\n}\nobserve [0,0]\npredict'''
        # The failure is in budget estimation, before vector shape validation.
        with self.assertRaisesRegex(CompileError, 'budget exceeded'):
            parse(src)

    def test_budget_explicit_forward_macs(self):
        src='''brain Test {\n senses 2\n neurons h 4 relu int8\n neurons output 1 sigmoid\n budget 65536 11\n}\nobserve [0, 0]\npredict'''
        with self.assertRaisesRegex(CompileError, 'budget exceeded'):
            parse(src)
        ir=parse(src.replace('65536 11','65536 12'))
        self.assertEqual(ir['brain']['macs'],12)

    def test_byte_counts_and_ir_types(self):
        src='''brain Test {\n senses 2\n neurons hidden 4 relu int8\n neurons output 1 sigmoid\n budget 65536 200000\n}\nobserve [0, 0]\npredict'''
        ir=parse(src)
        self.assertEqual(ir['brain']['weight_bytes'],8 + 4*8)
        # Includes gates, biases, a/z cache and input.
        self.assertEqual(ir['brain']['storage_bytes'],16 + 8*2 + 4*24 + 4*9 + 1*24)

    def test_float_only_unaffected(self):
        src='''brain F {\n senses 2\n neurons out 1 linear\n}\nobserve [0,1]\npredict'''
        ir=parse(src)
        self.assertEqual(ir['brain']['weight_bytes'],16)
        self.assertEqual(ir['brain']['storage_bytes'],16+2*9+24)


class NativeInt8Tests(unittest.TestCase):
    def test_cpp_int8_no_shadow_numerical_and_checkpoint(self):
        cxx = shutil.which('g++') or shutil.which('clang++')
        if not cxx:self.skipTest('No C++17 compiler')
        with tempfile.TemporaryDirectory() as td:
            src=pathlib.Path(td)/'test.cpp';out=pathlib.Path(td)/'test'
            src.write_text(CXX)
            build=subprocess.run([cxx,'-std=c++17','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT/'runtime'),str(src),'-o',str(out)],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            ran=subprocess.run([str(out)],cwd=td,capture_output=True,text=True)
            self.assertEqual(ran.returncode,0,ran.stderr + '\n'+ran.stdout)
            self.assertIn('INT8_SHADOW_FREE_AND_DELTA_EQUAL_PASS',ran.stdout)
            self.assertIn('QUANTIZATION_ROUNDING_DELTA_EQUAL_PASS',ran.stdout)


if __name__=='__main__':unittest.main(verbosity=2)
