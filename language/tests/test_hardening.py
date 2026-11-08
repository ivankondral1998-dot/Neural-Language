import sys
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from compiler import parse, CompileError

class HardeningTests(unittest.TestCase):
    def test_report_newline_rejected(self):
        text='brain Test {\n senses 1\n neurons a 1 linear\n}\nreport "a\\nb.json"'
        with self.assertRaises(CompileError):parse(text)
    def test_report_bidi_rejected(self):
        text='brain Test {\n senses 1\n neurons a 1 linear\n}\nreport "safe\\u202Ejson.txt"'
        with self.assertRaises(CompileError):parse(text)
    def test_report_utf8_byte_limit(self):
        text='brain Test {\n senses 1\n neurons a 1 linear\n}\nreport "'+'ж'*41+'.json"'
        with self.assertRaises(CompileError):parse(text)
    def test_invalid_unicode_escape_is_compile_error(self):
        text='brain Test {\n senses 1\n neurons a 1 linear\n}\nreport "\\ud800"'
        with self.assertRaises(CompileError):parse(text)
    def test_very_large_integer_is_compile_error(self):
        text='brain Test {\n senses '+'9'*5000+'\n neurons a 1 linear\n}\npredict'
        with self.assertRaises(CompileError):parse(text)
    def test_report_safe_unicode(self):
        text='brain Test {\n senses 1\n neurons a 1 linear\n}\nreport "отчёт.json"'
        ir=parse(text)
        self.assertEqual(ir['operations'][0]['filename'],'отчёт.json')
    def test_dangling_symlink_report_not_followed(self):
        cpp='''#include "neural_runtime.hpp"
int main(){neural::Model m("test",1);m.layer("o",1,0,1);m.report("link.json");}'''
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);(d/'reports').mkdir();(d/'outside').mkdir()
            (d/'reports/link.json').symlink_to(d/'outside/escaped.json')
            src=d/'link.cpp';src.write_text(cpp);exe=d/'run'
            p=subprocess.run(['g++','-std=c++17','-O2','-I',str(ROOT/'runtime'),str(src),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            p=subprocess.run([str(exe)],cwd=d/'reports',capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertFalse((d/'outside/escaped.json').exists())
            self.assertTrue((d/'reports/link_1.json').is_file())
    def test_expect_nan_rejected_in_runtime(self):
        cpp='''#include "neural_runtime.hpp"
#include <limits>
int main(){
 neural::Model m("test",1);m.layer("out",1,0,42);m.observe({0});
 auto nan=std::numeric_limits<double>::quiet_NaN();
 try {m.expect({nan},0);return 1;}catch(const std::exception&){}
 try {m.expect({0},nan);return 2;}catch(const std::exception&){}
 try {m.expect({0},-1.0);return 3;}catch(const std::exception&){}
 m.expect({0},0);return 0;
}'''
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/'test.cpp';src.write_text(cpp)
            exe=Path(td)/'test'
            for extra in ([],['-DNDEBUG']):
                p=subprocess.run(['g++','-std=c++17','-O2',*extra,'-I',str(ROOT/'runtime'),str(src),'-o',str(exe)],capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stderr)
                p=subprocess.run([str(exe)],capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stdout+p.stderr)

if __name__=='__main__':unittest.main()
