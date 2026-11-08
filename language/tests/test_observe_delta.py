"""Neural Language observe_delta: compile-time semantics and native differential tests."""
import hashlib
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
from compiler import parse, emit, analyze_ir, CompileError

BRAIN='brain Sensor {\n senses 3\n neurons a 8 relu\n neurons o 2 sigmoid\n init 23\n}\n'

class CompilerDelta(unittest.TestCase):
    def test_ir_contract_and_output(self):
        src=BRAIN+'observe [0,0,0]\nobserve_delta 1 add -0.25\npredict\n'
        ir=parse(src)
        self.assertEqual(ir['neural_changes']['contract'],'NeuralChanges/1')
        self.assertEqual(ir['neural_changes']['occurrences'][0]['index'],1)
        self.assertEqual(ir['neural_changes']['topology'],'sequential_dense_only')
        self.assertEqual(ir['neural_changes']['dependencies'],[
            {'source':'input','target':'a'},{'source':'a','target':'o'}])
        self.assertIn('brain.change_input(1,-0.25);',emit(ir))
        self.assertEqual(analyze_ir(ir)['operations']['observe_delta'],1)
    def test_static_invalid_programs(self):
        cases=(
          (BRAIN+'observe_delta 0 add 0.1\npredict\n','preceding observe'),
          (BRAIN+'observe [0,0,0]\nobserve_delta 3 add 0.1\n','outside'),
          (BRAIN+'observe [0,0,0]\nobserve_delta 1 add nan\n','invalid number'),
          (BRAIN+'observe [0,0,0]\nremember x\nrewind x\nobserve_delta 1 add 0.1\n','preceding observe'),
          (BRAIN+'cycle 2 {\n observe_delta 0 add 0.1\n}\n','preceding observe'),
        )
        for source, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(CompileError,error):parse(source)
    def test_tampered_neural_ir_rejected_before_codegen(self):
        ir=parse(BRAIN+'observe [0,0,0]\nobserve_delta 1 add 0.1\npredict\n')
        cases=[('index',5),('delta',float('nan')),('delta',float('inf'))]
        for key,value in cases:
            edited=__import__('copy').deepcopy(ir)
            edited['operations'][1][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(CompileError):
                emit(edited)
        edited=__import__('copy').deepcopy(ir)
        edited['neural_changes']['dependencies'][0]['target']='wrong'
        with self.assertRaises(CompileError):emit(edited)
        edited=__import__('copy').deepcopy(ir)
        del edited['neural_changes']
        with self.assertRaises(CompileError):analyze_ir(edited)
        legacy=parse(BRAIN+'observe [0,0,0]\npredict\n')
        legacy['neural_changes']=ir['neural_changes']
        with self.assertRaises(CompileError):emit(legacy)

    def test_cycle_accepted_after_observation(self):
        ir=parse(BRAIN+'observe [0,0,0]\ncycle 4 {\nobserve_delta 0 add 0.01\npredict\n}\n')
        self.assertEqual(ir['estimated_max_operations'],9)
    def test_legacy_cpp_identical_to_frozen_golden(self):
        s=BRAIN+'observe [0.1,0.2,0.3]\npredict\n'
        cpp=emit(parse(s))
        self.assertNotIn('change_input(',cpp)
        self.assertNotIn('neural_changes',parse(s))
        self.assertEqual(hashlib.sha256(cpp.encode()).hexdigest(),'a0f8bd71c459a19bfcb5cb9968e6e7fd370bcae76845f35d555ffaa055e96479')

@unittest.skipUnless(shutil.which('g++') and shutil.which('clang++'),'both GCC and Clang required')
class NativeDelta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='nl_delta_')
        cls.dir=Path(cls.temp.name)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_gcc_and_clang_compiled_programs_equivalent(self):
        source=BRAIN+'observe [0.3,-0.2,0.5]\npredict\ntune incremental\n'
        full=BRAIN+'observe [0.3,-0.2,0.5]\npredict\ntune full\n'
        values=[0.3,-0.2,0.5]
        for i in range(34):
            idx=i%3; delta=(i%7-3)*0.001
            values[idx]+=delta
            source+=f'observe_delta {idx} add {delta:.17g}\npredict\n'
            full+='observe ['+','.join(format(v,'.17g') for v in values)+']\npredict\n'
        for compiler in ('g++','clang++'):
            for name,src in (('delta',source),('full',full)):
                file=self.dir/f'{name}_{compiler}.nir';file.write_text(src)
                output=self.dir/f'{name}_{compiler}'
                b=subprocess.run([sys.executable,str(ROOT/'compiler.py'),str(file),'-o',str(output),'--compiler',compiler],capture_output=True,text=True,timeout=55)
                self.assertEqual(b.returncode,0,b.stderr)
                r=subprocess.run([str(output)],capture_output=True,text=True,timeout=15)
                self.assertEqual(r.returncode,0,r.stderr)
                setattr(self,f'{name}_{compiler}',[list(map(float,match.split(','))) for match in re.findall(r'^PREDICT version=\d+ values=([^\n]+)',r.stdout,re.M)])
            a=getattr(self,f'delta_{compiler}');b=getattr(self,f'full_{compiler}')
            self.assertEqual(len(a),len(b))
            for step,(av,bv) in enumerate(zip(a,b)):
                with self.subTest(compiler=compiler,step=step):
                    for x,y in zip(av,bv):self.assertAlmostEqual(x,y,delta=1e-9)

    def test_gcc_clang_randomized_oracle_stress(self):
        src=ROOT/'tests/change_input_stress.cpp'
        for compiler in ('g++','clang++'):
            output=self.dir/f'change_stress_{compiler}'
            build=subprocess.run([compiler,'-std=c++17','-O2','-Wall','-Wextra','-Werror',
                                  '-I',str(ROOT/'runtime'),str(src),'-o',str(output)],capture_output=True,text=True,timeout=65)
            self.assertEqual(build.returncode,0,build.stderr)
            result=subprocess.run([str(output)],capture_output=True,text=True,timeout=25)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('CHANGE_INPUT_STRESS_PASS',result.stdout)

if __name__=='__main__':unittest.main(verbosity=2)
