"""Language-level acceptance: ordinary observe reuses unchanged neural state."""
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
from compiler import parse,emit


class AutomaticObserveReuse(unittest.TestCase):
    def test_no_new_grammar_and_existing_ir(self):
        sample=('brain Live {\n senses 2\n neurons out 1 linear\n init 11\n}\n'
                'observe [0.25,0.5]\npredict\nobserve [0.25,0.51]\npredict\n')
        ir=parse(sample)
        self.assertEqual([x['op'] for x in ir['operations']],['observe','predict','observe','predict'])
        self.assertNotIn('neural_changes',ir)  # old source/IR contract untouched
        self.assertIn('brain.observe(std::vector<double>',emit(ir))

    @unittest.skipUnless(shutil.which('g++') and shutil.which('clang++'),'requires GCC and Clang')
    def test_existing_nir_syntax_and_native_report(self):
        with tempfile.TemporaryDirectory(prefix='nl_observe_reuse_') as tmp:
            root=Path(tmp)
            senses=1024
            first=[0.2]*senses
            second=first.copy(); second[71]=0.201
            vectors=['['+','.join(map(str,v))+']' for v in (first,second)]
            before=('brain Sensor {\n senses 1024\n neurons hidden 128 relu\n'
                    ' neurons answer 16 linear\n init 23\n}\n')
            results={}
            for compiler in ('g++','clang++'):
                for mode in ('auto','full'):
                    name=f'{compiler}_{mode}'.replace('+','p')
                    program=(before+f'tune {mode}\nobserve {vectors[0]}\npredict\n'
                             f'observe {vectors[1]}\npredict\nobserve {vectors[1]}\npredict\n'
                             f'report "{name}.json"\n')
                    source=root/(name+'.nir'); source.write_text(program)
                    target=root/name
                    compilation=subprocess.run([sys.executable,str(ROOT/'compiler.py'),str(source),
                        '-o',str(target),'--compiler',compiler],text=True,capture_output=True,timeout=75)
                    self.assertEqual(compilation.returncode,0,compilation.stderr)
                    execution=subprocess.run([str(target)],cwd=root,text=True,capture_output=True,timeout=15)
                    self.assertEqual(execution.returncode,0,execution.stderr)
                    answers=[[float(z) for z in row.split(',')]
                             for row in re.findall(r'^PREDICT version=\d+ values=([^\n]+)',execution.stdout,re.M)]
                    self.assertEqual(len(answers),3)
                    report=json.loads((root/f'{name}.json').read_text())
                    results[compiler,mode]=(answers,report['counts'])
                auto,stats=results[compiler,'auto']
                full,_=results[compiler,'full']
                self.assertGreater(stats['input_delta'],0)
                self.assertEqual(auto[1],auto[2])
                for a,b in zip(auto,full):
                    for x,y in zip(a,b):self.assertAlmostEqual(x,y,delta=1e-9)

    @unittest.skipUnless(shutil.which('g++') and shutil.which('clang++'),'requires GCC and Clang')
    def test_independent_stress_both_compilers(self):
        with tempfile.TemporaryDirectory(prefix='nl_observe_cpp_') as tmp:
            for compiler in ('g++','clang++'):
                binary=Path(tmp)/('clang' if compiler=='clang++' else 'gcc')
                compile=subprocess.run([compiler,'-std=c++17','-O2','-Wall','-Wextra','-Werror',
                    '-I',str(ROOT/'runtime'),str(ROOT/'tests/observe_reuse_stress.cpp'),'-o',str(binary)],
                    text=True,capture_output=True,timeout=75)
                self.assertEqual(compile.returncode,0,compile.stderr)
                outcome=subprocess.run([str(binary)],text=True,capture_output=True,timeout=35)
                self.assertEqual(outcome.returncode,0,outcome.stderr)
                self.assertIn('OBSERVE_REUSE_PASS',outcome.stdout)

if __name__=='__main__': unittest.main(verbosity=2)
