"""Independent numerical and error tests for Neural Language mobile inference."""
import math
import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'language' / 'mobile_infer.cpp'


def write_model(path, inputs, definitions):
    """Produce exact NLMODEL2 format, independent of C++ inference code."""
    parts = ['NLMODEL2', f'{inputs} {len(definitions)}', '0']
    width = inputs
    oracle = []
    for k, (out, act, storage, biases, values, gates) in enumerate(definitions):
        assert len(biases) == out and len(values) == out*width and len(gates) == len(values)
        weights = [v * (1/127) for v in values] if storage else values
        parts += [f'L{k} {width} {out} {act} {storage}',
                  ' '.join(format(x, '.17g') for x in biases),
                  ' '.join(format(x, '.17g') for x in weights),
                  ' '.join(map(str, gates))]
        oracle.append((width, out, act, biases, weights, gates))
        width = out
    path.write_text('\n'.join(parts+['END','']), encoding='utf-8')

    def reference(inp):
        data = inp
        for width, out, act, biases, weights, gates in oracle:
            result=[]
            for r in range(out):
                z = biases[r]
                for c in range(width):
                    z += weights[r*width+c] * gates[r*width+c] * data[c]
                if act==1: z=max(0,z)
                if act==2: z=1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
                result.append(z)
            data=result
        return data
    return reference


class MobileInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='neural_mobile_')
        cls.dir = Path(cls.tmp.name)
        cls.compilers = {}
        for compiler in ('g++', 'clang++'):
            if not shutil.which(compiler):
                continue
            out = cls.dir / ('clang' if 'clang' in compiler else 'gcc')
            result = subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                                     str(SOURCE), '-o', str(out)], capture_output=True, text=True, timeout=90)
            if result.returncode: raise AssertionError(result.stderr)
            cls.compilers[compiler]=out
        if not cls.compilers: raise unittest.SkipTest('no C++17 compiler')

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls,'tmp'): cls.tmp.cleanup()

    def run_model(self, binary, path, x, *args):
        return subprocess.run([str(binary), '--model', str(path), '--input', ','.join(map(str,x)), *args],
                              capture_output=True, text=True, timeout=8)

    def test_mixed_int8_float_masks_and_activations(self):
        path = self.dir/'mixed.nlw'
        reference = write_model(path, 2, [
            (3,1,1,[0.2,0.1,-0.3],[7,-43,127,-10,20,30],[1,0,1,1,1,0]),
            (1,2,0,[0.04],[0.3,-0.7,0.5],[1,1,0])])
        for compiler,binary in self.compilers.items():
            for x in ([0,0],[1.2,-0.3],[-4,2],[0.2,0.1]):
                with self.subTest(compiler=compiler,vector=x):
                    result=self.run_model(binary,path,x,'--stats')
                    self.assertEqual(result.returncode,0,result.stderr)
                    got=[float(y) for y in result.stdout.split('values=')[1].splitlines()[0].split(',')]
                    self.assertAlmostEqual(got[0],reference(x)[0],delta=1e-12)
                    self.assertIn('payload_not_process_RSS',result.stdout)

    def test_100_random_deterministic_networks_gcc_and_clang(self):
        rng=random.Random(20261108)
        for trial in range(100):
            inp=rng.randint(1,6)
            definitions=[]
            width=inp
            for _ in range(rng.randint(1,4)):
                out=rng.randint(1,9)
                storage=rng.choice((0,1))
                weights=[rng.randint(-127,127) if storage else rng.uniform(-1,1) for _ in range(out*width)]
                definitions.append((out,rng.choice((0,1,2)),storage,
                                    [rng.uniform(-0.5,0.5) for _ in range(out)],weights,
                                    [rng.choice((0,1)) for _ in weights]))
                width=out
            path=self.dir/f'test_{trial}.nlw'
            reference=write_model(path,inp,definitions)
            x=[rng.uniform(-2,2) for _ in range(inp)]
            for compiler,binary in self.compilers.items():
                with self.subTest(trial=trial,compiler=compiler):
                    result=self.run_model(binary,path,x)
                    self.assertEqual(result.returncode,0,result.stderr)
                    got=[float(v) for v in result.stdout.split('values=')[1].strip().split(',')]
                    expected=reference(x)
                    self.assertEqual(len(got),len(expected))
                    for a,b in zip(got,expected):
                        self.assertAlmostEqual(a,b,delta=1e-12*(1+abs(b)))

    def test_budgets_bad_checkpoint_and_input(self):
        binary=next(iter(self.compilers.values()))
        model=self.dir/'broken.nlw'
        write_model(model,2,[(1,0,0,[0.2],[0.4,-0.1],[1,1])])
        original=model.read_text()
        self.assertEqual(self.run_model(binary,model,[1,2],'--max-tensor-bytes','4096').returncode,0)
        for arg in [('--max-tensor-bytes','1'),('--max-tensor-bytes','-1'),('--max-tensor-bytes','garbage')]:
            self.assertNotEqual(self.run_model(binary,model,[1,2],*arg).returncode,0)
        for data in ([1],[1,float('nan')],[1,float('inf')],[1,2,3]):
            self.assertNotEqual(self.run_model(binary,model,data).returncode,0)
        replacements=[('NLMODEL2','NLMODEL3'),('2 1\n','2 9\n'),
                      ('L0 2 1 0 0','L0 4 1 0 0'),('END\n','END\nEXTRA\n'),
                      ('0.40000000000000002','nan'),('1 1\nEND','2 1\nEND')]
        for old,new in replacements:
            with self.subTest(mutation=new):
                model.write_text(original.replace(old,new,1))
                self.assertNotEqual(self.run_model(binary,model,[1,2]).returncode,0)
        model.write_text(original)

    def test_native_c_abi_with_python_ctypes(self):
        import ctypes
        if not shutil.which('g++'): self.skipTest('g++ required')
        shared=self.dir/'libneural_mobile.so'
        cmd=['g++','-std=c++17','-O2','-shared','-fPIC','-DNL_MOBILE_LIBRARY_ONLY',
             str(SOURCE),'-o',str(shared)]
        built=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
        self.assertEqual(built.returncode,0,built.stderr)
        lib=ctypes.CDLL(str(shared))
        lib.nlmobile_open.argtypes=[ctypes.c_char_p,ctypes.c_size_t]
        lib.nlmobile_open.restype=ctypes.c_void_p
        lib.nlmobile_close.argtypes=[ctypes.c_void_p]
        lib.nlmobile_input_count.argtypes=[ctypes.c_void_p]
        lib.nlmobile_input_count.restype=ctypes.c_size_t
        lib.nlmobile_output_count.argtypes=[ctypes.c_void_p]
        lib.nlmobile_output_count.restype=ctypes.c_size_t
        lib.nlmobile_predict.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),
                                      ctypes.c_size_t,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t]
        lib.nlmobile_predict.restype=ctypes.c_int
        lib.nlmobile_last_error.restype=ctypes.c_char_p
        path=self.dir/'cabi.nlw'
        reference=write_model(path,2,[(1,2,0,[0.15],[0.4,-0.7],[1,1])])
        handle=lib.nlmobile_open(str(path).encode(),4096)
        self.assertTrue(handle,lib.nlmobile_last_error())
        try:
            self.assertEqual(lib.nlmobile_input_count(handle),2)
            self.assertEqual(lib.nlmobile_output_count(handle),1)
            x=(ctypes.c_double*2)(1.0,0.5)
            result=(ctypes.c_double*1)()
            self.assertEqual(lib.nlmobile_predict(handle,x,2,result,1),1)
            self.assertAlmostEqual(result[0],reference([1,0.5])[0],delta=1e-13)
            self.assertEqual(lib.nlmobile_predict(handle,x,2,result,0),-1)
            self.assertIn(b'buffer',lib.nlmobile_last_error())
        finally: lib.nlmobile_close(handle)
        self.assertFalse(lib.nlmobile_open(str(path).encode(),1))

    def test_embedding_without_cli_main(self):
        if not shutil.which('g++'): self.skipTest('g++ required')
        src=self.dir/'embed.cpp'
        src.write_text('#define NL_MOBILE_LIBRARY_ONLY\n#include "'+str(SOURCE)+'"\n'
                       'int main(){return nlmobile::parse_csv("1,2").size()==2?0:1;}\n')
        exe=self.dir/'embed'
        build=subprocess.run(['g++','-std=c++17','-O2',str(src),'-o',str(exe)],capture_output=True,text=True,timeout=30)
        self.assertEqual(build.returncode,0,build.stderr)
        self.assertEqual(subprocess.run([str(exe)],capture_output=True).returncode,0)

if __name__=='__main__': unittest.main(verbosity=2)
