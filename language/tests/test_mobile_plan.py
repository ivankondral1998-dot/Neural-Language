"""mobile-cpu plans: FP64 arithmetic, packed INT8 storage and generic fallback.

Run from language/: python3 -m unittest -v tests.test_mobile_plan
INT8 tests compile actual .nir and generated plans with strict FP64 arithmetic;
they do not establish native integer dot products or Android performance.
"""
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compiler import CompileError, emit, make_mobile_plan, mobile_function_source, parse

TINY = """brain Tiny {
  senses 2
  neurons hidden 4 tanh
  neurons out 2 linear
  init 5
}
observe [0.25, -0.5]
predict
edge hidden 1 0 off
predict
report "tiny.json"
"""

TRAIN = """brain Train {
  senses 2
  neurons hidden 4 relu
  neurons out 2 linear
  init 9
}
train_data "tiny_train.txt" samples 4 epochs 2 rate 0.05 seed 7
observe [0.1, -0.2]
predict
save_model "trained_mobile.nlw"
report "train.json"
"""

MIXED = """brain Mixed {
  senses 5
  neurons hidden 7 relu int8
  neurons middle 3 sigmoid
  neurons out 5 tanh int8
  init 11
}
tune full
observe [0.25, -0.5, 0.1, 0.3, -0.2]
predict
edge hidden 1 0 off
predict
edge hidden 1 0 on
weight out 0 1 add 0.02
predict
learn [0.1, -0.2, 0.3, -0.1, 0.2] rate 0.05
predict
save_model "mixed.nlw"
report "mixed.json"
"""


def compile_nir(source, directory, name, profile):
    path = directory / (name + '.nir')
    path.write_text(source, encoding='utf-8')
    command = [sys.executable, str(ROOT / 'compiler.py'), str(path), '-o', str(directory / name)]
    if profile != 'generic':
        command.extend(['--profile', profile])
    built = subprocess.run(command, text=True, capture_output=True, timeout=60)
    return built


class PlanIr(unittest.TestCase):
    def test_generic_emit_has_no_plan(self):
        cpp = emit(parse(TINY))
        self.assertNotIn('mobile_dense_ilp4', cpp)
        self.assertNotIn('mobile_forward', cpp)

    def test_mobile_plan_records_shapes_and_contract(self):
        ir = parse(TINY)
        ir['compute_plan'] = make_mobile_plan(ir)
        cpp = emit(ir)
        self.assertEqual(ir['compute_plan']['contract'], 'NeuralPlan/1')
        self.assertEqual(ir['compute_plan']['numeric'], 'fp64-strict-row-order')
        self.assertEqual(ir['compute_plan']['layers'][0]['in'], 2)
        self.assertEqual(ir['compute_plan']['layers'][0]['out'], 4)
        self.assertEqual(ir['compute_plan']['layers'][1]['activation'], 'linear')
        self.assertIn('neural::mobile_dense_ilp4<2,4,3>', cpp)
        self.assertIn('neural::mobile_dense_ilp4<4,2,0>', cpp)
        self.assertIn('brain.mobile_forward=&nl_mobile_try;', cpp)

    def test_mobile_dataset_uses_full_buffer_reuse(self):
        source = """brain Rows {
  senses 2
  neurons out 2 linear
}
load_model "weights.nlw"
tune full
predict_data "rows.txt" samples 2
"""
        generic = emit(parse(source))
        self.assertIn('neural::predict_data(brain,"rows.txt",2);', generic)
        self.assertNotIn('neural::predict_data(brain,"rows.txt",2,true);', generic)
        ir = parse(source)
        ir['compute_plan'] = make_mobile_plan(ir)
        mobile = emit(ir)
        self.assertIn('neural::predict_data(brain,"rows.txt",2,true);', mobile)

    def test_mixed_storage_plan_describes_fp64_dequantization(self):
        ir = parse(MIXED)
        try:
            plan = make_mobile_plan(ir)
        except CompileError as error:
            self.fail(f'mobile-cpu must support packed INT8 storage: {error}')
        self.assertEqual(plan['numeric'], 'fp64-strict-row-order')
        self.assertEqual(plan['layers'][0]['weight_order'], 'row-major-int8-dequant-fp64')
        self.assertEqual(plan['layers'][1]['weight_order'], 'row-major-fp64')
        self.assertIn('double(q) * scale', plan['layers'][0]['dequantization'])
        ir['compute_plan'] = plan
        cpp = emit(ir)
        self.assertIn('neural::mobile_dense_int8_ilp4<5,7,1>', cpp)
        self.assertIn('m.layers[0].q.data()', cpp)
        self.assertIn('m.layers[0].scale', cpp)
        self.assertIn('neural::mobile_dense_ilp4<7,3,2>', cpp)

    def test_graph_rejected(self):
        graph = 'brain G {\n senses 2\n neurons a 2 relu from input\n neurons b 1 linear from input a\n}\nobserve [0, 1]\npredict\n'
        with self.assertRaises(CompileError):
            make_mobile_plan(parse(graph))


class PlanNative(unittest.TestCase):
    def test_mixed_plan_native_numerics_and_fallback(self):
        ir = parse(MIXED)
        try:
            ir['compute_plan'] = make_mobile_plan(ir)
        except CompileError as error:
            self.fail(f'mobile-cpu must support packed INT8 storage: {error}')
        for cxx in ('g++', 'clang++'):
            if not shutil.which(cxx):
                continue
            with self.subTest(cxx=cxx), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'generated_mobile_plan.hpp').write_text(mobile_function_source(ir))
                exe = root / 'int8_plan'
                built = subprocess.run(
                    [cxx, '-std=c++17', '-O2', '-ffp-contract=off', '-Wall',
                     '-Wextra', '-Werror', '-DNEURAL_MOBILE_PLAN', '-I', str(root),
                     '-I', str(ROOT / 'runtime'),
                     str(ROOT / 'tests/runtime_mobile_int8_plan.cpp'), '-o', str(exe)],
                    capture_output=True, text=True, timeout=60)
                self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
                run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=20)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                self.assertIn('MOBILE_INT8_PLAN_PASS', run.stdout)

    def test_mixed_compiled_nir_matches_generic_weights_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = {}
            for name, profile in (('generic', 'generic'), ('mobile', 'mobile-cpu')):
                directory = Path(tmp) / name
                directory.mkdir()
                built = compile_nir(MIXED, directory, name, profile)
                self.assertEqual(built.returncode, 0, built.stderr)
                run = subprocess.run([str(directory / name)], cwd=directory,
                                     capture_output=True, text=True, timeout=20)
                self.assertEqual(run.returncode, 0, run.stderr)
                outputs[name] = (run.stdout, (directory / 'mixed.nlw').read_bytes())
                if name == 'mobile':
                    counts = json.loads((directory / 'mixed.json').read_text())['counts']
                    self.assertGreater(counts['mobile_plan'], 0)
                    self.assertGreater(counts['full'], counts['mobile_plan'])
            self.assertEqual(outputs['generic'], outputs['mobile'])

    def test_input_reuse_matches_full_without_reallocation(self):
        # This program tests the actual native runtime, independently of .nir
        # codegen, including bad-data rollback and stable input allocation.
        probe = ROOT / 'tests/runtime_mobile_input_reuse.cpp'
        for cxx in ('g++', 'clang++'):
            if not shutil.which(cxx):
                continue
            with self.subTest(cxx=cxx), tempfile.TemporaryDirectory() as tmp:
                exe = Path(tmp) / 'input_reuse'
                build = subprocess.run(
                    [cxx, '-std=c++17', '-O2', '-ffp-contract=off', '-Wall',
                     '-Wextra', '-Werror', '-I', str(ROOT / 'runtime'),
                     str(probe), '-o', str(exe)],
                    capture_output=True, text=True, timeout=60)
                self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
                run = subprocess.run([str(exe)], capture_output=True,
                                     text=True, timeout=20)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                self.assertIn('MOBILE_INPUT_REUSE_PASS', run.stdout)

    def test_predict_matches_generic_and_plan_runs(self):
        with tempfile.TemporaryDirectory(prefix='nl_mobile_') as temp:
            root = Path(temp)
            generic_dir = root / 'generic'
            mobile_dir = root / 'mobile'
            generic_dir.mkdir()
            mobile_dir.mkdir()
            generic = compile_nir(TINY, generic_dir, 'generic', 'generic')
            mobile = compile_nir(TINY, mobile_dir, 'mobile', 'mobile-cpu')
            self.assertEqual(generic.returncode, 0, generic.stderr)
            self.assertEqual(mobile.returncode, 0, mobile.stderr)
            g = subprocess.run([str(generic_dir / 'generic')], cwd=generic_dir, text=True, capture_output=True, timeout=20)
            m = subprocess.run([str(mobile_dir / 'mobile')], cwd=mobile_dir, text=True, capture_output=True, timeout=20)
            self.assertEqual(g.returncode, 0, g.stderr)
            self.assertEqual(m.returncode, 0, m.stderr)
            self.assertEqual(g.stdout, m.stdout)
            report = json.loads((mobile_dir / 'tiny.json').read_text(encoding='utf-8'))
            self.assertGreater(report['counts']['mobile_plan'], 0)
            generic_report = json.loads((generic_dir / 'tiny.json').read_text(encoding='utf-8'))
            self.assertNotIn('mobile_plan', generic_report['counts'])
            ir = json.loads((mobile_dir / 'mobile.nir.json').read_text(encoding='utf-8'))
            self.assertEqual(ir['compute_plan']['profile'], 'mobile-cpu')
            self.assertNotIn('compute_plan', json.loads((generic_dir / 'generic.nir.json').read_text(encoding='utf-8')))

    def test_mobile_dataset_prediction_matches_generic(self):
        source = """brain PredictRows {
  senses 3
  neurons hidden 4 tanh
  neurons out 2 linear
  init 7
}
tune full
predict_data "rows.txt" samples 4
report "rows_report.json"
"""
        rows = "0 0 0\n0.2 -0.1 0.5\n-0.3 0.4 -0.2\n0.2 -0.1 0.5\n"
        with tempfile.TemporaryDirectory(prefix='nl_reuse_data_') as tmp:
            outputs = {}
            for name, profile in (('generic', 'generic'), ('mobile', 'mobile-cpu')):
                directory = Path(tmp) / name
                directory.mkdir()
                (directory / 'rows.txt').write_text(rows, encoding='utf-8')
                build = compile_nir(source, directory, name, profile)
                self.assertEqual(build.returncode, 0, build.stderr)
                run = subprocess.run([str(directory / name)], cwd=directory,
                                     capture_output=True, text=True, timeout=20)
                self.assertEqual(run.returncode, 0, run.stderr)
                outputs[name] = run.stdout
                if name == 'mobile':
                    report = json.loads((directory / 'rows_report.json').read_text())
                    self.assertEqual(report['counts']['mobile_plan'], 4)
            self.assertEqual(outputs['generic'], outputs['mobile'])
            self.assertEqual(outputs['mobile'].count('DATA_PREDICT index='), 4)

    def test_training_weights_match_generic(self):
        rows = ['0 0.2 0.1', '1 -0.3 0.4', '0 0.5 -0.2', '1 0.1 0.3']
        with tempfile.TemporaryDirectory(prefix='nl_mobile_train_') as temp:
            root = Path(temp)
            outputs = {}
            for name, profile in (('generic', 'generic'), ('mobile', 'mobile-cpu')):
                directory = root / name
                directory.mkdir()
                (directory / 'tiny_train.txt').write_text('\n'.join(rows) + '\n', encoding='utf-8')
                built = compile_nir(TRAIN, directory, name, profile)
                self.assertEqual(built.returncode, 0, built.stderr)
                ran = subprocess.run([str(directory / name)], cwd=directory, text=True, capture_output=True, timeout=20)
                self.assertEqual(ran.returncode, 0, ran.stderr)
                outputs[name] = (ran.stdout, (directory / 'trained_mobile.nlw').read_bytes())
            self.assertEqual(outputs['generic'], outputs['mobile'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
