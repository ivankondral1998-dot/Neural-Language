"""mobile-cpu profile: IR plan, fallback, and FP64 agreement with generic FULL."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compiler import CompileError, emit, make_mobile_plan, parse

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

    def test_int8_and_graph_rejected(self):
        int8 = 'brain Q {\n senses 2\n neurons h 3 relu int8\n neurons o 1 linear\n}\nobserve [0, 0]\npredict\n'
        graph = 'brain G {\n senses 2\n neurons a 2 relu from input\n neurons b 1 linear from input a\n}\nobserve [0, 1]\npredict\n'
        with self.assertRaises(CompileError):
            make_mobile_plan(parse(int8))
        with self.assertRaises(CompileError):
            make_mobile_plan(parse(graph))


class PlanNative(unittest.TestCase):
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
