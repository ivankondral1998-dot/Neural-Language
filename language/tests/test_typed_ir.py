"""Regression tests for NeuralIR/0.2 type/shape verification at codegen boundary."""
from copy import deepcopy
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compiler import CompileError, emit, parse

SRC = '''brain TypedDemo {
 senses 2
 neurons hidden 3 relu int8
 neurons output 1 linear
 init 31
 budget 65536 128
}
observe [1, 0.5]
learn [0.25] rate 0.1
predict
weight hidden 0 1 add 0.2
edge output 0 2 off
validate [0, 1] target [0.4] tolerance 0.01
save_model "weights.nlw"
'''


def model():
    return parse(SRC)


def invalid(ir, expected):
    with unittest.TestCase().assertRaisesRegex(CompileError, expected):
        emit(ir)


class NeuralShapeContractTests(unittest.TestCase):
    def test_additive_shape_contract_in_ir(self):
        ir = model()
        self.assertEqual(ir['shape_contract']['version'], 'NeuralShapes/1')
        self.assertEqual(ir['shape_contract']['input'], {'dtype': 'f64', 'shape': [2]})
        self.assertEqual(ir['shape_contract']['output'], {'dtype': 'f64', 'shape': [1]})
        self.assertEqual(ir['shape_contract']['layers'][0]['weight_shape'], [3, 2])
        self.assertEqual(ir['shape_contract']['layers'][1]['weight_shape'], [1, 3])
        self.assertIn('brain.layer(', emit(ir))

    def test_legacy_ir_without_shape_metadata_remains_valid(self):
        ir = model()
        original = emit(ir)
        del ir['shape_contract']
        self.assertEqual(emit(ir), original)

    def test_json_roundtrip_preserves_emission(self):
        ir = model()
        self.assertEqual(emit(json.loads(json.dumps(ir))), emit(ir))

    def test_mismatch_declared_shape_metadata_rejected(self):
        ir = model()
        ir['shape_contract']['layers'][1]['weight_shape'][0] = 900
        invalid(ir, 'shape contract mismatch')

    def test_mutated_input_vector_rejected_before_native(self):
        ir = model()
        ir['operations'][0]['values'] = [1.0]
        invalid(ir, 'observe.*shape')

    def test_mutated_training_target_rejected_before_native(self):
        ir = model()
        ir['operations'][1]['values'] = [0.1, 0.2]
        invalid(ir, 'learn.*shape')

    def test_mutated_expect_shape_rejected(self):
        ir = parse(SRC + 'expect [0.1] tolerance 1\n')
        ir['operations'][-1]['values'] = [0.1, 0.2]
        invalid(ir, 'expect.*shape')

    def test_mutated_validate_vectors_rejected(self):
        ir = model()
        ir['operations'][5]['input'] = [1.0]
        invalid(ir, 'validate.*shape')
        ir = model()
        ir['operations'][5]['target'] = [0.0, 0.1]
        invalid(ir, 'validate.*shape')

    def test_invalid_scalar_nonfinite_and_bool_rejected(self):
        for value in (math.nan, math.inf, -math.inf, True, '0.1', 1e10, 10**10000):
            with self.subTest(value=value):
                ir = model()
                ir['operations'][1]['rate'] = value
                invalid(ir, 'rate')

    def test_invalid_float_in_weights_or_inputs_rejected(self):
        ir = model()
        ir['operations'][0]['values'][0] = math.nan
        invalid(ir, 'observe.*numeric')
        ir = model()
        ir['operations'][3]['delta'] = math.inf
        invalid(ir, 'weight.*delta')

    def test_weight_index_and_layer_revalidated(self):
        ir = model()
        ir['operations'][3]['row'] = 999
        invalid(ir, 'weight.*index')
        ir = model()
        ir['operations'][3]['layer'] = 'absent'
        invalid(ir, 'weight.*layer')
        ir = model()
        ir['operations'][4]['col'] = 999
        invalid(ir, 'edge.*index')
        ir = model()
        ir['operations'][4]['enabled'] = 1
        invalid(ir, 'edge enabled')

    def test_changed_brain_shape_and_aggregate_stats_rejected(self):
        ir = model()
        ir['brain']['neurons'][0]['size'] = 0
        invalid(ir, 'size')
        ir = model()
        ir['brain']['neurons'][0]['size'] = 4
        invalid(ir, '(weights|tensor|shape|storage|MAC)')
        ir = model()
        ir['brain']['weights'] = 5000
        invalid(ir, 'weights')

    def test_bad_model_identity_and_topology_rejected(self):
        ir = model()
        ir['brain']['name'] = 'unsafe\"); system(\"rm -rf\"'
        invalid(ir, 'identifier')
        ir = model()
        ir['brain']['neurons'][1]['name'] = 'hidden'
        invalid(ir, 'duplicate')
        ir = model()
        ir['brain']['senses'] = True
        invalid(ir, 'senses')
        ir = model()
        ir['brain']['neurons'][0]['activation'] = 'evil'
        invalid(ir, 'activation')
        ir = model()
        ir['brain']['neurons'][0]['storage'] = 'u1'
        invalid(ir, 'storage')

    def test_invalid_metadata_and_budget_rejected(self):
        ir = model()
        ir['ir_version'] = 'NeuralIR/0.99'
        invalid(ir, 'IR version')
        ir = model()
        ir['estimated_max_operations'] = 0
        invalid(ir, 'estimated_max_operations')
        ir = model()
        ir['brain']['budget'] = [1, 1]
        invalid(ir, 'budget')
        ir = model()
        ir['brain']['macs'] = 12
        invalid(ir, 'macs')

    def test_tampered_filename_and_mode_rejected(self):
        ir = model()
        ir['operations'][-1]['filename'] = '../escape.nlw'
        invalid(ir, 'filename')
        ir = model()
        ir['operations'][-1]['filename'] = 'inva\u202elid.nlw'
        invalid(ir, 'filename')
        ir = model()
        ir['operations'][2]['op'] = 'tune'
        ir['operations'][2]['mode'] = 'nonsense'
        invalid(ir, 'tune.*mode')

    def test_malformed_loop_structure_and_bounds_rejected(self):
        src = 'brain A {\n senses 1\n neurons out 1 linear\n}\ncycle 2 {\n observe [1]\n predict\n}\n'
        ir = parse(src)
        ir['operations'][0]['count'] = 0
        invalid(ir, 'cycle.*count')
        ir = parse(src)
        ir['operations'][0]['count'] = 100001
        invalid(ir, 'cycle.*count')
        ir = parse(src)
        ir['operations'][0]['body'] = 'not an operation array'
        invalid(ir, 'cycle.*body')

    def test_corrupt_line_numbers_and_numeric_types_rejected(self):
        ir = model()
        ir['operations'][0]['line'] = -1
        invalid(ir, 'line')
        ir = model()
        ir['operations'][0]['line'] = True
        invalid(ir, 'line')
        ir = model()
        ir['brain']['init'] = -1
        invalid(ir, 'init')

    def test_randomized_dense_shape_oracle(self):
        # Independent hand-derived dense-layer matrix dimensions, reproducible.
        import random
        rng = random.Random(20261108)
        for trial in range(350):
            senses = rng.randint(1, 8)
            count = rng.randint(1, 5)
            widths = [rng.randint(1, 12) for _ in range(count)]
            stores = [rng.choice(('float', 'int8')) for _ in range(count)]
            lines = [f'brain M{trial} {{', f' senses {senses}']
            for i, (width, storage) in enumerate(zip(widths, stores)):
                lines.append(f' neurons layer{i} {width} relu {storage}')
            lines.extend(['}', 'observe [' + ','.join(['0'] * senses) + ']',
                          'learn [' + ','.join(['0'] * widths[-1]) + '] rate 0.1'])
            ir = parse('\n'.join(lines))
            shapes = ir['shape_contract']
            prev = senses
            total = 0
            bytes_ = 8 * senses
            weight_bytes = 0
            for i, (width, storage) in enumerate(zip(widths, stores)):
                self.assertEqual(shapes['layers'][i]['weight_shape'], [width, prev])
                self.assertEqual(shapes['layers'][i]['input']['shape'], [prev])
                self.assertEqual(shapes['layers'][i]['output']['shape'], [width])
                total += prev * width
                element_size = 1 if storage == 'int8' else 8
                bytes_ += prev * width * (element_size + 1) + 24 * width
                weight_bytes += prev * width * element_size
                prev = width
            self.assertEqual(ir['brain']['weights'], total)
            self.assertEqual(ir['brain']['storage_bytes'], bytes_)
            self.assertEqual(ir['brain']['weight_bytes'], weight_bytes)
            self.assertIn('brain.learn(', emit(ir))
            # Adversarial mutation: one incorrect rank-1 target must be blocked.
            ir['operations'][-1]['values'].append(0.0)
            invalid(ir, 'learn.*shape')

    def test_bad_ir_value_types_fail_with_compile_error_not_python_exception(self):
        for bad in (None, {}, [], 0, 'wrong', {'op': ['predict']},
                    {'op': 'cycle', 'count': [], 'body': []},
                    {'op': 'observe', 'values': {'0': 1}}):
            ir = model()
            ir['operations'][0] = bad
            with self.subTest(bad=repr(bad)), self.assertRaises(CompileError):
                emit(ir)

    def test_unknown_ir_operation_rejected(self):
        ir = model()
        ir['operations'][2]['op'] = 'execute_os_command'
        invalid(ir, '(unsupported IR operation|operation)')


if __name__ == '__main__':
    unittest.main(verbosity=2)
