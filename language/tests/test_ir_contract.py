"""Neural IR static-effects contract: compile-time rejection, schema and native parity."""
import sys
import unittest
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compiler import parse, emit, CompileError

B = 'brain Contract {\n senses 1\n neurons output 1 linear\n init 31\n}\n'


class SnapshotContractTests(unittest.TestCase):
    def test_unknown_rewind_rejected_before_runtime(self):
        with self.assertRaisesRegex(CompileError, r'line 6: unknown checkpoint.*missing'):
            parse(B + 'rewind missing\n')

    def test_forward_rewind_rejected(self):
        with self.assertRaisesRegex(CompileError, 'unknown checkpoint'):
            parse(B + 'rewind later\nremember later\n')

    def test_duplicate_checkpoint_rejected(self):
        with self.assertRaisesRegex(CompileError, 'checkpoint already exists'):
            parse(B + 'remember original\nremember original\n')

    def test_repeated_cycle_checkpoint_rejected(self):
        with self.assertRaisesRegex(CompileError, 'cycle.*checkpoint|checkpoint.*cycle'):
            parse(B + 'cycle 2 {\n remember saved\n}\n')

    def test_nested_repeated_cycle_checkpoint_rejected(self):
        with self.assertRaisesRegex(CompileError, 'cycle.*checkpoint|checkpoint.*cycle'):
            parse(B + 'cycle 1 {\n cycle 3 {\n remember saved\n }\n}\n')

    def test_single_cycle_checkpoint_allowed(self):
        ir = parse(B + 'cycle 1 {\n observe [1]\n remember original\n}\nrewind original\npredict\n')
        self.assertIn('original', ir['semantic_contract']['checkpoints'])

    def test_checkpoint_predefined_outside_repeated_cycle_allowed(self):
        ir = parse(B + 'observe [1]\nremember original\ncycle 4 {\n rewind original\n predict\n}\n')
        self.assertTrue(ir['semantic_contract']['final_state']['input_bound'])

    def test_checkpoint_count_bound(self):
        ok = B + ''.join(f'remember c{i}\n' for i in range(64))
        ir = parse(ok)
        self.assertEqual(ir['semantic_contract']['snapshot_count'], 64)
        with self.assertRaisesRegex(CompileError, 'checkpoint limit'):
            parse(ok + 'remember c64\n')

    def test_input_lost_after_first_loop_iteration_rejected(self):
        with self.assertRaisesRegex(CompileError, 'predict requires observe'):
            parse(B + 'remember cold\nobserve [1]\ncycle 2 {\n predict\n rewind cold\n}\n')

    def test_restore_cold_checkpoint_invalidates_input(self):
        with self.assertRaisesRegex(CompileError, 'requires observe'):
            parse(B + 'remember cold\nobserve [1]\nrewind cold\npredict\n')


class EffectContractTests(unittest.TestCase):
    def test_contract_emitted_in_ir_without_changing_native_translation(self):
        ir = parse(B + 'observe [1]\nremember s\nweight output 0 0 add 0.2\n'
                   'validate [1] target [0] tolerance 1\nrewind s\npredict\n')
        c = ir['semantic_contract']
        self.assertEqual(c['version'], 'NeuralEffects/1')
        self.assertEqual(c['final_state']['input_bound'], True)
        self.assertEqual(c['checkpoints'], {'s': {'input_bound': True}})
        self.assertIn('model_parameters', c['effects']['weight']['writes'])
        self.assertNotIn('current_input', c['effects']['validate']['writes'])
        self.assertIn('brain.infer();', emit(ir))

    def test_codegen_rechecks_operations_after_ir_mutation(self):
        ir = parse(B + 'observe [1]\npredict\n')
        ir['operations'].insert(0, {'op':'predict','line':6})
        with self.assertRaisesRegex(CompileError, 'requires observe'):
            emit(ir)

    def test_codegen_rejects_tampered_effect_metadata(self):
        ir = parse(B + 'observe [1]\npredict\n')
        ir['semantic_contract']['final_state']['input_bound'] = False
        with self.assertRaisesRegex(CompileError, 'contract mismatch'):
            emit(ir)

    def test_file_effects_are_explicit_and_load_does_not_bind_input(self):
        ir = parse(B + 'load_model "x.nlw"\nsave_model "y.nlw"\nreport "r.json"\n')
        c = ir['semantic_contract']
        self.assertFalse(c['final_state']['input_bound'])
        self.assertEqual(c['effects']['load_model']['filesystem'], 'read')
        self.assertEqual(c['effects']['save_model']['filesystem'], 'write')
        self.assertEqual(c['effects']['report']['filesystem'], 'write')

    def test_loop_semantics_preserve_distinct_effect_counts(self):
        ir = parse(B + 'cycle 3 {\n observe [1]\n predict\n}\n')
        c = ir['semantic_contract']
        self.assertEqual(c['operation_counts']['observe'], 3)
        self.assertEqual(c['operation_counts']['predict'], 3)
        self.assertEqual(c['final_state']['input_bound'], True)



@unittest.skipUnless(shutil.which('g++'), 'g++ unavailable')
class NativeGuardTests(unittest.TestCase):
    def test_cpp_rejects_duplicate_and_missing_checkpoint_even_without_compiler(self):
        src = r'''
#include "neural_runtime.hpp"
#include <stdexcept>
int main() {
    neural::Model m("guard", 1);
    m.layer("out", 1, 0, 9);
    m.checkpoint("saved");
    bool duplicate = false, missing = false;
    try { m.checkpoint("saved"); } catch(const std::runtime_error&) { duplicate=true; }
    try { m.restore("not_here"); } catch(const std::runtime_error&) { missing=true; }
    return (duplicate && missing) ? 0 : 1;
}
'''
        with tempfile.TemporaryDirectory(prefix='nl_ir_runtime_') as td:
            d = Path(td)
            (d/'guard.cpp').write_text(src)
            proc = subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror',
                                   '-I', str(ROOT/'runtime'), str(d/'guard.cpp'),'-o',str(d/'guard')],
                                  capture_output=True, text=True, timeout=40)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            run = subprocess.run([str(d/'guard')], capture_output=True, text=True, timeout=5)
            self.assertEqual(run.returncode, 0, run.stderr)


class AbstractInterpreterParityTests(unittest.TestCase):
    def test_seeded_model_state_trace_matches_direct_semantics(self):
        # The generated traces use only deterministic input/snapshot commands.
        import random
        from compiler import validate_semantics
        rng=random.Random(849113)
        labels=['s0','s1','s2']
        kinds=['observe','predict','learn','remember','rewind','cycle']
        def generated(depth=0):
            out=[]
            for index in range(rng.randint(1,6)):
                kind=rng.choice(kinds if depth<2 else kinds[:-1])
                operation={'op':kind,'line':index+1}
                if kind in ('remember','rewind'):operation['name']=rng.choice(labels)
                if kind=='cycle':
                    operation['count']=rng.choice((1,2,3))
                    operation['body']=generated(depth+1)
                out.append(operation)
            return out
        def direct(program):
            bound=False;snapshots={}
            def walk(operations):
                nonlocal bound
                for operation in operations:
                    kind=operation['op']
                    if kind=='cycle':
                        for _ in range(operation['count']):walk(operation['body'])
                    elif kind=='observe':bound=True
                    elif kind in ('predict','learn'):
                        if not bound:raise RuntimeError('input')
                    elif kind=='remember':
                        if operation['name'] in snapshots or len(snapshots)>=64:raise RuntimeError('checkpoint')
                        snapshots[operation['name']]=bound
                    elif kind=='rewind':
                        if operation['name'] not in snapshots:raise RuntimeError('checkpoint')
                        bound=snapshots[operation['name']]
            walk(program)
        for sample in range(1000):
            program=generated()
            try:validate_semantics(program);compiled=True
            except CompileError:compiled=False
            try:direct(program);executable=True
            except RuntimeError:executable=False
            self.assertEqual(compiled,executable, f'trace {sample}: {program}')

if __name__ == '__main__':
    unittest.main()
