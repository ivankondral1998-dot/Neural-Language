"""Contract tests: mandatory input state in neural programs, no changes to runtime."""
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from compiler import CompileError, parse, emit

BRAIN='brain T {\n senses 1\n neurons o 1 linear\n init 7\n}\n'

class StaticInputStateTests(unittest.TestCase):
    def must_reject(self, program, offending, reason='observe'):
        with self.assertRaisesRegex(CompileError, reason) as error:
            parse(BRAIN+program)
        self.assertIn('line ',str(error.exception))
        self.assertIn(offending,str(error.exception))

    def test_first_predict_requires_input(self):
        self.must_reject('predict\n','predict')

    def test_learn_expect_verify_optimize_advise_require_input(self):
        for stmt in ('learn [1] rate 0.1','expect [1] tolerance 0.1',
                     'verify 2','optimize 20','advise 10'):
            with self.subTest(stmt=stmt):
                self.must_reject(stmt+'\n',stmt.split()[0])

    def test_isolated_validation_does_not_bind_input(self):
        self.must_reject('validate [1] target [0] tolerance 1\npredict\n','predict')

    def test_load_model_does_not_bind_input(self):
        self.must_reject('load_model "trained.nlw"\npredict\n','predict')

    def test_cycle_nested_requires_input_at_first_iteration(self):
        self.must_reject('cycle 2 {\n cycle 3 {\n learn [0] rate 0.1\n }\n}\n','learn')

    def test_rewind_restores_absence_of_input(self):
        self.must_reject('remember cold\nobserve [0]\nrewind cold\npredict\n','predict')

    def test_valid_checkpoint_restores_input(self):
        ir=parse(BRAIN+'observe [1]\nremember warm\nlearn [0] rate 0.1\nrewind warm\npredict\n')
        self.assertEqual(ir['operations'][-1]['op'],'predict')

    def test_loop_observe_binds_input_for_later_and_next(self):
        ir=parse(BRAIN+'cycle 4 {\n observe [0]\n learn [0] rate 0.1\n}\npredict\n')
        self.assertIn('brain.learn',emit(ir))

    def test_read_only_and_mutations_without_observe_are_valid(self):
        ir=parse(BRAIN+'inspect\nweight o 0 0 add 0.1\nedge o 0 0 off\n'
                 'remember empty\nreport "r.json"\nvalidate [1] target [0] tolerance 1\n')
        self.assertEqual(len(ir['operations']),6)

    def test_restore_within_nested_loop_updates_outer_flow(self):
        self.must_reject('remember cold\ncycle 2 {\n observe [1]\n rewind cold\n}\npredict\n','predict')

    def test_ir_contains_explicit_semantic_contract(self):
        ir=parse(BRAIN+'observe [0]\npredict\n')
        self.assertEqual(ir['ir_version'],'NeuralIR/0.2')
        self.assertEqual(set(ir),{'ir_version','brain','operations','estimated_max_operations','semantic_contract','shape_contract'})
        self.assertEqual(ir['semantic_contract']['version'], 'NeuralEffects/1')

if __name__=='__main__': unittest.main()
