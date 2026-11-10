"""Contract tests for the bounded Neural Language code-improvement assistant."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assistant import ExperimentError, collect_training, load_evaluation, propose_branch, run_experiment
from compiler import CompileError, parse
from reference_eval import ReferenceError, read_model, predict

SOURCE = (ROOT/'examples/assistant_hinge_baseline.nir')
DATA = ROOT/'examples/assistant_hinge_evaluation.json'
LINEAR_SOURCE = ROOT/'examples/assistant_linear_baseline.nir'
LINEAR_DATA = ROOT/'examples/assistant_linear_evaluation.json'


class AssistantContractTests(unittest.TestCase):
    def test_model_proposal_is_real_graph_and_original_stays_unchanged(self):
        text=SOURCE.read_text(encoding='utf-8')
        ir=parse(text)
        candidate=propose_branch(ir, 8)
        cir=parse(candidate)
        self.assertFalse(ir['brain'].get('graph', False))
        self.assertTrue(cir['brain']['graph'])
        self.assertEqual(cir['brain']['weights'],30)
        self.assertEqual([x['sources'] for x in cir['shape_contract']['layers']],
                         [['input'],['input'],['assist_direct','assist_relu']])
        self.assertEqual(SOURCE.read_text(encoding='utf-8'),text)
        def strip_locations(ops):
            return [{key: (strip_locations(value) if key == 'body' else value)
                     for key, value in op.items() if key != 'line'} for op in ops]
        self.assertEqual(strip_locations(ir['operations']), strip_locations(cir['operations']))

    def test_learning_requires_observe_and_is_finite(self):
        header = 'brain Test {\n senses 2\n neurons out 1 linear\n}\n'
        # The compiler now rejects missing input before assistant validation.
        with self.assertRaisesRegex(CompileError, 'learn requires observe'):
            parse(header + 'learn [1] rate 0.1')
        # A second learn is legal in the language, but this bounded experiment
        # requires every target to have its own immediately preceding observe.
        ir = parse(header + 'observe [0, 0]\nlearn [1] rate 0.1\nlearn [1] rate 0.1')
        with self.assertRaisesRegex(ExperimentError, 'immediately follow'):
            collect_training(ir['operations'])

    def test_leakage_does_not_pass(self):
        train, _=collect_training(parse(SOURCE.read_text())['operations'])
        with tempfile.TemporaryDirectory(prefix='nl05_leak_') as td:
            file=Path(td)/'evaluation.json'
            data=json.loads(DATA.read_text())
            data['validation'][0]['input']=[0, 0]  # training point
            file.write_text(json.dumps(data))
            with self.assertRaisesRegex(ExperimentError, 'leaks'):
                load_evaluation(file, train, 2, 1)

    def test_validation_test_overlap_does_not_pass(self):
        train, _=collect_training(parse(SOURCE.read_text())['operations'])
        with tempfile.TemporaryDirectory(prefix='nl05_overlap_') as td:
            file=Path(td)/'evaluation.json'
            data=json.loads(DATA.read_text())
            data['test'][0]['input']=data['validation'][0]['input']
            file.write_text(json.dumps(data))
            with self.assertRaisesRegex(ExperimentError, 'leaks'):
                load_evaluation(file, train, 2, 1)

    def test_bad_json_and_nonfinite_rejected(self):
        train, _=collect_training(parse(SOURCE.read_text())['operations'])
        with tempfile.TemporaryDirectory(prefix='nl05_bad_') as td:
            file=Path(td)/'evaluation.json'
            data=json.loads(DATA.read_text())
            data['validation'][0]['target']=[float('nan')]
            file.write_text(json.dumps(data))
            with self.assertRaisesRegex(ExperimentError, 'finite'):
                load_evaluation(file, train, 2, 1)

    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory(prefix='nl05_keep_') as td:
            output=Path(td)/'out';output.mkdir();sentinel=output/'precious';sentinel.write_text('unchanged')
            with self.assertRaisesRegex(ExperimentError, 'already exists'):
                run_experiment(SOURCE,DATA,output)
            self.assertEqual(sentinel.read_text(),'unchanged')

    def test_restricted_source_rejects_hidden_side_effects(self):
        with tempfile.TemporaryDirectory(prefix='nl05_effect_') as td:
            p=Path(td)/'src.nir';p.write_text(SOURCE.read_text()+'report "report.json"\n')
            with self.assertRaisesRegex(ExperimentError,'may contain only'):
                run_experiment(p,DATA,Path(td)/'out')

    def test_unexpected_model_format_rejected(self):
        with tempfile.TemporaryDirectory(prefix='nl05_ref_') as td:
            p=Path(td)/'m.nlw';p.write_text('NLMODEL9\n2 1\n0\nEND\n')
            with self.assertRaises(ReferenceError):read_model(p)

    def test_reference_linear_formula(self):
        with tempfile.TemporaryDirectory(prefix='nl05_refmath_') as td:
            p=Path(td)/'test.nlw'
            p.write_text('NLMODEL2\n2 1\n0\noutput 2 1 0 0\n0.1 \n0.4 0.3 \n1 1 \nEND\n')
            m=read_model(p)
            self.assertAlmostEqual(predict(m,[0.5,-0.5])[0],0.15)


@unittest.skipUnless(shutil.which('g++'), 'g++ not available')
class AssistantNativeTests(unittest.TestCase):
    def test_accept_nonlinear_candidate_and_reference(self):
        with tempfile.TemporaryDirectory(prefix='nl05_accept_') as td:
            output=Path(td)/'experiment'
            source_before=SOURCE.read_bytes()
            record=run_experiment(SOURCE,DATA,output)
            self.assertEqual(record['decision'],'ACCEPT')
            self.assertLess(record['candidate']['validation_rmse'],record['baseline']['validation_rmse']*0.8)
            self.assertEqual(record['candidate']['reference_check'],'PASS')
            self.assertEqual(record['baseline']['reference_check'],'PASS')
            self.assertTrue((output/'recommended.nir').is_file())
            self.assertEqual(SOURCE.read_bytes(),source_before)
            self.assertEqual(record['candidate']['static_weights'],30)
            self.assertTrue((output/'baseline.nlw').is_file())
            self.assertTrue((output/'candidate.nlw').is_file())
            self.assertEqual(json.loads((output/'evidence.json').read_text())['decision'],'ACCEPT')

    def test_unnecessary_graph_is_rejected_on_genuinely_linear_task(self):
        with tempfile.TemporaryDirectory(prefix='nl05_linear_') as td:
            out=Path(td)/'experiment'
            result=run_experiment(LINEAR_SOURCE,LINEAR_DATA,out)
            self.assertEqual(result['decision'],'REJECT_NO_VALIDATION_GAIN')
            self.assertLess(result['baseline']['validation_rmse'],result['candidate']['validation_rmse'])
            self.assertFalse((out/'recommended.nir').exists())

    def test_resource_gate_rejects_and_does_not_publish(self):
        with tempfile.TemporaryDirectory(prefix='nl05_budget_') as td:
            out=Path(td)/'experiment'
            result=run_experiment(SOURCE,DATA,out,max_weights=10)
            self.assertEqual(result['decision'],'REJECT_RESOURCE_LIMIT')
            self.assertFalse((out/'recommended.nir').exists())
            self.assertFalse((out/'candidate').exists())
            self.assertEqual(result['baseline']['reference_check'],'PASS')

    def test_no_gain_rejects_and_keeps_original(self):
        with tempfile.TemporaryDirectory(prefix='nl05_nogain_') as td:
            out=Path(td)/'experiment'
            before=SOURCE.read_bytes()
            result=run_experiment(SOURCE,DATA,out,min_gain=0.999999)
            self.assertEqual(result['decision'],'REJECT_NO_VALIDATION_GAIN')
            self.assertFalse((out/'recommended.nir').exists())
            self.assertEqual(SOURCE.read_bytes(),before)

    def test_test_targets_are_not_used_for_decision(self):
        with tempfile.TemporaryDirectory(prefix='nl05_blind_') as td:
            d=Path(td)
            modified=json.loads(DATA.read_text())
            for row in modified['test']:
                row['target']=[row['target'][0]+2.0]
            different=d/'different.json';different.write_text(json.dumps(modified))
            first=run_experiment(SOURCE,DATA,d/'regular')
            second=run_experiment(SOURCE,different,d/'altered')
            self.assertEqual(first['decision'],second['decision'])
            self.assertEqual(first['baseline']['validation_rmse'],second['baseline']['validation_rmse'])
            self.assertEqual(first['candidate']['validation_rmse'],second['candidate']['validation_rmse'])
            self.assertNotEqual(first['candidate']['test_rmse'],second['candidate']['test_rmse'])

    @unittest.skipUnless(shutil.which('clang++'),'clang++ not available')
    def test_clang_compilation_on_both_variants(self):
        with tempfile.TemporaryDirectory(prefix='nl05_clang_') as td:
            out=Path(td)/'experiment'
            result=run_experiment(SOURCE,DATA,out,compiler='clang++')
            self.assertEqual(result['decision'],'ACCEPT')
            self.assertEqual(result['candidate']['reference_check'],'PASS')

    def test_reference_disagreement_blocks_acceptance(self):
        with tempfile.TemporaryDirectory(prefix='nl05_inject_') as td:
            out=Path(td)/'experiment'
            with patch('assistant.reference_predict',return_value=[9876543.0]):
                with self.assertRaisesRegex(ExperimentError,'independent FP64 reference mismatch'):
                    run_experiment(SOURCE,DATA,out)
            self.assertFalse((out/'recommended.nir').exists())
            self.assertFalse((out/'evidence.json').exists())

if __name__=='__main__':unittest.main(verbosity=2)
