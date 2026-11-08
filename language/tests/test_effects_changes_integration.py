"""NeuralEffects/1 + NeuralChanges/1 integration against actual native programs."""
import copy
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
from compiler import CompileError, parse, emit, analyze_ir, verify_neural_changes, verify_semantic_contract

HEAD='brain CheckpointFlow {\n senses 2\n neurons h 4 relu\n neurons out 1 linear\n init 31\n}\n'

class EffectContractTests(unittest.TestCase):
    def test_restoring_input_snapshot_enables_delta(self):
        text=(HEAD+'observe [0.2,0.3]\nremember yes\nobserve [0.5,0.6]\n'
              'rewind yes\nobserve_delta 0 add 0.1\npredict\n')
        ir=parse(text)
        meta=ir['semantic_contract']
        self.assertEqual(meta['version'],'NeuralEffects/1')
        self.assertTrue(meta['checkpoints']['yes']['input_bound'])
        self.assertEqual(meta['operation_counts']['observe_delta'],1)
        self.assertEqual(ir['neural_changes']['occurrences'][0]['index'],0)
        self.assertIn('brain.change_input(0,0.10000000000000001)',emit(ir))

    def test_restoring_unbound_snapshot_rejects_delta(self):
        text=HEAD+'remember empty\nobserve [0.2,0.3]\nrewind empty\nobserve_delta 0 add 0.1\n'
        with self.assertRaisesRegex(CompileError,'preceding observe'):parse(text)

    def test_rewinding_missing_checkpoint_rejected(self):
        with self.assertRaisesRegex(CompileError,'unknown checkpoint'):
            parse(HEAD+'observe [0.2,0.3]\nrewind missing\npredict\n')

    def test_duplicate_and_loop_checkpoints_rejected(self):
        with self.assertRaisesRegex(CompileError,'already exists'):
            parse(HEAD+'observe [0.2,0.3]\nremember dup\nremember dup\npredict\n')
        with self.assertRaisesRegex(CompileError,'repeating cycle'):
            parse(HEAD+'cycle 2 {\nremember bad\n}\nobserve [0.2,0.3]\n')

    def test_second_iteration_after_restore_unbound_snapshot_rejected(self):
        # First iteration starts observed; it ends unobserved. Second iteration
        # must reject observe_delta because the snapshot was created before observe.
        text=(HEAD+'remember empty\nobserve [0.2,0.3]\n'
              'cycle 2 {\nobserve_delta 0 add 0.1\nrewind empty\n}\n')
        with self.assertRaisesRegex(CompileError,'preceding observe'):parse(text)

    def test_second_iteration_with_bound_snapshot_accepted(self):
        text=(HEAD+'observe [0.2,0.3]\nremember yes\n'
              'cycle 3 {\nrewind yes\nobserve_delta 0 add 0.1\npredict\n}\n')
        ir=parse(text)
        self.assertEqual(ir['semantic_contract']['operation_counts']['observe_delta'],3)
        self.assertEqual(ir['estimated_max_operations'],11)

    def test_effect_metadata_checked_prior_to_native_emit(self):
        ir=parse(HEAD+'observe [0.2,0.3]\nremember yes\nrewind yes\nobserve_delta 0 add 0.1\npredict\n')
        variants=[]
        bad=copy.deepcopy(ir);bad['semantic_contract']['checkpoints']['yes']['input_bound']=False;variants.append(bad)
        bad=copy.deepcopy(ir);bad['semantic_contract']['effects']['observe_delta']['requires']=[];variants.append(bad)
        bad=copy.deepcopy(ir);bad['semantic_contract']['operation_counts']['observe_delta']=0;variants.append(bad)
        bad=copy.deepcopy(ir);bad['operations'][3]['op']='rewind';bad['operations'][3]['name']='missing';variants.append(bad)
        bad=copy.deepcopy(ir);bad['operations'][2]['name']='missing';variants.append(bad)
        for bad in variants:
            with self.subTest(kind=bad['operations'][2]['op']),self.assertRaises(CompileError):
                emit(bad)
        legacy=copy.deepcopy(ir);legacy.pop('semantic_contract')
        self.assertEqual(verify_semantic_contract(legacy),ir['semantic_contract'])
        self.assertIsInstance(analyze_ir(legacy),dict)

    def test_no_cpp_codegen_change_for_previous_valid_program(self):
        text=HEAD+'observe [0.2,0.3]\npredict\n'
        self.assertNotIn('semantic_contract',emit(parse(text)))
        self.assertEqual(emit(parse(text)),emit(parse(text)))

    @unittest.skipUnless(shutil.which('g++') and shutil.which('clang++'),'requires GCC and Clang')
    def test_restored_snapshot_native_matches_full_oracle(self):
        inputs=[(0.2,0.3),(0.45,0.6),(0.2,0.3),(0.3,0.3)]
        source=(HEAD+'tune incremental\nobserve [0.2,0.3]\nremember checkpoint\npredict\n'
                'observe [0.45,0.6]\npredict\nrewind checkpoint\n'
                'observe_delta 0 add 0.1\npredict\n')
        oracle=(HEAD+'tune full\nobserve [0.2,0.3]\npredict\n'
                'observe [0.45,0.6]\npredict\nobserve [0.3,0.3]\npredict\n')
        with tempfile.TemporaryDirectory(prefix='nl_effects_') as root:
            root=Path(root)
            for compiler in ('g++','clang++'):
                trajectories=[]
                for label,src in (('delta',source),('oracle',oracle)):
                    sp=root/(f'{label}_{compiler}.nir');sp.write_text(src,encoding='utf-8')
                    target=root/(f'{label}_{compiler}')
                    compiled=subprocess.run([sys.executable,str(ROOT/'compiler.py'),str(sp),
                            '-o',str(target),'--compiler',compiler],capture_output=True,text=True,timeout=55)
                    self.assertEqual(compiled.returncode,0,compiled.stderr)
                    run=subprocess.run([str(target)],capture_output=True,text=True,timeout=10)
                    self.assertEqual(run.returncode,0,run.stderr)
                    values=[float(x.split(',')[0]) for x in re.findall(r'PREDICT version=\d+ values=([^\n]+)',run.stdout)]
                    trajectories.append(values)
                self.assertEqual(len(trajectories[0]),3)
                for actual,expected in zip(*trajectories):
                    self.assertAlmostEqual(actual,expected,delta=1e-9)

if __name__=='__main__':unittest.main(verbosity=2)
