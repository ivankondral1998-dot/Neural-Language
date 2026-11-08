"""NeuralEffects/1 + NeuralChanges/1 integration against actual native programs."""
import copy
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
        # Frozen baseline hashes from the NL-COMP-003 compiler, independently
        # generated before this integration. This protects the native output
        # for all nine shipped language samples, not just self-consistency.
        golden={
            'implicit_observe.nir':'13378142548e21a7a07f837cf69352e0e715e86f0f79b7a9e9421c4e08b4f46e',
            'inference.nir':'daca90366ddd4930a91d098ab6dc4d0fe3333c241f1169db14d2b41c3cd8d781',
            'integrated.nir':'fc3e3f055f33124e8ae2592c5ce7dd8a9508e91450bd1b5d18e60f9a2509275b',
            'learning.nir':'13fddbdd25a375862436ea784a39b26d280c896a22838877eca730937c73f76b',
            'minimal.nir':'b06ae3bd2706f797bf8f760574c4f99f4b967277beeafb132503eb917da01da4',
            'observe_delta.nir':'56ef8ab7c6b2fb730b76af46603aec03dd19424856971f5197a336b833d4a369',
            'optimization_profile.nir':'5dfcf57100720cbe579031f7e39f1cdb4cfeee5e10aefa48808107afa2486071',
            'performance.nir':'05b55babb03a17b429931abe91e0cc7cc17be6bc39e4afa301b395ac3d87d96c',
            'self_check.nir':'c647a62891c5d1da7f775df0c189db5526d96544b98d0c4dcca11d12e5520897',
        }
        for filename,expected in golden.items():
            with self.subTest(example=filename):
                source=(ROOT/'examples'/filename).read_text(encoding='utf-8')
                self.assertEqual(hashlib.sha256(emit(parse(source)).encode()).hexdigest(),expected)

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
