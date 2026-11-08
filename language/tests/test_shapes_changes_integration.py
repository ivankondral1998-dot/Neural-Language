"""NL-COMP-005: canonical NeuralShapes/1 with NeuralEffects/1 + NeuralChanges/1.

These tests deliberately mutate already-parsed IR to ensure the code generator
is not merely trusting the source parser. Full model shape and operation
signatures are checked before generated native source can be emitted.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from compiler import CompileError,parse,emit,analyze_ir,verify_typed_ir

SAMPLE=('brain TypedLive {\n senses 3\n neurons a 4 relu float\n'
        ' neurons b 2 sigmoid int8\n init 31\n budget 500000 500000\n}\n'
        'observe [0.1,0.2,0.3]\nremember saved\n'
        'observe_delta 1 add 0.02\npredict\n'
        'weight a 1 2 add -0.05\nedge b 0 1 off\n'
        'rewind saved\nobserve_delta 2 add 0.015\n'
        'validate [0.1,0.2,0.3] target [0.3,0.4] tolerance 1\n'
        'expect [0.3,0.4] tolerance 1\npredict\n')

GOLDEN={
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

class ShapesEffectsChanges(unittest.TestCase):
    def setUp(self):self.ir=parse(SAMPLE)
    def invalid(self, mutator, message=None):
        other=deepcopy(self.ir)
        mutator(other)
        with self.assertRaises(CompileError) as raised:emit(other)
        with self.assertRaises(CompileError):analyze_ir(other)
        if message is not None:self.assertIn(message,str(raised.exception))

    def test_combined_metadata_is_consistent(self):
        ir=self.ir
        self.assertEqual(ir['shape_contract']['version'],'NeuralShapes/1')
        self.assertEqual(ir['semantic_contract']['version'],'NeuralEffects/1')
        self.assertEqual(ir['neural_changes']['contract'],'NeuralChanges/1')
        self.assertEqual(ir['shape_contract']['input']['shape'],[3])
        self.assertEqual(ir['shape_contract']['layers'][0]['weight_shape'],[4,3])
        self.assertEqual(ir['shape_contract']['layers'][1]['weight_shape'],[2,4])
        self.assertEqual(ir['shape_contract']['operation_signatures']['observe_delta'],
                         {'index':'input_index','delta':'f64'})
        self.assertEqual(ir['shape_contract']['operation_signatures']['validate']['target']['shape'],[2])
        self.assertTrue(ir['semantic_contract']['checkpoints']['saved']['input_bound'])
        self.assertEqual(verify_typed_ir(ir),ir['shape_contract'])

    def test_json_roundtrip_and_legacy_ir_without_shapes(self):
        ir=json.loads(json.dumps(self.ir))
        self.assertEqual(emit(ir),emit(self.ir))
        del ir['shape_contract']
        self.assertEqual(verify_typed_ir(ir),self.ir['shape_contract'])
        self.assertEqual(emit(ir),emit(self.ir))
        ir.pop('semantic_contract')
        self.assertEqual(emit(ir),emit(self.ir))

    def test_mutated_brain_fields(self):
        for name,value in (('name','9bad'),('senses',2),('senses',False),('senses',2049),
                           ('init',4294967296),('init','12'),('weights',999),('macs',-1),
                           ('weight_bytes',1),('storage_bytes',1),('budget_scope','fake'),
                           ('budget',(1,1)),('budget',(True,1))):
            with self.subTest(field=name,value=value):
                self.invalid(lambda ir:ir['brain'].__setitem__(name,value))

    def test_mutated_layers(self):
        for attr,value in (('size',1),('size',0),('size',False),('activation','bad'),
                           ('storage','mystery'),('name','9invalid'),('name','b')):
            with self.subTest(attr=attr,value=value):
                self.invalid(lambda ir:ir['brain']['neurons'][0].__setitem__(attr,value))
        self.invalid(lambda ir:ir['brain']['neurons'].append(deepcopy(ir['brain']['neurons'][0])))

    def test_input_shapes_and_numeric_value_types(self):
        cases=(lambda ir:ir['operations'][0].__setitem__('values',[0.1]),
               lambda ir:ir['operations'][0]['values'].__setitem__(0,float('nan')),
               lambda ir:ir['operations'][0]['values'].__setitem__(0,False),
               lambda ir:ir['operations'][0]['values'].__setitem__(0,float('inf')),
               lambda ir:ir['operations'][0]['values'].__setitem__(0,1e9),
               lambda ir:ir['operations'][8].__setitem__('input',[0.1]),
               lambda ir:ir['operations'][8].__setitem__('target',[0.3]),
               lambda ir:ir['operations'][9].__setitem__('values',[0.3]),
               lambda ir:ir['operations'][8].__setitem__('tolerance',True))
        for f in cases:
            with self.subTest(mutation=str(f)):self.invalid(f)

    def test_delta_and_weight_operand_mutations(self):
        for idx,val in (('index',3),('index',-1),('index',False),('delta',float('nan')),
                        ('delta',True),('delta',1e12)):
            with self.subTest(field=idx,value=val):
                self.invalid(lambda ir:ir['operations'][2].__setitem__(idx,val))
        self.invalid(lambda ir:ir['operations'][4].__setitem__('col',3))
        self.invalid(lambda ir:ir['operations'][4].__setitem__('row',False))
        self.invalid(lambda ir:ir['operations'][4].__setitem__('delta',float('inf')))
        self.invalid(lambda ir:ir['operations'][5].__setitem__('enabled',1))

    def test_metadata_tamper(self):
        self.invalid(lambda ir:ir['shape_contract']['input']['shape'].__setitem__(0,7),'shape contract mismatch')
        self.invalid(lambda ir:ir['shape_contract']['layers'][1].__setitem__('weight_shape',[1,8]),'shape contract mismatch')
        self.invalid(lambda ir:ir['shape_contract']['operation_signatures']['observe_delta'].__setitem__('index','float'),'shape contract mismatch')
        self.invalid(lambda ir:ir['semantic_contract']['operation_counts'].__setitem__('observe_delta',777),'semantic contract mismatch')
        self.invalid(lambda ir:ir['neural_changes']['dependencies'][0].__setitem__('target','wrong'),'dependencies')

    def test_mutated_operation_lifetime_and_count(self):
        self.invalid(lambda ir:ir.__setitem__('estimated_max_operations',999))
        self.invalid(lambda ir:ir['operations'][2].__setitem__('line',0))
        self.invalid(lambda ir:ir['operations'][2].__setitem__('line',False))
        self.invalid(lambda ir:ir['operations'][2].__setitem__('op','unexpected'))
        self.invalid(lambda ir:ir['operations'][6].__setitem__('name','missing'))

    def test_malformed_checkpoint_label_is_compile_error_not_python_error(self):
        for bad_name in ([], {}, 4, False, None):
            with self.subTest(value=bad_name):
                self.invalid(lambda ir:ir['operations'][6].__setitem__('name',bad_name),
                             'checkpoint name invalid')

    def test_filename_guards(self):
        ir=parse('brain FileCheck {\n senses 1\n neurons out 1 linear\n}\n'
                 'observe [0]\nsave_model "safe.nlw"\n')
        for fn in ('../pwn','a/b','\u202etrick','a\nfilename', 'z'*81):
            bad=deepcopy(ir);bad['operations'][1]['filename']=fn
            with self.assertRaises(CompileError):emit(bad)

    def test_randomized_dense_shape_oracle(self):
        rng=random.Random(20261108)
        for n in range(120):
            inp=rng.randrange(1,45)
            dims=[rng.randrange(1,56) for _ in range(rng.randrange(1,6))]
            names=['n'+str(i) for i in range(len(dims))]
            storage=[rng.choice(['float','int8']) for _ in dims]
            src='brain Random {\n senses '+str(inp)+'\n'
            for name,size,st in zip(names,dims,storage):
                src+=f' neurons {name} {size} relu {st}\n'
            src+='}\nobserve ['+','.join(['0.2']*inp)+']\n'
            for i in range(6):src+=f'observe_delta {rng.randrange(inp)} add 0.0001\n'
            src+='predict\n'
            ir=parse(src)
            self.assertEqual(ir['shape_contract']['layers'][-1]['output']['shape'],[dims[-1]])
            calculated=sum(a*b for a,b in zip([inp]+dims[:-1],dims))
            self.assertEqual(ir['brain']['weights'],calculated)
            self.assertEqual(len(emit(ir))>100,True)

    def test_frozen_cpp_golden(self):
        for filename,digest in GOLDEN.items():
            with self.subTest(program=filename):
                ir=parse((ROOT/'examples'/filename).read_text(encoding='utf-8'))
                self.assertEqual(hashlib.sha256(emit(ir).encode()).hexdigest(),digest)

if __name__=='__main__':unittest.main(verbosity=2)
