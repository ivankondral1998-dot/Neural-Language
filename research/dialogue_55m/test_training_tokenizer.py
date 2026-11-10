"""Fast CPU integration using fresh tiny models and synthetic Unicode only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dialogue_tokenizer import ByteTokenizer, train_tokenizer
from train import (ByteDialogue, checkpoint_tokenizer, examples, iter_examples,
                   parameter_count, read_nir, train, MAX_STREAM_BYTES)
from chat import reply
from assistant_dialogue import measure, run

torch.set_num_threads(4)


class TrainingTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root/'synthetic.jsonl'
        self.data.write_text('\n'.join(json.dumps({'user': u, 'assistant': a}, ensure_ascii=False)
                                      for u,a in [('Привет','Мир'), ('Hello','Hi')])+'\n', encoding='utf-8')
        corpus = self.root/'text.txt'
        corpus.write_text(('Привет мир. Hello world!\n'*10), encoding='utf-8')
        self.tok_path = self.root/'tokenizer.json'
        self.tok = train_tokenizer([corpus], self.tok_path, 4096)
        self.cfg = {'context':64, 'width':12, 'heads':3, 'blocks':1,
                    'feedforward':24, 'seed':42}
        self.source = self.root/'tiny.nir'
        self.source.write_text('synthetic source for mock parser\n', encoding='utf-8')

    def tearDown(self):
        self.tmp.cleanup()

    def test_locked_profiles_and_parameter_counts(self):
        here = Path(__file__).parent
        old, *_ = read_nir(here/'experiment.nir')
        self.assertEqual(parameter_count(old), 55439360)
        new, *_ = read_nir(here/'experiment_bpe.nir')
        self.assertEqual((new['context'],new['feedforward']), (1024,1824))
        self.assertEqual(parameter_count(dict(new,vocab_size=4096)),55630656)
        tiny = ByteDialogue(self.cfg)
        self.assertEqual(parameter_count(self.cfg),sum(p.numel() for p in tiny.parameters()))

    def test_streaming_mask_and_overlong_rejection(self):
        with patch.object(Path, 'read_text', side_effect=AssertionError('must stream')):
            pairs = list(iter_examples(self.data,64,self.tok))
        self.assertEqual(len(pairs),2)
        x,y = pairs[0]
        prompt = [self.tok.bos,self.tok.user]+self.tok.encode('Привет')+[self.tok.assistant]
        self.assertTrue(torch.all(y[0,:len(prompt)-1] == -100))
        self.assertEqual(int(y[0,-1]),self.tok.eos)
        with self.assertRaisesRegex(ValueError,'exceeds'):
            examples(self.data,2,self.tok)

    def test_streaming_and_materialization_have_separate_bounds(self):
        sparse = self.root/'sparse.jsonl'
        with sparse.open('wb') as stream:
            stream.write(b'{"user":"a","assistant":"b"}\n')
            stream.truncate(16*1024*1024+1)
        with self.assertRaisesRegex(ValueError,'16 MiB'):
            examples(sparse,64)
        iterator = iter_examples(sparse,64)
        self.assertEqual(int(next(iterator)[0][0,0]),256)
        iterator.close()
        with sparse.open('ab') as stream:
            stream.truncate(MAX_STREAM_BYTES+1)
        with self.assertRaisesRegex(ValueError,'too big'):
            next(iter_examples(sparse,64))

    def test_legacy_checkpoint_and_mismatched_tokenizer(self):
        model = ByteDialogue(self.cfg)
        package = {'architecture':self.cfg,'state_dict':model.state_dict(),'tokenizer':'byte260'}
        self.assertEqual(checkpoint_tokenizer(package).metadata(),'byte260')
        saved = self.root/'legacy.pt'; torch.save(package,saved)
        self.assertIsInstance(reply(saved,'hi',1),str)
        package['tokenizer'] = self.tok.metadata()
        with self.assertRaisesRegex(ValueError,'mismatch'):
            checkpoint_tokenizer(package)

    def _train(self, folder, steps, cfg=None, resume=None, tokenizer=None):
        args = (cfg or self.cfg, self.data.name, steps, .0001, 'fresh.pt')
        with patch('train.read_nir',return_value=args):
            return train(self.source,self.root/folder,tokenizer_path=tokenizer,resume=resume)

    def test_exact_resume_and_no_overwrite(self):
        full = self._train('four',4)
        first = self._train('two',3)
        continued = self._train('continued',1,resume=Path(first['checkpoint']))
        a = torch.load(full['checkpoint'],weights_only=True)
        b = torch.load(continued['checkpoint'],weights_only=True)
        self.assertEqual(b['trained_steps'],4)
        self.assertTrue(continued['optimizer_state_resumed'])
        for key,value in a['state_dict'].items():
            self.assertTrue(torch.equal(value,b['state_dict'][key]),key)
        with self.assertRaisesRegex(ValueError,'overwrite'):
            self._train('two',1)
        saved = Path(first['checkpoint']); baseline = saved.read_bytes()
        a.pop('optimizer_state_dict'); torch.save(a,self.root/'no_optimizer.pt')
        with self.assertRaisesRegex(ValueError,'optimizer'):
            self._train('invalid',1,resume=self.root/'no_optimizer.pt')
        self.assertEqual(saved.read_bytes(),baseline)
        with self.assertRaisesRegex(ValueError,'dry-run'):
            train(self.source,self.root/'dry',dry_run=True,resume=saved)

    def test_bpe_training_chat_and_continuation_preserve_metadata(self):
        cfg = dict(self.cfg, tokenizer_kind='bpe4096', vocab_size=self.tok.vocab_size)
        result = self._train('bpe',1,cfg=cfg,tokenizer=self.tok_path)
        saved = Path(result['checkpoint'])
        package = torch.load(saved,weights_only=True)
        self.assertEqual(checkpoint_tokenizer(package).metadata(),self.tok.metadata())
        self.assertIsInstance(reply(saved,'🙂 Привет',2),str)
        val = self.root/'val.jsonl'; test = self.root/'test.jsonl'
        for path,prefix in [(val,'Val'),(test,'Test')]:
            path.write_text('\n'.join(json.dumps({'user':prefix+str(i),'assistant':'Мир'}) for i in range(2))+'\n')
        evidence = run(saved,self.data,val,test,self.root/'gate',steps=1)
        self.assertEqual(evidence['tokenizer_kind'],'bpe4096')
        self.assertIn('answer_tokens_scored',evidence['baseline_validation'])
        self.assertNotIn('answer_bytes_scored',evidence['baseline_validation'])
        if evidence['decision']=='ACCEPT':
            candidate = torch.load(self.root/'gate'/'candidate.pt',weights_only=True)
            self.assertEqual(candidate['tokenizer'],self.tok.metadata())

    def test_resume_rejects_changed_tokenizer_identity(self):
        cfg = dict(self.cfg,tokenizer_kind='bpe4096',vocab_size=self.tok.vocab_size)
        result = self._train('bpe',1,cfg=cfg,tokenizer=self.tok_path)
        alternate = self.root/'alternate.json'
        alternate.write_text(self.tok.raw+'\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'tokenizer mismatch'):
            self._train('bad_resume',1,cfg=cfg,resume=Path(result['checkpoint']),tokenizer=alternate)
        self.assertFalse((self.root/'bad_resume'/'fresh.pt').exists())


if __name__ == '__main__':
    unittest.main()
