"""Synthetic inputs only; no persona data or existing checkpoints."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from dialogue_tokenizer import ByteTokenizer, load_tokenizer, train_tokenizer


class TokenizerTests(unittest.TestCase):
    def test_legacy_ids(self):
        tok = load_tokenizer('byte260')
        self.assertEqual(tok.encode('Ё'), list('Ё'.encode()))
        self.assertEqual((tok.bos, tok.user, tok.assistant, tok.eos), (256,257,258,259))
        self.assertEqual(tok.metadata(), 'byte260')

    def test_bpe_roundtrip_digest_and_special_literals(self):
        with tempfile.TemporaryDirectory() as td:
            corpus = Path(td)/'text.txt'
            corpus.write_text(('Привет, мир! A byte tokenizer.\n' * 20), encoding='utf-8')
            output = Path(td)/'tokenizer.json'
            tok = train_tokenizer([corpus], output, 4096)
            restored = load_tokenizer(tok.metadata())
            for text in ['', 'Ёжик\n café 🙂 中文\x00', '<|bos|> <|user|> <|assistant|> <|eos|>']:
                ids = restored.encode(text)
                self.assertFalse(set(ids) & restored.special_ids)
                self.assertEqual(restored.decode(ids), text)
            self.assertLessEqual(tok.vocab_size, 4096)
            meta = tok.metadata(); meta['sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'digest'):
                load_tokenizer(meta)
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                train_tokenizer([corpus], output, 4096)

    def test_unknown_metadata_rejected(self):
        for bad in ['bpe4096', {}, {'kind':'other'}]:
            with self.assertRaises(ValueError):
                load_tokenizer(bad)

    def test_lossy_pipeline_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            corpus = Path(td)/'text.txt'; corpus.write_text('Привет world\n',encoding='utf-8')
            tok = train_tokenizer([corpus],Path(td)/'tokenizer.json')
            meta = tok.metadata()
            spec = json.loads(meta['json'])
            spec['normalizer'] = {'type':'Lowercase'}
            meta['json'] = json.dumps(spec)
            meta['sha256'] = hashlib.sha256(meta['json'].encode()).hexdigest()
            with self.assertRaisesRegex(ValueError,'unsupported'):
                load_tokenizer(meta)


if __name__ == '__main__':
    unittest.main()
