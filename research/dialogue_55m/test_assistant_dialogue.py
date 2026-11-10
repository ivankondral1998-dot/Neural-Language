import tempfile
import unittest
from pathlib import Path
import torch
from assistant_dialogue import check_splits, pairs_from_jsonl, measure
from train import examples

class PolicyTests(unittest.TestCase):
    def test_validation_leakage(self):
        with self.assertRaisesRegex(ValueError, 'leakage'):
            check_splits([{'user':'Привет','assistant':'Да'}],
                         [{'user':'привет','assistant':'Нет'}],
                         [{'user':'Спасибо','assistant':'Пожалуйста'}])

    def test_duplicate_prompt(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            check_splits([{'user':'X','assistant':'Y'}, {'user':'x','assistant':'Z'}], [], [])

    def test_bad_rows(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bad.jsonl';p.write_text('{"user":"a"}\n')
            with self.assertRaises(ValueError):pairs_from_jsonl(p)

    def test_masked_answer_metric(self):
        class Dummy(torch.nn.Module):
            def forward(self, x):
                return torch.zeros((1,x.shape[1],260))
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'data.jsonl';p.write_text('{"user":"a","assistant":"b"}\n')
            m=measure(Dummy(),examples(p,128))
            self.assertEqual(m['answer_bytes_scored'], 2)
            self.assertAlmostEqual(m['mean_answer_nll'],__import__('math').log(260),places=4)

if __name__=='__main__': unittest.main()
