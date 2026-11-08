"""Neural Language compiler CLI source-preservation regression tests."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPILER = ROOT / 'compiler.py'


class CompilerSourceSafetyTests(unittest.TestCase):
    SOURCE = 'brain A {\n senses 1\n neurons out 1 linear\n}\nobserve [0]\npredict\n'

    def invoke(self, source, target):
        return subprocess.run([sys.executable, str(COMPILER), str(source),
                               '-o', str(target), '--emit-only'],
                              capture_output=True, text=True, timeout=20)

    def test_binary_target_cannot_overwrite_source(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / 'original.nir'
            source.write_text(self.SOURCE)
            outcome = self.invoke(source, source)
            self.assertNotEqual(outcome.returncode, 0)
            self.assertEqual(source.read_text(), self.SOURCE)

    def test_generated_sidecar_cannot_overwrite_source(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / 'output.nir.json'
            source.write_text(self.SOURCE)
            outcome = self.invoke(source, Path(td) / 'output.exe')
            self.assertNotEqual(outcome.returncode, 0)
            self.assertEqual(source.read_text(), self.SOURCE)

    def test_real_line_numbers_can_exceed_logical_line_limit(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / 'comments.nir'
            source.write_text('# just a comment\n' * 5200 + self.SOURCE)
            output = Path(td) / 'compiled'
            outcome = self.invoke(source, output)
            self.assertEqual(outcome.returncode, 0, outcome.stderr)
            self.assertTrue(output.with_suffix('.cpp').exists())

    def test_normal_emit_preserves_input(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / 'source.nir'
            source.write_text(self.SOURCE)
            result = self.invoke(source, Path(td) / 'target')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(source.read_text(), self.SOURCE)
            self.assertTrue((Path(td) / 'target.cpp').exists())
            self.assertTrue((Path(td) / 'target.nir.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
