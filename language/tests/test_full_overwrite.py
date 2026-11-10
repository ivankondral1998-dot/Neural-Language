"""Full computation must overwrite poisoned buffers and recover after errors."""
from pathlib import Path
import shutil,subprocess,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]
class FullOverwrite(unittest.TestCase):
    def test_all_storage_masks_graphs_activations_and_failure_recovery(self):
        for compiler in ('g++','clang++'):
            if not shutil.which(compiler):continue
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory() as tmp:
                exe=Path(tmp)/'overwrite'
                subprocess.run([compiler,'-std=c++17','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-I',str(ROOT/'runtime'),str(ROOT/'tests/runtime_full_overwrite.cpp'),'-o',str(exe)],check=True,capture_output=True,text=True)
                run=subprocess.run([str(exe)],text=True,capture_output=True)
                self.assertEqual(run.returncode,0,run.stderr)
                self.assertIn('FULL_OVERWRITE_PASS cases=128 recovery=1',run.stdout)
