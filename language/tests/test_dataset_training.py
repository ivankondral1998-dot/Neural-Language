"""NL-TRAIN-001 tests through typed IR and independently compiled programs."""
import copy
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compiler

LANG = Path(compiler.__file__).parent
BRAIN = "brain Data {\n senses 2\n neurons out 2 linear\n init 42\n}\n"
TRAIN = 'train_data "train.txt" samples 2 epochs 3 rate 0.1 seed 17\n'
PREDICT = 'predict_data "test.txt" samples 2\n'


class DatasetTrainingTests(unittest.TestCase):
    def test_ir_typed_effects_and_budget(self):
        ir = compiler.parse(BRAIN + TRAIN + PREDICT + "predict\n")
        self.assertEqual(ir["estimated_max_operations"], 9)
        contract = ir["shape_contract"]["operation_signatures"]
        self.assertEqual(contract["train_data"]["features"], {"dtype": "f64", "rank": 2, "columns": 2})
        self.assertEqual(contract["train_data"]["label"], {"dtype": "i32", "classes": 2})
        self.assertEqual(ir["semantic_contract"]["effects"]["train_data"]["filesystem"], "read")
        self.assertTrue(ir["semantic_contract"]["final_state"]["input_bound"])
        self.assertIn('neural::train_data(brain,"train.txt",2,3,', compiler.emit(ir))

    def test_bad_source_and_mutated_ir(self):
        for bad in (TRAIN.replace("train.txt", "../train.txt"), TRAIN.replace("epochs 3", "epochs 0"),
                    TRAIN.replace("rate 0.1", "rate 1.01"), TRAIN.replace("seed 17", "seed 4294967296"),
                    TRAIN.replace("samples 2 epochs 3", "samples 10000 epochs 101"),
                    TRAIN.replace("samples 2", "samples 100001")):
            with self.subTest(source=bad), self.assertRaises(compiler.CompileError):
                compiler.parse(BRAIN + bad)
        ir = compiler.parse(BRAIN + TRAIN)
        for key, value in (("samples", True), ("epochs", 0), ("rate", float("inf")), ("seed", -1),
                           ("filename", "a/b"), ("samples", 99999)):
            edited = copy.deepcopy(ir)
            edited["operations"][0][key] = value
            with self.subTest(key=key), self.assertRaises(compiler.CompileError):
                compiler.emit(edited)

    def test_vector_target_contract_and_invalid_modes(self):
        source = BRAIN + 'train_data "train.txt" samples 4 epochs 80 rate 0.1 seed 17 targets vector\n'
        ir = compiler.parse(source)
        self.assertEqual(ir["shape_contract"]["operation_signatures"]["train_data"]["targets"],
                         {"dtype": "f64", "shape": [2]})
        self.assertNotIn("label", ir["shape_contract"]["operation_signatures"]["train_data"])
        self.assertIn('neural::train_data(brain,"train.txt",4,80,0.10000000000000001,17u,true)',
                      compiler.emit(ir))
        altered = copy.deepcopy(ir)
        altered["operations"][0]["target_mode"] = "something_else"
        with self.assertRaises(compiler.CompileError):
            compiler.emit(altered)
        altered = copy.deepcopy(ir)
        altered["operations"][0]["target_mode"] = False
        with self.assertRaises(compiler.CompileError):
            compiler.emit(altered)
        with self.assertRaises(compiler.CompileError):
            compiler.parse(BRAIN + 'predict_data "test.txt" samples 4 targets vector\n')
        # Single-output regression is valid; one-hot classification needs >=2 classes.
        single = "brain Data {\n senses 2\n neurons out 1 linear\n init 42\n}\n"
        self.assertIn("targets", compiler.parse(single + source.split("}\n")[-1])["shape_contract"]["operation_signatures"]["train_data"])
        with self.assertRaises(compiler.CompileError):
            compiler.parse(single + 'train_data "train.txt" samples 4 epochs 80 rate 0.1 seed 17\n')

    def test_native_vector_targets_learn_reload_and_bad_rows(self):
        regress = 'train_data "targets.txt" samples 4 epochs 80 rate 0.1 seed 17 targets vector\n'
        infer = 'predict_data "features.txt" samples 2\n'
        targets = "0 0 0 0\n0 1 0 1\n1 0 1 0\n1 1 1 1\n"
        # Withheld interior points; not present in the SGD training rows.
        features = "0.25 0.75\n0.8 0.2\n"
        for cxx in ("g++", "clang++"):
            if not shutil.which(cxx):
                self.skipTest(f"{cxx} unavailable")
            with self.subTest(compiler=cxx), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                trainer = self.build(root, "fit", BRAIN + regress + 'save_model "weights.nlw"\n', cxx)
                predict = self.build(root, "predict", BRAIN + 'load_model "weights.nlw"\n' + infer, cxx)
                for run_name in ("first", "repeat"):
                    wd = root / run_name
                    wd.mkdir()
                    (wd / "targets.txt").write_text(targets)
                    (wd / "features.txt").write_text(features)
                    learned = self.run_native(trainer, wd)
                    self.assertEqual(learned.returncode, 0, learned.stdout + learned.stderr)
                    self.assertIn("steps=320", learned.stdout)
                    prediction = self.run_native(predict, wd)
                    self.assertEqual(prediction.returncode, 0, prediction.stdout + prediction.stderr)
                    vals = [list(map(float, line.split("values=")[1].split(",")))
                            for line in prediction.stdout.splitlines() if "DATA_PREDICT index=" in line]
                    self.assertEqual(len(vals), 2)
                    for row, actual in zip(((0.25, 0.75), (0.8, 0.2)), vals):
                        self.assertLess(max(abs(got - goal) for got, goal in zip(actual, row)), 0.08)
                self.assertEqual((root / "first/weights.nlw").read_bytes(),
                                 (root / "repeat/weights.nlw").read_bytes())
                invalid = root / "invalid"
                invalid.mkdir()
                for content in ("0 0 0 0\n", "0 0 0 0\n1 1 1\n0 1 0 1\n1 0 1 0\n",
                                "0 0 0 0\n1 1 NaN 1\n0 1 0 1\n1 0 1 0\n",
                                "0 0 0 0\n1 1 1 1 2\n0 1 0 1\n1 0 1 0\n"):
                    (invalid / "targets.txt").write_text(content)
                    result = self.run_native(trainer, invalid)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("dataset", result.stderr)
                    self.assertFalse((invalid / "weights.nlw").exists())

    def build(self, root, name, source, cxx):
        path = root / (name + ".nir")
        path.write_text(source)
        result = subprocess.run(
            [sys.executable, str(LANG / "compiler.py"), str(path), "-o", str(root / name),
             "--compiler", cxx], text=True, capture_output=True, timeout=75,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return root / name

    def run_native(self, exe, cwd):
        return subprocess.run([str(exe)], cwd=cwd, text=True, capture_output=True, timeout=25)

    def test_native_train_repeat_load_and_bad_data(self):
        for cxx in ("g++", "clang++"):
            if not shutil.which(cxx):
                self.skipTest(f"{cxx} unavailable")
            with self.subTest(compiler=cxx), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                trainer = self.build(root, "trainer", BRAIN + TRAIN + 'save_model "weights.nlw"\n', cxx)
                predictor = self.build(root, "infer", BRAIN + 'load_model "weights.nlw"\n' + PREDICT, cxx)
                for name in ("first", "repeat"):
                    run = root / name
                    run.mkdir()
                    (run / "train.txt").write_text("0 1 0\n1 0 1\n")
                    (run / "test.txt").write_text("1 0\n0 1\n")
                    result = self.run_native(trainer, run)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("steps=6", result.stdout)
                    result = self.run_native(predictor, run)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.count("DATA_PREDICT index="), 2)
                self.assertEqual((root / "first/weights.nlw").read_bytes(),
                                 (root / "repeat/weights.nlw").read_bytes())
                bad = root / "bad"
                bad.mkdir()
                for data in ("0 1 0\n", "0 1 0\n1 0\n", "0 1 0\n2 0 1\n",
                             "0 1 0\n1 NaN 0\n", "0 1-1\n1 0 1\n", "0 1 0\n1 0 1 2\n", "0 1 0\n1 0 1\n0 0 0\n"):
                    (bad / "train.txt").write_text(data)
                    result = self.run_native(trainer, bad)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("dataset", result.stderr)
                    self.assertFalse((bad / "weights.nlw").exists())
                (bad / "weights.nlw").write_bytes((root / "first/weights.nlw").read_bytes()[:-12])
                (bad / "test.txt").write_text("1 0\n0 1\n")
                result = self.run_native(predictor, bad)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("DATA_PREDICT", result.stdout)
