"""Neural Language 0.3 foundation: deterministic declarations and old-program ABI."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compiler import CompileError, emit, parse


class SeedContractTests(unittest.TestCase):
    BASIC = '''brain Core {
 senses 1
 neurons result 1 linear
%s}
observe [0]
predict'''

    def test_duplicate_seed_rejected_before_any_code_generation(self):
        source = self.BASIC % ' init 4\n init 9\n'
        with self.assertRaisesRegex(CompileError, 'duplicate init'):
            parse(source)

    def test_repeating_the_same_seed_is_also_rejected(self):
        source = self.BASIC % ' init 4\n init 4\n'
        with self.assertRaisesRegex(CompileError, 'duplicate init'):
            parse(source)

    def test_explicit_seed_still_applies(self):
        ir = parse(self.BASIC % ' init 41\n')
        self.assertEqual(ir['brain']['init'], 41)
        self.assertIn('static_cast<uint32_t>(41ULL+0*31337ULL)', emit(ir))

    def test_default_seed_does_not_change(self):
        ir = parse(self.BASIC % '')
        self.assertEqual(ir['brain']['init'], 11)

    def test_existing_trainer_program_preserves_ir_format(self):
        source = (ROOT / 'examples/integrated.nir').read_text(encoding='utf-8')
        ir = parse(source)
        self.assertEqual(ir['ir_version'], 'NeuralIR/0.2')
        self.assertEqual(ir['brain']['init'], 17)


if __name__ == '__main__':
    unittest.main()
