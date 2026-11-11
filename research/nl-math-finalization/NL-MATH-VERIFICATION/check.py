#!/usr/bin/env python3
"""Независимые малые контрпримеры и алгебраические проверки; без обучения."""
import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction

results = {}
a = 1e20
b = math.nextafter(a, math.inf)
scale = max(abs(a), abs(b))
y = [a / scale, b / scale]
mu = math.fsum(y) / 2
var = math.fsum((v - mu) ** 2 for v in y) / 2
old = [(v - mu) / math.hypot(math.sqrt(var), math.sqrt(1e-5) / scale) for v in y]
with localcontext() as context:
    context.prec = 120
    da, db = Decimal.from_float(a), Decimal.from_float(b)
    midpoint = (da + db) / 2
    variance = ((da - midpoint) ** 2 + (db - midpoint) ** 2) / 2
    expected = [float((v - midpoint) / (variance + Decimal.from_float(1e-5)).sqrt()) for v in [da, db]]
assert max(abs(x - y) for x, y in zip(old, expected)) > 0.9
results['E01'] = {'actual_old_expression': old, 'decimal_reference': expected}
try:
    math.fsum([1e308, 1e308]) / 2
    raise AssertionError('expected overflow')
except OverflowError:
    results['E02'] = {'old_unnormalized': 'OverflowError', 'exact_answer': 1e308}
small = math.log1p(math.exp(-40.0))
assert math.log(math.fsum([1.0, math.exp(-40.0)])) == 0.0 and small > 0
results['E03'] = {'old': 0, 'reference': small}
integers = [2**53 + 1, -(2**53)]
assert sum(Fraction(v) for v in integers) == 1
assert sum(Fraction.from_float(float(v)) for v in integers) == 0
results['E04'] = {'exact_integer': 1, 'after_float_cast': 0}
w = [9e8, 1.0, -9e8]
def row(x):
    value = 0.0
    for a, b in zip(w, x):
        value += a * b
    return value
before = row([9e8, 0.0, 9e8])
assert row([9e8, 1.0, 9e8]) == 0.0 and before + 1.0 == 1.0
results['delta'] = {'full': 0, 'naive': 1}
assert not abs(0.0 - 1.5) <= 1 and abs(0.0 - 0.75) <= 1 and abs(0.75 - 1.5) <= 1
results['E09'] = 'tolerance is not transitive'
assert (0.0 - 0.1 * (0.3 * 0.7)) != (0.0 - (0.1 * 0.3) * 0.7)
results['E16'] = {'oracle_grouping': 0.0 - 0.1 * (0.3 * 0.7), 'native_grouping': 0.0 - (0.1 * 0.3) * 0.7}
print(json.dumps(results, indent=2, allow_nan=False))
