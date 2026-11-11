#!/usr/bin/env python3
"""Проверка сохранённых Python-эталонов по оригинальному NL-MATH-006 ZIP."""
import io
import json
import math
import sys
import types
import zipfile
from decimal import Decimal, localcontext

def read_zip(blob, name):
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        return archive.read(name)

def load(name, data):
    module = types.ModuleType(name)
    module.__file__ = name + '.py'
    sys.modules[name] = module
    exec(compile(data, module.__file__, 'exec'), module.__dict__)
    return module

with open(sys.argv[1], 'rb') as source:
    a006 = source.read()
a005 = read_zip(a006, 'sources/NL_MATH_005_INDEPENDENT_AUDIT_20261111.zip')
a003 = read_zip(a005, 'sources/NL_MATH_003_OWNER_REVIEW_20261111.zip')
a001 = read_zip(a003, 'original/Neural_Language_Math_001.zip')
a002 = read_zip(a003, 'original/Neural_Language_Math_002.zip')
core = load('nl_oracle', read_zip(a001, 'math_oracle.py'))
helpers = load('nl_helpers', read_zip(a002, 'neural_language_math_002/assistant_math.py'))
fixed = load('nl_corrected', read_zip(a006, 'numeric_corrections.py'))
a = 1e20
b = math.nextafter(a, math.inf)
actual = fixed.layernorm_stable([a, b], [1, 1], [0, 0], 1e-5)
with localcontext() as ctx:
    ctx.prec = 120
    vals = [Decimal.from_float(a), Decimal.from_float(b)]
    mean = sum(vals) / 2
    var = sum((x - mean)**2 for x in vals) / 2
    expected = [float((x - mean) / (var + Decimal.from_float(1e-5)).sqrt()) for x in vals]
assert max(abs(x-y) for x,y in zip(actual, expected)) > .9
try:
    fixed.attention_real_dot_tiny([[0.0], [0.0]], [[0.0], [0.0]], [[1e308], [1e308]])
    raise AssertionError('E02 did not reproduce')
except OverflowError:
    pass
assert fixed.nll_stable([0, -40], 0) == 0 and math.log1p(math.exp(-40)) > 0
assert fixed.real_dot([2**53 + 1, -(2**53)], [1, 1]) == 0
graph = core.Graph((core.Layer((-1,), ((0.,),), (0.,), ((True,),)),), 1)
cache = core.full(graph, (0.,))
for bad in (1e9, float('nan')):
    try:
        core.full(graph, (bad,))
        raise AssertionError('full accepted invalid value')
    except ValueError:
        pass
    assert core.sparse(graph, (0.,), cache, graph, (bad,))[1][0][0] == 0
p = helpers.ImmutableProposal('h', 2**64, 'a'*64, 'b'*64, 'c'*64, 1, 1)
assert helpers.commit_candidate(2**64, 'a'*64, p)['status'] == 'ACCEPT_FOR_REVIEW'
p = helpers.ImmutableProposal('h', 0, 'a'*64, 'b'*64, 'c'*64, float('inf'), float('inf'))
assert helpers.commit_candidate(0, 'a'*64, p)['status'] == 'ACCEPT_FOR_REVIEW'
print(json.dumps({'E01': {'actual': actual, 'decimal_reference': expected}, 'E02': 'OverflowError', 'E03': 'loss rounded to zero', 'E04': 'integer input rounded before exact dot', 'E05': 'FULL rejects, sparse accepts', 'E06': 'invalid revision and infinite checks accepted'}, indent=2))
