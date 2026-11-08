#!/usr/bin/env python3
"""Neural Language research assistant: a bounded, evidence-gated code proposal.

Not an LLM, autonomous agent, production optimizer or proof of generalization.
Uses the actual .nir compiler and native execution; never edits the input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from compiler import CompileError, parse
from reference_eval import ReferenceError, read_model, predict as reference_predict

PREDICTION = re.compile(r"^PREDICT version=\d+ values=([^\n]+)$", re.MULTILINE)
TRAIN_OPS = frozenset(("cycle", "observe", "learn"))


class ExperimentError(Exception):
    pass


def fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def finite_vector(value, size: int, context: str):
    if not isinstance(value, list) or len(value) != size:
        raise ExperimentError(f"{context}: expected vector of {size} numbers")
    if any(type(v) not in (float, int) or not math.isfinite(v) or abs(v) >= 1e9 for v in value):
        raise ExperimentError(f"{context}: values must be finite numbers within supported range")
    return [float(x) for x in value]


def collect_training(operations):
    # A narrow, inspectable experiment contract: every train target belongs to
    # the immediately preceding observation. No hidden evaluation operations.
    examples = []
    count = 0

    def visit(ops, multiplier=1):
        nonlocal count
        last = None
        for op in ops:
            if op['op'] not in TRAIN_OPS:
                raise ExperimentError('training source may contain only observe, learn and cycle')
            if op['op'] == 'cycle':
                visit(op['body'], multiplier * op['count'])
                last = None
            elif op['op'] == 'observe':
                last = op['values']
            else:
                if last is None:
                    raise ExperimentError('every learn must immediately follow its own observe')
                examples.append((tuple(last), tuple(op['values'])))
                count += multiplier
                last = None
    visit(operations)
    if not examples or count > 100000:
        raise ExperimentError('requires 1..100000 explicit learning steps')
    by_input = {}
    for x, y in examples:
        if x in by_input and by_input[x] != y:
            raise ExperimentError('training contains contradictory targets for one input')
        by_input[x] = y
    return by_input, count


def load_evaluation(path: Path, train_inputs, input_size: int, output_size: int):
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ExperimentError(f'cannot read evaluation JSON: {exc}') from exc
    if not isinstance(raw, dict) or set(raw) != {'validation', 'test'}:
        raise ExperimentError('evaluation must have exactly validation and test sets')
    result = {}
    used = set(train_inputs)
    for split in ('validation', 'test'):
        items = raw[split]
        if not isinstance(items, list) or not 4 <= len(items) <= 128:
            raise ExperimentError(f'{split}: requires 4..128 examples')
        parsed = []
        for k, item in enumerate(items):
            if not isinstance(item, dict) or set(item) != {'input', 'target'}:
                raise ExperimentError(f'{split}[{k}]: expected input and target')
            x = finite_vector(item['input'], input_size, f'{split}[{k}].input')
            y = finite_vector(item['target'], output_size, f'{split}[{k}].target')
            if tuple(x) in used:
                raise ExperimentError(f'{split}[{k}]: input leaks across training/validation/test')
            used.add(tuple(x))
            parsed.append({'input': x, 'target': y})
        result[split] = parsed
    return result


def numeric(v):
    return format(float(v), '.17g')


def render_vector(values):
    return '[' + ', '.join(numeric(x) for x in values) + ']'


def render_training_op(operation, indent=''):
    kind = operation['op']
    if kind == 'cycle':
        nested = '\n'.join(render_training_op(item, indent + '  ') for item in operation['body'])
        return f"{indent}cycle {operation['count']} {{\n{nested}\n{indent}}}"
    if kind == 'observe':
        return indent + 'observe ' + render_vector(operation['values'])
    if kind == 'learn':
        return indent + 'learn ' + render_vector(operation['values']) + ' rate ' + numeric(operation['rate'])
    raise ExperimentError('cannot render a non-training instruction')


def propose_branch(ir: dict, width: int):
    """A structured Neural IR transformation, NOT textual source substitution."""
    b = ir['brain']
    layers = b['neurons']
    if b['graph'] or len(layers) != 1 or layers[0]['activation'] != 'linear' or layers[0]['storage'] != 'float':
        raise ExperimentError('v0.5 proposal supports only a single FP64 linear output layer')
    if width < 2 or width > 32:
        raise ExperimentError('candidate width must be in 2..32')
    last = layers[0]
    if last['name'] in ('assist_direct', 'assist_relu'):
        raise ExperimentError('reserved assistant layer name already used')
    lines = ['# Generated by the bounded Neural Language assistant (research proposal).',
             '# Retains the same training operations, seed, output shape and original budget.',
             f'brain {b["name"]} {{',
             f'  senses {b["senses"]}',
             '  neurons assist_direct 2 linear from input',
             f'  neurons assist_relu {width} relu from input',
             f'  neurons {last["name"]} {last["size"]} linear from assist_direct assist_relu',
             f'  init {b["init"]}']
    if b['budget'] is not None:
        lines.append(f'  budget {b["budget"][0]} {b["budget"][1]}')
    lines.append('}')
    lines.extend(render_training_op(item) for item in ir['operations'])
    return '\n'.join(lines) + '\n'


def add_evaluation(source: str, evaluation, checkpoint_name: str):
    lines = [source.rstrip()]
    for split in ('validation', 'test'):
        for sample in evaluation[split]:
            lines.append('observe [' + ', '.join(numeric(x) for x in sample['input']) + ']')
            lines.append('predict')
    lines.append('save_model ' + json.dumps(checkpoint_name))
    return '\n'.join(lines) + '\n'


def compile_run(source: str, name: str, workspace: Path, compiler: str, count: int, timeout: int):
    path = workspace / (name + '.nir')
    path.write_text(source, encoding='utf-8')
    exe = workspace / name
    cmd = [sys.executable, str(Path(__file__).resolve().with_name('compiler.py')),
           str(path), '-o', str(exe), '--compiler', compiler]
    try:
        build = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ExperimentError(f'{name}: native compilation timed out') from exc
    (workspace / (name + '.compile.log')).write_text(build.stdout + '\n' + build.stderr, encoding='utf-8')
    if build.returncode:
        raise ExperimentError(f'{name}: compilation failed, see {name}.compile.log')
    try:
        run = subprocess.run([str(exe)], cwd=workspace, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ExperimentError(f'{name}: native run timed out') from exc
    (workspace / (name + '.run.log')).write_text(run.stdout + '\n' + run.stderr, encoding='utf-8')
    if run.returncode:
        raise ExperimentError(f'{name}: native execution failed, see {name}.run.log')
    results = []
    for match in PREDICTION.finditer(run.stdout):
        try:
            values = [float(v) for v in match.group(1).split(',')]
        except ValueError as exc:
            raise ExperimentError(f'{name}: malformed native prediction') from exc
        if len(values) != count or not all(math.isfinite(v) for v in values):
            raise ExperimentError(f'{name}: non-finite or wrong-shape native prediction')
        results.append(values)
    return results


def rmse(predictions, samples):
    if len(predictions) != len(samples):
        raise ExperimentError('prediction count mismatch')
    errors = [((p - y) ** 2) for pred, row in zip(predictions, samples)
              for p, y in zip(pred, row['target'])]
    value = math.sqrt(sum(errors) / len(errors))
    if not math.isfinite(value):
        raise ExperimentError('non-finite evaluation metric')
    return value


def evaluate_program(source, name, stage: Path, evaluation, compiler: str, timeout: int):
    checkpoint = name + '.nlw'
    program = add_evaluation(source, evaluation, checkpoint)
    num_out = len(evaluation['validation'][0]['target'])
    predictions = compile_run(program, name, stage, compiler, num_out, timeout)
    nval = len(evaluation['validation'])
    ntest = len(evaluation['test'])
    if len(predictions) != nval + ntest:
        raise ExperimentError(f'{name}: expected {nval + ntest} predictions; received {len(predictions)}')
    saved = stage / checkpoint
    if not saved.is_file():
        raise ExperimentError(f'{name}: native model was not saved')
    try:
        model = read_model(saved)
        examples = evaluation['validation'] + evaluation['test']
        oracle = [reference_predict(model, sample['input']) for sample in examples]
    except (OSError, ReferenceError, ValueError) as exc:
        raise ExperimentError(f'{name}: independent reference could not read the model') from exc
    maximum_error = max(abs(x - y) for pred, expected in zip(predictions, oracle)
                        for x, y in zip(pred, expected))
    maximum_scale = max(abs(x) for predicted in oracle for x in predicted)
    # Native stdout uses 10 significant digits, not the full FP64 representation.
    tolerance = 1e-7 * (1 + maximum_scale)
    if maximum_error > tolerance:
        raise ExperimentError(f'{name}: independent FP64 reference mismatch')
    return {
        'validation_rmse': rmse(predictions[:nval], evaluation['validation']),
        'test_rmse': rmse(predictions[nval:], evaluation['test']),
        'reference_max_abs_error': maximum_error,
        'reference_tolerance': tolerance,
        'reference_check': 'PASS',
    }


def run_experiment(source_path: Path, eval_path: Path, output: Path, *, width=8, compiler='g++',
                   min_gain=0.20, max_weights=4096, max_macs=4096, timeout=60):
    if output.exists() or output.is_symlink():
        raise ExperimentError('output directory already exists; refusing to overwrite')
    if not 0 < min_gain < 1:
        raise ExperimentError('minimum relative gain must be between 0 and 1')
    if max_weights <= 0 or max_macs <= 0 or not 5 <= timeout <= 120:
        raise ExperimentError('resource limits or timeout invalid')
    if shutil.which(compiler) is None:
        raise ExperimentError('requested native compiler not found')
    try:
        source_bytes = source_path.read_bytes()
        source = source_bytes.decode('utf-8')
        ir = parse(source)
    except (OSError, UnicodeError, CompileError) as exc:
        raise ExperimentError(f'invalid source .nir: {exc}') from exc
    train, nsteps = collect_training(ir['operations'])
    evaluation_bytes = eval_path.read_bytes()
    evaluation = load_evaluation(eval_path, train, ir['brain']['senses'], ir['brain']['neurons'][-1]['size'])
    proposed = propose_branch(ir, width)
    # Parse candidate independently: use all existing grammar/shape limits.
    candidate_status = 'PROPOSED'
    try:
        proposed_ir = parse(proposed)
    except CompileError as exc:
        if 'budget exceeded' in str(exc):
            candidate_status = 'REJECT_RESOURCE_LIMIT'
            proposed_ir = None
        else:
            raise ExperimentError(f'candidate fails Neural IR validation: {exc}') from exc
    if proposed_ir and (proposed_ir['brain']['weights'] > max_weights or proposed_ir['brain']['macs'] > max_macs):
        candidate_status = 'REJECT_RESOURCE_LIMIT'
    output_parent = output.absolute().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.nl_assist_stage_', dir=output_parent))
    try:
        (staging/'baseline_source.nir').write_bytes(source_bytes)
        (staging/'proposed_source.nir').write_text(proposed, encoding='utf-8')
        (staging/'evaluation.json').write_bytes(evaluation_bytes)
        baseline = evaluate_program(source, 'baseline', staging, evaluation, compiler, timeout)
        candidate = None
        if candidate_status == 'PROPOSED':
            candidate = evaluate_program(proposed, 'candidate', staging, evaluation, compiler, timeout)
            # The test split is NOT used for decision-making. It is reported only
            # as post-selection evidence to detect shortcomings of the policy.
            improved = (baseline['validation_rmse'] - candidate['validation_rmse'] > 1e-5
                        and candidate['validation_rmse'] <= baseline['validation_rmse'] * (1 - min_gain))
            candidate_status = 'ACCEPT' if improved else 'REJECT_NO_VALIDATION_GAIN'
        if candidate_status == 'ACCEPT':
            (staging/'recommended.nir').write_text(proposed, encoding='utf-8')
        record = {
            'schema': 'NeuralLanguageAssistant/0.5',
            'scope': 'single-output-layer baseline -> two-branch ReLU DAG candidate; deterministic offline search, not LLM',
            'source_sha256': fingerprint(source_bytes),
            'evaluation_sha256': fingerprint(evaluation_bytes),
            'candidate_sha256': fingerprint(proposed.encode('utf-8')),
            'compiler': compiler,
            'compiler_sha256': fingerprint(Path(__file__).resolve().with_name('compiler.py').read_bytes()),
            'native_runtime_sha256': fingerprint((Path(__file__).resolve().parent/'runtime'/'neural_runtime.hpp').read_bytes()),
            'assistant_sha256': fingerprint(Path(__file__).resolve().read_bytes()),
            'train_steps': nsteps,
            'training_unique_inputs': len(train),
            'validation_count': len(evaluation['validation']),
            'test_count': len(evaluation['test']),
            'policy': {'candidate_width': width, 'minimum_relative_validation_rmse_gain': min_gain,
                       'absolute_validation_rmse_gain_floor': 1e-5,
                       'max_weights': max_weights, 'max_macs_per_forward': max_macs,
                       'test_split_used_for_selection': False},
            'baseline': {'static_weights': ir['brain']['weights'], 'forward_macs': ir['brain']['macs'], **baseline},
            'candidate': ({'static_weights': proposed_ir['brain']['weights'], 'forward_macs': proposed_ir['brain']['macs'], **candidate}
                          if proposed_ir and candidate else None),
            'decision': candidate_status,
            'original_source_edited': False,
            'model_accepted_into_original_runtime': False,
            'limitations': ['Static resource estimates only, not RSS or energy limits',
                            'Train and validation sets may still share an underlying distribution or bias',
                            'Only one proposed architecture, seed and training schedule',
                            'No transfer learning; candidate trains from scratch',
                            'Test RMSE is reported AFTER validation-only selection; cannot prove generalization',
                            'No autonomous source mutation, safety sandbox, mobile tests or optimized DAG kernel'],
        }
        (staging/'evidence.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        # Do not overwrite existing output, even if another process races in.
        if output.exists() or output.is_symlink():
            raise ExperimentError('output directory appeared during the run')
        # Atomic exclusive directory creation: never replace a concurrent user's
        # existing output. The evidence file is published LAST, marking success.
        output.mkdir(mode=0o700, exist_ok=False)
        try:
            for item in sorted(staging.iterdir(), key=lambda x: (x.name == 'evidence.json', x.name)):
                os.rename(item, output / item.name)
        except BaseException:
            shutil.rmtree(output, ignore_errors=True)
            raise
        staging.rmdir()
        return record
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv=None):
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument('source', type=Path, help='a restricted .nir learning program')
    arg.add_argument('--evaluation', type=Path, required=True, help='JSON validation/test sets')
    arg.add_argument('--output-dir', type=Path, required=True, help='must not exist')
    arg.add_argument('--candidate-width', type=int, default=8)
    arg.add_argument('--minimum-gain', type=float, default=0.20)
    arg.add_argument('--max-weights', type=int, default=4096)
    arg.add_argument('--max-macs', type=int, default=4096)
    arg.add_argument('--timeout', type=int, default=60)
    arg.add_argument('--compiler', default='g++')
    args = arg.parse_args(argv)
    try:
        result = run_experiment(args.source, args.evaluation, args.output_dir, width=args.candidate_width,
                                compiler=args.compiler, min_gain=args.minimum_gain,
                                max_weights=args.max_weights, max_macs=args.max_macs, timeout=args.timeout)
        print('ASSISTANT DECISION:', result['decision'])
        print('EVIDENCE:', args.output_dir/'evidence.json')
        return 0
    except (ExperimentError, OSError, ValueError) as exc:
        print('ASSISTANT ERROR:', exc, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
