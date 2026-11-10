#!/usr/bin/env python3
"""Evidence-gated continuation trainer for the isolated 55M byte dialogue model.

Adapts Assistant 0.5's propose/evaluate/accept-or-reject discipline.
Not a general agent or native .nir feature, and not a replacement for the
original assistant in language/assistant.py.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import torch
from torch.nn import functional as F
from train import ByteDialogue, examples, checkpoint_tokenizer, parameter_count, MAX_MATERIALIZED_BYTES


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for data in iter(lambda: f.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def pairs_from_jsonl(path):
    if not path.is_file() or path.stat().st_size > MAX_MATERIALIZED_BYTES:
        raise ValueError('assistant split missing or exceeds 16 MiB materialization bound')
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            raise ValueError('blank dataset record')
        obj = json.loads(line)
        if set(obj) != {'user', 'assistant'} or not all(isinstance(obj[k], str) and obj[k] for k in obj):
            raise ValueError('dataset must contain nonempty user and assistant strings')
        rows.append(obj)
    if len(rows) < 2:
        raise ValueError('each split needs at least two messages')
    return rows


def check_splits(train, validation, test):
    sets = [set(), set(), set()]
    for i, rows in enumerate((train, validation, test)):
        for row in rows:
            key = row['user'].strip().casefold()
            if key in sets[i]:
                raise ValueError('duplicate user prompt within one split')
            sets[i].add(key)
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('user prompt leakage between training/validation/test')


def measure(model, pairs, tokenizer_kind='byte260'):
    model.eval()
    loss_total = 0.0
    tokens_total = 0
    correct_total = 0
    with torch.inference_mode():
        for x, y in pairs:
            logits = model(x).flatten(0, 1)
            gold = y.flatten()
            token_mask = gold != -100
            losses = F.cross_entropy(logits, gold, reduction='none', ignore_index=-100)
            loss_total += float(losses.sum())
            tokens_total += int(token_mask.sum())
            correct_total += int(((logits.argmax(-1) == gold) & token_mask).sum())
    if tokens_total == 0:
        raise ValueError('no scored answer tokens')
    loss = loss_total / tokens_total
    if not math.isfinite(loss):
        raise ValueError('nonfinite validation loss')
    result = {'mean_answer_nll': loss, 'answer_token_accuracy': correct_total/tokens_total,
              'answer_tokens_scored': tokens_total, 'tokenizer_kind': tokenizer_kind}
    if tokenizer_kind == 'byte260':
        # Legacy aliases include EOS, as the original metric did.
        result.update(answer_byte_accuracy=correct_total/tokens_total,
                      answer_bytes_scored=tokens_total)
    return result


def run(checkpoint, train_path, val_path, test_path, output, steps=2, rate=5e-5,
        min_gain=0.01, save_candidate=True):
    if output.exists() or output.is_symlink():
        raise ValueError('refusing overwrite of existing output')
    if not 1 <= steps <= 8 or not 0 < rate <= 0.001 or not 0 < min_gain < 1:
        raise ValueError('invalid bounded experiment parameters')
    for path in (checkpoint, train_path, val_path, test_path):
        if not path.is_file():
            raise ValueError('missing input: ' + str(path))
    rows = [pairs_from_jsonl(p) for p in (train_path, val_path, test_path)]
    check_splits(*rows)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(20261110)
    package = torch.load(checkpoint, map_location='cpu', weights_only=True)
    tokenizer = checkpoint_tokenizer(package)
    cfg = package['architecture']
    model = ByteDialogue(cfg)
    model.load_state_dict(package['state_dict'], strict=True)
    train_pairs, val_pairs, test_pairs = [examples(p, cfg['context'], tokenizer) for p in (train_path,val_path,test_path)]
    params = sum(p.numel() for p in model.parameters())
    if params != parameter_count(cfg):
        raise ValueError('unexpected architecture parameter count')
    before = measure(model, val_pairs, tokenizer.kind)
    start = time.perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate)
    model.train()
    for i in range(steps):
        x, y = train_pairs[i % len(train_pairs)]
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            raise ValueError('nonfinite train loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    after = measure(model, val_pairs, tokenizer.kind)
    relative_gain = (before['mean_answer_nll'] - after['mean_answer_nll']) / before['mean_answer_nll']
    decision = 'ACCEPT' if relative_gain >= min_gain and before['mean_answer_nll'] - after['mean_answer_nll'] > 1e-4 else 'REJECT_NO_VALIDATION_GAIN'
    test_after = measure(model, test_pairs, tokenizer.kind)  # reported only, NOT used for selection
    evidence = {
        'assistant_method': 'NeuralLanguageAssistant/0.5 evidence gate adapted for byte dialogue',
        'original_assistant_invoked_on_transformer': False,
        'status': 'short_finetuning_experiment_only',
        'architecture_parameters': params, 'baseline_sha256': sha256(checkpoint),
        'tokenizer_kind': tokenizer.kind, 'vocab_size': tokenizer.vocab_size,
        'tokenizer_sha256': getattr(tokenizer, 'digest', None),
        'train_sha256': sha256(train_path), 'validation_sha256': sha256(val_path), 'test_sha256': sha256(test_path),
        'rows': {'train': len(rows[0]), 'validation': len(rows[1]), 'test': len(rows[2])},
        'optimizer_steps': steps, 'new_optimizer_state': True, 'rate': rate,
        'baseline_validation': before, 'candidate_validation': after,
        'relative_validation_gain': relative_gain, 'min_gain': min_gain,
        'candidate_test_after_decision': test_after, 'test_used_for_decision': False,
        'decision': decision, 'elapsed_training_and_eval_sec': round(time.perf_counter()-start, 3),
        'conversational_ability_proven': False,
        'limitations': ['tiny synthetic corpus', 'no human conversational evaluation',
                        'no optimizer state from original checkpoint',
                        'no independent numerical forward oracle', 'PyTorch not native .nir'],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.nl55_candidate_', dir=output.parent))
    try:
        if decision == 'ACCEPT' and save_candidate:
            target = stage / 'candidate.pt'
            torch.save({'architecture': cfg, 'state_dict': model.state_dict(),
                        'tokenizer': tokenizer.metadata(), 'trained_steps': package.get('trained_steps',0)+steps,
                        'continuation_mode': 'fresh_optimizer_evidence_gate'}, target)
            evidence['candidate_bytes'] = target.stat().st_size
            evidence['candidate_sha256'] = sha256(target)
        (stage/'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        if output.exists() or output.is_symlink():
            raise ValueError('output appeared during experiment')
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--rate', type=float, default=5e-5)
    parser.add_argument('--minimum-gain', type=float, default=0.01)
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()
    try:
        result = run(args.checkpoint, args.train, args.validation, args.test,
                     args.output, args.steps, args.rate, args.minimum_gain, not args.no_save)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f'ASSISTANT ERROR: {exc}\n')
    print('ASSISTANT DECISION:', result['decision'])
    print('EVIDENCE:', args.output/'evidence.json')

if __name__ == '__main__':
    main()
