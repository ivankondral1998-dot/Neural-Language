#!/usr/bin/env python3
"""Isolated experimental .nir -> PyTorch causal-LM trainer.

Not the production Neural Language compiler, not UNO, not a conversationally
trained model. Its only purpose is to make a testable 55M-weight experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import torch
from torch import nn
from torch.nn import functional as F
from dialogue_tokenizer import ByteTokenizer, load_tokenizer, tokenizer_file

VOCAB = 260
BOS, USER, ASSISTANT, EOS = 256, 257, 258, 259
MAX_STREAM_BYTES = 16 * 1024 * 1024 * 1024
MAX_MATERIALIZED_BYTES = 16 * 1024 * 1024


def read_nir(path: Path):
    raw = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in raw if line and not line.startswith("#")]
    if not lines or not re.fullmatch(r"brain [A-Za-z_][A-Za-z0-9_]* \{", lines[0]):
        raise ValueError("expected brain Name {")
    if len(lines) != 11 or lines[8] != "}":
        raise ValueError("expected seven model directives, close brace, train and save")
    expected = ("tokens", "context", "width", "heads", "blocks", "feedforward", "seed")
    opts = {}
    for lineno, (line, key) in enumerate(zip(lines[1:8], expected), 2):
        pieces = line.split()
        if len(pieces) != 2 or pieces[0] != key:
            raise ValueError(f"line {lineno}: expected {key}")
        opts[key] = pieces[1]
    if opts["tokens"] not in ("byte260", "bpe4096"):
        raise ValueError("expected byte260 or bpe4096 tokenizer")
    cfg = {key: int(opts[key]) for key in expected[1:]}
    profile = {"context": 128, "width": 768, "heads": 12,
               "blocks": 10, "feedforward": 2048, "seed": 42}
    if opts["tokens"] == "bpe4096":
        profile.update(context=1024, feedforward=1824)
    if cfg != profile:
        raise ValueError("experiment accepts only the locked byte260 or bpe4096 profile")
    if opts["tokens"] == "bpe4096":
        cfg["tokenizer_kind"] = "bpe4096"
    m = re.fullmatch(r'train_dialogue "([A-Za-z0-9_.-]+)" steps (\d+) rate (0\.\d+)', lines[9])
    n = re.fullmatch(r'save_model "([A-Za-z0-9_.-]+\.pt)"', lines[10])
    if not m or not n or m.group(1) in ('.', '..'):
        raise ValueError("invalid training/output directive")
    steps, rate = int(m.group(2)), float(m.group(3))
    if not 1 <= steps <= 1000000 or not 0 < rate <= 0.01:
        raise ValueError("invalid steps or rate")
    return cfg, m.group(1), steps, rate, n.group(1)


class DecoderBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ff), nn.GELU(), nn.Linear(ff, width))

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


class ByteDialogue(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["width"]
        self.context = cfg["context"]
        self.embedding = nn.Embedding(cfg.get("vocab_size", VOCAB), d)
        self.position = nn.Embedding(self.context, d)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList([DecoderBlock(d, cfg["heads"], cfg["feedforward"])
                                     for _ in range(cfg["blocks"])])
        self.norm = nn.LayerNorm(d)
        # Output projection shares the embedding table: no fake/padding parameters.

    def forward(self, token_ids):
        b, n = token_ids.shape
        if n > self.context:
            raise ValueError("input exceeds context")
        positions = torch.arange(n, device=token_ids.device)
        x = self.embedding(token_ids) + self.position(positions)[None, :, :]
        mask = torch.ones((n, n), dtype=torch.bool, device=token_ids.device).triu(1)
        for block in self.blocks:
            x = block(x, mask)
        return F.linear(self.norm(x), self.embedding.weight)


def parameter_count(cfg):
    """Exact count for this tied-embedding architecture (no padding weights)."""
    d, f = cfg['width'], cfg['feedforward']
    return (cfg.get('vocab_size', VOCAB) + cfg['context']) * d + 2*d + cfg['blocks']*(
        4*d*d + 2*d*f + 9*d + f)


def checkpoint_tokenizer(package):
    tokenizer = load_tokenizer(package.get('tokenizer'))
    cfg = package['architecture']
    if (cfg.get('vocab_size', VOCAB) != tokenizer.vocab_size
            or cfg.get('tokenizer_kind', 'byte260') != tokenizer.kind):
        raise ValueError('checkpoint architecture/tokenizer mismatch')
    embedding = package['state_dict'].get('embedding.weight')
    if embedding is None or tuple(embedding.shape) != (tokenizer.vocab_size, cfg['width']):
        raise ValueError('checkpoint embedding/tokenizer mismatch')
    return tokenizer


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def iter_examples(path: Path, context: int, tokenizer=None):
    """Read bounded JSONL records; retain only the current tensor pair."""
    tokenizer = tokenizer or ByteTokenizer()
    maximum = MAX_STREAM_BYTES
    if not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("training file missing or too big")
    count, consumed = 0, 0
    with path.open('rb') as stream:
        while line := stream.readline(1024 * 1024 + 1):
            count += 1
            consumed += len(line)
            if len(line) > 1024 * 1024 or consumed > maximum:
                raise ValueError(f'row {count} or dataset exceeds input bound')
            try:
                row = json.loads(line.decode('utf-8'))
                user, assistant = row["user"], row["assistant"]
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"bad JSONL row {count}") from e
            if not isinstance(user, str) or not isinstance(assistant, str) or not user or not assistant:
                raise ValueError(f"bad message row {count}")
            prompt = [tokenizer.bos, tokenizer.user] + tokenizer.encode(user) + [tokenizer.assistant]
            response = tokenizer.encode(assistant) + [tokenizer.eos]
            sequence = prompt + response
            if len(sequence) - 1 > context:
                raise ValueError(f"row {count} exceeds {context}-token context")
            x = torch.tensor(sequence[:-1], dtype=torch.long)[None, :]
            labels = sequence[1:]
            labels[:len(prompt)-1] = [-100] * (len(prompt)-1)
            y = torch.tensor(labels, dtype=torch.long)[None, :]
            yield x, y
    if not count:
        raise ValueError("no training dialogue")


def examples(path: Path, context: int, tokenizer=None):
    """Compatibility helper for callers that explicitly want materialized pairs."""
    if not path.is_file() or path.stat().st_size > MAX_MATERIALIZED_BYTES:
        raise ValueError('materialized dataset missing or exceeds 16 MiB bound; use iter_examples')
    return list(iter_examples(path, context, tokenizer))


def train(nir: Path, outdir: Path, dry_run=False, no_save=False, tokenizer_path=None, resume=None):
    if dry_run and resume is not None:
        raise ValueError('--dry-run cannot validate an optimizer resume; omit --resume')
    cfg, datafile, steps, rate, output = read_nir(nir)
    tokenizer = ByteTokenizer()
    if cfg.get('tokenizer_kind') == 'bpe4096':
        if tokenizer_path is None:
            raise ValueError('bpe4096 profile requires explicit --tokenizer JSON')
        tokenizer = tokenizer_file(tokenizer_path)
        cfg['vocab_size'] = tokenizer.vocab_size
    elif tokenizer_path is not None:
        raise ValueError('byte260 profile does not accept a BPE tokenizer')
    saved = outdir / output
    if not dry_run and not no_save and (saved.exists() or saved.is_symlink()):
        raise ValueError('refusing checkpoint overwrite')
    dataset = nir.parent / datafile
    samples = sum(1 for _ in iter_examples(dataset, cfg['context'], tokenizer))
    dataset_digest = file_sha256(dataset)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(cfg["seed"])
    model = ByteDialogue(cfg)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != parameter_count(cfg):
        raise ValueError(f"unexpected parameter count: {count}")
    result = {"model": "ByteDialogue/experimental" if tokenizer.kind == 'byte260' else 'BPEDialogue/experimental', "architecture": cfg,
              "trainable_parameters": count, "within_55m_parameter_band": 55_000_000 <= count <= 56_000_000,
              "tokenizer": tokenizer.kind, "vocab_size": tokenizer.vocab_size,
              "tokenizer_sha256": getattr(tokenizer, 'digest', None),
              "samples": samples, "steps_requested": steps, "dataset_streamed": True,
              "source_sha256": hashlib.sha256(nir.read_bytes()).hexdigest(),
              "dataset_sha256": dataset_digest,
              "backend": f"torch {torch.__version__} CPU", "full_conversational_training": False}
    if dry_run:
        result["status"] = "shape_check_only"
        return result
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate)
    previous_steps = 0
    if resume is not None:
        package = torch.load(resume, map_location='cpu', weights_only=True)
        old_tokenizer = checkpoint_tokenizer(package)
        if package['architecture'] != cfg or old_tokenizer.metadata() != tokenizer.metadata():
            raise ValueError('resume architecture/tokenizer mismatch; conversion is not supported')
        if ('optimizer_state_dict' not in package or 'rng_state' not in package
                or package.get('dataset_sha256') != dataset_digest or package.get('training_rate') != rate):
            raise ValueError('exact resume requires optimizer, RNG, matching dataset and rate')
        model.load_state_dict(package['state_dict'], strict=True)
        optimizer.load_state_dict(package['optimizer_state_dict'])
        torch.set_rng_state(package['rng_state'])
        previous_steps = package['trained_steps']
        if type(previous_steps) is not int or previous_steps < 0:
            raise ValueError('invalid resume step count')
    result['previous_steps'] = previous_steps
    result['optimizer_state_resumed'] = resume is not None
    result['resume_semantics'] = 'same-tokenizer dataset order, AdamW and torch CPU RNG'
    pairs = iter_examples(dataset, cfg['context'], tokenizer)
    for _ in range(previous_steps % samples):
        next(pairs)
    start = time.perf_counter()
    for step in range(steps):
        try:
            x, y = next(pairs)
        except StopIteration:
            pairs = iter_examples(dataset, cfg['context'], tokenizer)
            x, y = next(pairs)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0:
            result["first_step_loss"] = float(loss.detach())
        result["last_step_loss"] = float(loss.detach())
    model.eval()
    with torch.no_grad():
        first_x, first_y = next(iter_examples(dataset, cfg['context'], tokenizer))
        result["post_train_loss_first_sample"] = float(F.cross_entropy(
            model(first_x).flatten(0, 1), first_y.flatten()))
    result["elapsed_sec"] = round(time.perf_counter() - start, 3)
    result["steps_completed"] = steps
    if not no_save:
        outdir.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also closes the race after the early existence check.
        with saved.open('xb') as stream:
            torch.save({"architecture": cfg, "state_dict": model.state_dict(),
                        "tokenizer": tokenizer.metadata(), "trained_steps": previous_steps + steps,
                        "optimizer_state_dict": optimizer.state_dict(), "rng_state": torch.get_rng_state(),
                        "dataset_sha256": dataset_digest, "training_rate": rate}, stream)
        result["checkpoint_bytes"] = saved.stat().st_size
        result["checkpoint_sha256"] = file_sha256(saved)
        result["checkpoint"] = str(saved)
    result["status"] = "one_or_more_optimizer_steps_not_conversational_training"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dialogue55m"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--tokenizer", type=Path, help='explicit portable JSON for bpe4096 profile')
    parser.add_argument("--resume", type=Path, help='exact same-tokenizer optimizer/RNG resume only')
    args = parser.parse_args()
    try:
        result = train(args.source, args.output_dir, args.dry_run, args.no_save, args.tokenizer, args.resume)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
