#!/usr/bin/env python3
"""Isolated experimental .nir -> PyTorch byte-causal-LM trainer.

Not the production Neural Language compiler, not UNO, not a conversationally
trained model. Its only purpose is to make a testable 55M-weight experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

VOCAB = 260
BOS, USER, ASSISTANT, EOS = 256, 257, 258, 259


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
    if opts["tokens"] != "byte260":
        raise ValueError("only byte260 tokenizer supported")
    cfg = {key: int(opts[key]) for key in expected[1:]}
    if cfg != {"context": 128, "width": 768, "heads": 12,
               "blocks": 10, "feedforward": 2048, "seed": 42}:
        raise ValueError("experiment accepts only the locked 55M architecture")
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
        self.embedding = nn.Embedding(VOCAB, d)
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


def examples(path: Path, context: int):
    if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("training file missing or too big")
    pairs = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
            user, assistant = row["user"], row["assistant"]
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"bad JSONL row {number}") from e
        if not isinstance(user, str) or not isinstance(assistant, str) or not user or not assistant:
            raise ValueError(f"bad message row {number}")
        prompt = [BOS, USER] + list(user.encode("utf-8")) + [ASSISTANT]
        response = list(assistant.encode("utf-8")) + [EOS]
        sequence = prompt + response
        if len(sequence) - 1 > context:
            raise ValueError(f"row {number} exceeds {context}-token context")
        x = torch.tensor(sequence[:-1], dtype=torch.long)[None, :]
        labels = sequence[1:]
        labels[:len(prompt)-1] = [-100] * (len(prompt)-1)
        y = torch.tensor(labels, dtype=torch.long)[None, :]
        pairs.append((x, y))
    if not pairs:
        raise ValueError("no training dialogue")
    return pairs


def train(nir: Path, outdir: Path, dry_run=False, no_save=False):
    cfg, datafile, steps, rate, output = read_nir(nir)
    pairs = examples(nir.parent / datafile, cfg["context"])
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(cfg["seed"])
    model = ByteDialogue(cfg)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not 55_000_000 <= count <= 56_000_000:
        raise ValueError(f"unexpected parameter count: {count}")
    result = {"model": "ByteDialogue/experimental", "architecture": cfg,
              "trainable_parameters": count, "tokenizer": "utf8 bytes + 4 delimiters",
              "samples": len(pairs), "steps_requested": steps,
              "source_sha256": hashlib.sha256(nir.read_bytes()).hexdigest(),
              "dataset_sha256": hashlib.sha256((nir.parent / datafile).read_bytes()).hexdigest(),
              "backend": f"torch {torch.__version__} CPU", "full_conversational_training": False}
    if dry_run:
        result["status"] = "shape_check_only"
        return result
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate)
    start = time.perf_counter()
    for step in range(steps):
        x, y = pairs[step % len(pairs)]
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
        first_x, first_y = pairs[0]
        result["post_train_loss_first_sample"] = float(F.cross_entropy(
            model(first_x).flatten(0, 1), first_y.flatten()))
    result["elapsed_sec"] = round(time.perf_counter() - start, 3)
    result["steps_completed"] = steps
    if not no_save:
        outdir.mkdir(parents=True, exist_ok=True)
        saved = outdir / output
        torch.save({"architecture": cfg, "state_dict": model.state_dict(),
                    "tokenizer": "byte260", "trained_steps": steps}, saved)
        result["checkpoint_bytes"] = saved.stat().st_size
        result["checkpoint_sha256"] = hashlib.sha256(saved.read_bytes()).hexdigest()
        result["checkpoint"] = str(saved)
    result["status"] = "one_or_more_optimizer_steps_not_conversational_training"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dialogue55m"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    try:
        result = train(args.source, args.output_dir, args.dry_run, args.no_save)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
