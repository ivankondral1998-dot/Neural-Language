#!/usr/bin/env python3
"""Inference check for experimental byte-level dialogue weights. Not PocketPal/GGUF."""
import argparse
from pathlib import Path
import torch
from train import ByteDialogue, checkpoint_tokenizer


def reply(path: Path, prompt: str, max_new_tokens: int = 32) -> str:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    tokenizer = checkpoint_tokenizer(checkpoint)
    cfg = checkpoint["architecture"]
    model = ByteDialogue(cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    sequence = [tokenizer.bos, tokenizer.user] + tokenizer.encode(prompt) + [tokenizer.assistant]
    if len(sequence) > cfg['context']:
        raise ValueError('prompt exceeds token context; truncation is not implicit')
    output = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(torch.tensor([sequence[-cfg["context"]:]], dtype=torch.long))[0, -1].clone()
            logits[tokenizer.bos] = logits[tokenizer.user] = logits[tokenizer.assistant] = float("-inf")
            token = int(logits.argmax())
            if token == tokenizer.eos:
                break
            output.append(token)
            sequence.append(token)
    return tokenizer.decode(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("prompt")
    p.add_argument("--tokens", type=int, default=16)
    args = p.parse_args()
    if not 1 <= args.tokens <= 1024:
        p.error("--tokens must be 1..1024")
    print(reply(args.checkpoint, args.prompt, args.tokens))
