#!/usr/bin/env python3
"""Inference check for experimental byte-level dialogue weights. Not PocketPal/GGUF."""
import argparse
from pathlib import Path
import torch
from train import ByteDialogue, BOS, USER, ASSISTANT, EOS


def reply(path: Path, prompt: str, max_new_tokens: int = 32) -> str:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg = checkpoint["architecture"]
    model = ByteDialogue(cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    sequence = [BOS, USER] + list(prompt.encode("utf-8")) + [ASSISTANT]
    output = bytearray()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(torch.tensor([sequence[-cfg["context"]:]], dtype=torch.long))[0, -1].clone()
            logits[BOS] = logits[USER] = logits[ASSISTANT] = float("-inf")
            token = int(logits.argmax())
            if token == EOS:
                break
            output.append(token)
            sequence.append(token)
    return output.decode("utf-8", errors="replace")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("prompt")
    p.add_argument("--tokens", type=int, default=16)
    args = p.parse_args()
    if not 1 <= args.tokens <= 128:
        p.error("--tokens must be 1..128")
    print(reply(args.checkpoint, args.prompt, args.tokens))
