"""Create tiny deterministic dummy token files for smoke tests only.

Do NOT use this for paper runs. Real runs require real pre-tokenized data at
`data/train_fixed_tokens.pt` and `data/val_fixed_tokens.pt`.
"""

from __future__ import annotations

import argparse
import os

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-out", default="data/train_fixed_tokens.pt")
    ap.add_argument("--val-out", default="data/val_fixed_tokens.pt")
    ap.add_argument("--vocab-size", type=int, default=50257)
    ap.add_argument("--train-tokens", type=int, default=2_000_000)
    ap.add_argument("--val-tokens", type=int, default=250_000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    g = torch.Generator(device="cpu")
    g.manual_seed(args.seed)
    train = torch.randint(0, args.vocab_size, (args.train_tokens,), dtype=torch.int32, generator=g)
    # Different seed stream because generator advanced; still deterministic.
    val = torch.randint(0, args.vocab_size, (args.val_tokens,), dtype=torch.int32, generator=g)

    os.makedirs(os.path.dirname(args.train_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.val_out) or ".", exist_ok=True)
    torch.save(train, args.train_out)
    torch.save(val, args.val_out)
    print(f"wrote dummy train: {args.train_out} tokens={args.train_tokens}")
    print(f"wrote dummy val  : {args.val_out} tokens={args.val_tokens}")
    print("WARNING: dummy data is for smoke tests only, not experiments")


if __name__ == "__main__":
    main()
