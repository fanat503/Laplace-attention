"""Validate a training CSV log for monotonicity and finite critical metrics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_float(x: str) -> float:
    if x in {"", "nan", "NaN", "inf", "-inf"}:
        return float(x) if x else float("nan")
    return float(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--allow-nan-val", action="store_true")
    args = ap.parse_args()

    path = Path(args.csv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    header = None
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if header is None:
                header = row
            else:
                rows.append(dict(zip(header, row)))
    if header is None:
        raise RuntimeError("no CSV header found")
    if not rows:
        raise RuntimeError("no data rows found")

    prev_step = -1
    prev_tokens = -1
    for i, r in enumerate(rows):
        step = int(r["step"])
        tokens = int(r["tokens_seen"])
        if step <= prev_step:
            raise RuntimeError(f"non-monotonic step at row {i}: {step} <= {prev_step}")
        if tokens <= prev_tokens:
            raise RuntimeError(f"non-monotonic tokens at row {i}: {tokens} <= {prev_tokens}")
        prev_step, prev_tokens = step, tokens
        train_loss = parse_float(r["train_loss"])
        if not math.isfinite(train_loss):
            raise RuntimeError(f"non-finite train_loss at row {i}: {r['train_loss']}")
        val_loss = parse_float(r.get("val_loss", "nan"))
        if not args.allow_nan_val and not math.isfinite(val_loss):
            raise RuntimeError(f"non-finite val_loss at row {i}: {r.get('val_loss')}")
    print(f"LOG VALID: rows={len(rows)} final_step={prev_step} final_tokens={prev_tokens}")


if __name__ == "__main__":
    main()
