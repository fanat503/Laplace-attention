"""Sterile C4 -> GPT-2-token `.bin` dataset builder.

This script downloads/streams a pinned HuggingFace dataset split, tokenizes text
with a pinned tokenizer (`tiktoken` GPT-2 by default), and writes raw int32 token
files plus JSON sidecars accepted by `FixedDataset`.

Why C4 by default:
  - public, widely used LM pretraining corpus;
  - has train/validation splits;
  - deterministic streaming order when dataset revision is pinned;
  - large enough for 14B+ token studies.

Important:
  - This script does NOT shuffle. Order is dataset streaming order.
  - Pin `--revision` for paper runs.
  - Fill `DATA_CARD.md` with dataset/tokenizer details.
  - Output `.bin` files are intentionally used instead of giant `.pt` files for
    scalable, memory-mapped training.

Example for 700M/14B run:

  python scripts/prepare_c4_data.py \
    --dataset allenai/c4 --name en --revision <PINNED_REVISION> \
    --train-tokens 14013734400 \
    --val-tokens 20000000 \
    --out-dir data

The train token count above is the *stored token ids* requirement for the current
700M/14B config, not just effective training tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def atomic_write_json(path: str | os.PathLike[str], payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, p)


def sample_fingerprint(path: str, *, dtype: np.dtype, num_tokens: int) -> str:
    arr = np.memmap(path, mode="r", dtype=dtype, shape=(num_tokens,))
    h = hashlib.sha256()
    h.update(str(num_tokens).encode())
    h.update(str(dtype).encode())
    window = min(8192, num_tokens)
    starts = sorted(set([0, max(0, num_tokens // 2 - window // 2), max(0, num_tokens - window)]))
    for s in starts:
        h.update(str(s).encode())
        h.update(np.asarray(arr[s : s + window]).tobytes())
    return h.hexdigest()[:24]


def file_sha256(path: str, chunk_size: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def iter_texts(dataset_name: str, name: Optional[str], split: str, revision: Optional[str], text_field: str):
    ds = load_dataset(dataset_name, name, split=split, streaming=True, revision=revision)
    for ex in ds:
        text = ex.get(text_field)
        if isinstance(text, str) and text:
            yield text


def write_split(
    *,
    dataset_name: str,
    name: Optional[str],
    split: str,
    revision: Optional[str],
    text_field: str,
    tokenizer_name: str,
    target_tokens: int,
    out_path: str,
    add_eos: bool,
    full_hash: bool,
) -> Dict[str, Any]:
    enc = tiktoken.get_encoding(tokenizer_name)
    eos = enc.eot_token
    dtype = np.dtype("int32")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    arr = np.memmap(tmp_path, mode="w+", dtype=dtype, shape=(target_tokens,))

    pos = 0
    docs = 0
    empty_docs = 0
    truncated_last_doc = False
    stream_hash = hashlib.sha256()
    started = time.time()

    pbar = tqdm(total=target_tokens, unit="tok", desc=f"{split} -> {Path(out_path).name}")
    for text in iter_texts(dataset_name, name, split, revision, text_field):
        toks = enc.encode_ordinary(text)
        if add_eos:
            toks.append(eos)
        if not toks:
            empty_docs += 1
            continue
        remaining = target_tokens - pos
        if remaining <= 0:
            break
        if len(toks) > remaining:
            toks = toks[:remaining]
            truncated_last_doc = True
        chunk = np.asarray(toks, dtype=dtype)
        arr[pos : pos + len(chunk)] = chunk
        stream_hash.update(chunk.tobytes())
        pos += len(chunk)
        docs += 1
        pbar.update(len(chunk))
        if pos >= target_tokens:
            break
    pbar.close()
    arr.flush()
    del arr

    if pos != target_tokens:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise RuntimeError(
            f"Dataset split {split!r} ended before target_tokens: wrote={pos:,}, target={target_tokens:,}"
        )

    os.replace(tmp_path, out_path)
    meta = {
        "format": "raw_token_bin_v1",
        "dtype": "int32",
        "num_tokens": int(target_tokens),
        "dataset": dataset_name,
        "name": name,
        "split": split,
        "revision": revision,
        "text_field": text_field,
        "tokenizer": tokenizer_name,
        "vocab_size": enc.n_vocab,
        "eos_token": int(eos),
        "add_eos": bool(add_eos),
        "documents_consumed": int(docs),
        "empty_documents_skipped": int(empty_docs),
        "truncated_last_document": bool(truncated_last_doc),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": float(time.time() - started),
        "content_sha256_stream": stream_hash.hexdigest(),
        "sample_fingerprint": sample_fingerprint(out_path, dtype=dtype, num_tokens=target_tokens),
        "file_size_bytes": int(os.path.getsize(out_path)),
    }
    if full_hash:
        meta["file_sha256"] = file_sha256(out_path)
    atomic_write_json(f"{out_path}.json", meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="allenai/c4")
    ap.add_argument("--name", default="en")
    ap.add_argument("--revision", default=None, help="Pin this for paper runs, e.g. a HF commit hash")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="validation")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--train-tokens", type=int, required=True, help="Stored token ids to write for train")
    ap.add_argument("--val-tokens", type=int, default=20_000_000, help="Stored token ids to write for validation")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--train-out", default="train_fixed_tokens.bin")
    ap.add_argument("--val-out", default="val_fixed_tokens.bin")
    ap.add_argument("--no-eos", action="store_true")
    ap.add_argument("--full-hash", action="store_true", help="Compute full SHA256 of output files; slower")
    args = ap.parse_args()

    if args.revision is None:
        print("WARNING: --revision not set. For NeurIPS-grade runs, pin a HuggingFace commit hash.")
    out_dir = Path(args.out_dir)
    train_path = str(out_dir / args.train_out)
    val_path = str(out_dir / args.val_out)

    common = dict(
        dataset_name=args.dataset,
        name=args.name,
        revision=args.revision,
        text_field=args.text_field,
        tokenizer_name=args.tokenizer,
        add_eos=not args.no_eos,
        full_hash=args.full_hash,
    )
    train_meta = write_split(split=args.train_split, target_tokens=args.train_tokens, out_path=train_path, **common)
    val_meta = write_split(split=args.val_split, target_tokens=args.val_tokens, out_path=val_path, **common)

    manifest = {
        "schema_version": 1,
        "dataset": args.dataset,
        "name": args.name,
        "revision": args.revision,
        "tokenizer": args.tokenizer,
        "train": train_meta,
        "val": val_meta,
    }
    atomic_write_json(str(out_dir / "prepared_dataset_manifest.json"), manifest)
    print("DONE")
    print(f"train: {train_path}")
    print(f"val  : {val_path}")
    print(f"manifest: {out_dir / 'prepared_dataset_manifest.json'}")


if __name__ == "__main__":
    main()
