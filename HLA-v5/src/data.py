# Copyright 2026 Ivan Ivanov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.



"""
Sterile fixed-token dataset utilities for matched language-model experiments.

The dataset file must contain exactly one 1-D torch.Tensor of integer token ids.
Each example is a deterministic, non-overlapping slice of length `seq_len + 1`:

    input  = chunk[:-1]
    target = chunk[1:]

This module intentionally does not tokenize, shuffle, augment, sample randomly, or
silently cast malformed data. It is meant for paper-grade base-vs-HLA comparisons
where data order must be identical and reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


VALID_TOKEN_DTYPES = (torch.int16, torch.int32, torch.int64)


def _load_token_tensor(path: str, *, mmap: bool = True, weights_only: bool = True) -> torch.Tensor:
    """Load a 1-D token tensor from `.pt` or raw `.bin` + JSON sidecar.

    `.pt` files must contain one tensor and are loaded with torch.load.

    `.bin` files require a sidecar JSON at either:
      - `<path>.json`, or
      - `<stem>.json`

    Sidecar schema:
      {
        "format": "raw_token_bin_v1",
        "dtype": "int32",
        "num_tokens": 123,
        ...
      }
    """
    if path.endswith(".bin"):
        sidecars = [f"{path}.json", os.path.splitext(path)[0] + ".json"]
        meta_path = next((x for x in sidecars if os.path.exists(x)), None)
        if meta_path is None:
            raise FileNotFoundError(f"Raw .bin dataset requires sidecar JSON: tried {sidecars}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        dtype_name = str(meta.get("dtype", ""))
        dtype_map = {"int16": torch.int16, "int32": torch.int32, "int64": torch.int64}
        if dtype_name not in dtype_map:
            raise TypeError(f"Unsupported raw .bin dtype={dtype_name!r} in {meta_path}")
        num_tokens = int(meta["num_tokens"])
        expected_bytes = num_tokens * torch.empty((), dtype=dtype_map[dtype_name]).element_size()
        actual_bytes = os.path.getsize(path)
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Raw .bin size mismatch for {path}: actual_bytes={actual_bytes:,}, "
                f"expected_bytes={expected_bytes:,} from num_tokens={num_tokens:,}, dtype={dtype_name}"
            )
        return torch.from_file(path, shared=False, size=num_tokens, dtype=dtype_map[dtype_name])

    kwargs: Dict[str, Any] = {"map_location": "cpu"}
    if mmap:
        kwargs["mmap"] = True
    if weights_only:
        kwargs["weights_only"] = True
    try:
        obj = torch.load(path, **kwargs)
    except TypeError:
        # Older PyTorch versions may not support mmap or weights_only.
        kwargs.pop("mmap", None)
        kwargs.pop("weights_only", None)
        obj = torch.load(path, **kwargs)
    if isinstance(obj, dict) and "tokens" in obj:
        obj = obj["tokens"]
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {path!r}, got {type(obj).__name__}")
    return obj


@dataclass(frozen=True)
class FixedDatasetInfo:
    path: str
    file_size_mb: float
    total_tokens: int
    seq_len: int
    block_size: int
    n_sequences: int
    tokens_used: int
    tokens_dropped: int
    dtype: str
    sample_min: int
    sample_max: int
    full_min: Optional[int] = None
    full_max: Optional[int] = None
    expected_vocab_size: Optional[int] = None
    sample_fingerprint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


class FixedDataset(Dataset):
    """Deterministic non-overlapping fixed-token LM dataset.

    Invariants:
      - no randomness in __getitem__;
      - idx maps to tokens[idx * (seq_len+1) : (idx+1) * (seq_len+1)];
      - incomplete final block is dropped;
      - only integer token tensors are accepted;
      - optional vocab/range validation catches corrupted token files;
      - returns `{"input_ids": chunk}` by default for trainer compatibility.

    Args:
        path: .pt file containing one 1-D integer tensor.
        seq_len: model context length. Dataset block is seq_len + 1.
        expected_vocab_size: if provided, token ids must be < expected_vocab_size
            in the checked windows, and in the full tensor when validate_full=True.
        validate_full: if True, scans the full tensor for min/max. This is the
            most sterile option but may be slow on multi-billion-token files.
        sample_windows: number of deterministic windows used for cheap validation
            when validate_full=False. Windows cover beginning/middle/end.
        mmap: use torch.load(..., mmap=True) when available.
        return_dict: return {"input_ids": chunk} if True; otherwise return chunk.
        cast_to_long: if True, each item is returned as torch.long. For high-throughput
            DataLoader use cast_to_long=False and `fixed_token_collate` to cast once per batch.
    """

    VALID_DTYPES = VALID_TOKEN_DTYPES

    def __init__(
        self,
        path: str,
        seq_len: int,
        *,
        expected_vocab_size: Optional[int] = None,
        validate_full: bool = False,
        sample_windows: int = 8,
        mmap: bool = True,
        return_dict: bool = True,
        cast_to_long: bool = True,
    ):
        if not isinstance(path, str) or path == "":
            raise ValueError("path must be a non-empty string")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        if int(seq_len) <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if expected_vocab_size is not None and int(expected_vocab_size) <= 0:
            raise ValueError(f"expected_vocab_size must be positive, got {expected_vocab_size}")

        self.path = path
        self.seq_len = int(seq_len)
        self.block = self.seq_len + 1
        self.expected_vocab_size = None if expected_vocab_size is None else int(expected_vocab_size)
        self.return_dict = bool(return_dict)
        self.cast_to_long = bool(cast_to_long)

        self.tokens = _load_token_tensor(path, mmap=mmap, weights_only=True)

        if self.tokens.ndim != 1:
            raise ValueError(f"Expected 1-D token tensor in {path}, got shape={tuple(self.tokens.shape)}")
        if self.tokens.dtype not in self.VALID_DTYPES:
            raise TypeError(f"Expected integer token dtype {self.VALID_DTYPES}, got {self.tokens.dtype}")
        if self.tokens.device.type != "cpu":
            raise ValueError(f"Dataset tensor must be CPU-backed, got device={self.tokens.device}")
        if len(self.tokens) < self.block:
            raise ValueError(f"Dataset has {len(self.tokens):,} tokens; need at least {self.block:,}")

        self.n_sequences = len(self.tokens) // self.block
        self.n_tokens_used = self.n_sequences * self.block
        self.n_tokens_dropped = len(self.tokens) - self.n_tokens_used

        sample_min, sample_max = self._sample_range_check(max(1, int(sample_windows)))
        full_min = full_max = None
        if validate_full:
            # Full validation scans the whole tensor. Use only for preflight or
            # smaller files if startup time matters.
            full_min = int(self.tokens.min().item())
            full_max = int(self.tokens.max().item())
            self._validate_range(full_min, full_max, scope="full tensor")

        self.info = FixedDatasetInfo(
            path=os.path.abspath(path),
            file_size_mb=os.path.getsize(path) / (1024 ** 2),
            total_tokens=int(len(self.tokens)),
            seq_len=self.seq_len,
            block_size=self.block,
            n_sequences=int(self.n_sequences),
            tokens_used=int(self.n_tokens_used),
            tokens_dropped=int(self.n_tokens_dropped),
            dtype=str(self.tokens.dtype),
            sample_min=int(sample_min),
            sample_max=int(sample_max),
            full_min=full_min,
            full_max=full_max,
            expected_vocab_size=self.expected_vocab_size,
            sample_fingerprint=self._sample_fingerprint(),
        )

    def _validate_range(self, min_val: int, max_val: int, *, scope: str) -> None:
        if min_val < 0:
            raise ValueError(f"Negative token ids in {scope}: min={min_val}")
        if self.expected_vocab_size is not None and max_val >= self.expected_vocab_size:
            raise ValueError(
                f"Token id out of vocab in {scope}: max={max_val}, "
                f"expected_vocab_size={self.expected_vocab_size}"
            )

    def _sample_range_check(self, sample_windows: int) -> tuple[int, int]:
        n = len(self.tokens)
        window = min(4096, n)
        if sample_windows <= 1:
            starts = [0]
        else:
            max_start = max(0, n - window)
            starts = sorted(set(int(round(i * max_start / (sample_windows - 1))) for i in range(sample_windows)))
        mins = []
        maxs = []
        for s in starts:
            chunk = self.tokens.narrow(0, s, min(window, n - s))
            mins.append(int(chunk.min().item()))
            maxs.append(int(chunk.max().item()))
        sample_min = min(mins)
        sample_max = max(maxs)
        self._validate_range(sample_min, sample_max, scope=f"{len(starts)} deterministic sample windows")
        return sample_min, sample_max

    def _sample_fingerprint(self) -> str:
        """Stable SHA256 over deterministic small windows, not a full-file hash."""
        n = len(self.tokens)
        window = min(8192, n)
        starts = sorted(set([0, max(0, n // 2 - window // 2), max(0, n - window)]))
        h = hashlib.sha256()
        h.update(str(n).encode())
        h.update(str(self.tokens.dtype).encode())
        h.update(str(self.seq_len).encode())
        for s in starts:
            chunk = self.tokens.narrow(0, s, min(window, n - s)).contiguous()
            h.update(str(s).encode())
            h.update(chunk.numpy().tobytes())
        return h.hexdigest()[:24]

    def __len__(self) -> int:
        return int(self.n_sequences)

    def __getitem__(self, idx: int):
        if isinstance(idx, torch.Tensor):
            idx = int(idx.item())
        idx = int(idx)
        if idx < 0:
            idx += self.n_sequences
        if idx < 0 or idx >= self.n_sequences:
            raise IndexError(f"FixedDataset index out of range: idx={idx}, len={self.n_sequences}")
        start = idx * self.block
        chunk = self.tokens.narrow(0, start, self.block)
        if self.cast_to_long:
            # Compatibility path. High-throughput training casts once per batch in
            # fixed_token_collate instead of once per sample.
            chunk = chunk.to(torch.long)
        if self.return_dict:
            return {"input_ids": chunk}
        return chunk

    def summary(self) -> str:
        d = self.info.to_dict()
        lines = ["FixedDataset("]
        for k in [
            "path",
            "file_size_mb",
            "total_tokens",
            "seq_len",
            "block_size",
            "n_sequences",
            "tokens_used",
            "tokens_dropped",
            "dtype",
            "sample_min",
            "sample_max",
            "full_min",
            "full_max",
            "expected_vocab_size",
            "sample_fingerprint",
        ]:
            lines.append(f"  {k:20s}= {d[k]}")
        lines.append(")")
        return "\n".join(lines)


def worker_init_fn(worker_id: int) -> None:
    """Deterministic worker seeding if num_workers > 0 is enabled."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def fixed_token_collate(batch):
    if not batch:
        raise ValueError("empty batch")
    if isinstance(batch[0], dict):
        xs = [b["input_ids"] for b in batch]
    else:
        xs = batch
    return {"input_ids": torch.stack(xs, dim=0).to(torch.long)}


def get_dataloader(
    path: str,
    seq_len: int,
    batch_size: int,
    drop_last: bool,
    *,
    seed: int = 0,
    expected_vocab_size: Optional[int] = None,
    validate_full: bool = False,
    num_workers: int = 0,
    print_summary: bool = False,
) -> DataLoader:
    ds = FixedDataset(
        path,
        seq_len,
        expected_vocab_size=expected_vocab_size,
        validate_full=validate_full,
        return_dict=True,
        cast_to_long=False,
    )
    if print_summary:
        print(ds.summary(), flush=True)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=fixed_token_collate,
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )


__all__ = [
    "FixedDataset",
    "FixedDatasetInfo",
    "get_dataloader",
    "worker_init_fn",
    "fixed_token_collate",
    "VALID_TOKEN_DTYPES",
]
