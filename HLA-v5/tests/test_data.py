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



"""Data pipeline tests: determinism, validation, sharding-compatible shapes."""
from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import FixedDataset, fixed_token_collate, get_dataloader  # noqa: E402


@pytest.fixture()
def token_file(tmp_path):
    tokens = (torch.arange(0, 1000, dtype=torch.int32) % 250)
    p = str(tmp_path / "tokens.pt")
    torch.save(tokens, p)
    return p


class TestFixedDataset:
    def test_len_and_non_overlap(self, token_file):
        ds = FixedDataset(token_file, seq_len=9, expected_vocab_size=250)
        assert len(ds) == 100  # 1000 // (9+1)
        a = ds[0]["input_ids"]
        b = ds[1]["input_ids"]
        assert a.shape == (10,)
        assert int(a[-1]) + 1 == int(b[0]), "chunks must be adjacent, non-overlapping"

    def test_deterministic(self, token_file):
        ds1 = FixedDataset(token_file, seq_len=9)
        ds2 = FixedDataset(token_file, seq_len=9)
        for i in (0, 7, 42, 99):
            assert torch.equal(ds1[i]["input_ids"], ds2[i]["input_ids"])

    def test_out_of_vocab_rejected_full_scan(self, token_file):
        with pytest.raises(ValueError):
            FixedDataset(token_file, seq_len=9, expected_vocab_size=100, validate_full=True)

    def test_float_tensor_rejected(self, tmp_path):
        p = str(tmp_path / "bad.pt")
        torch.save(torch.rand(100), p)
        with pytest.raises(TypeError):
            FixedDataset(p, seq_len=9)

    def test_2d_tensor_rejected(self, tmp_path):
        p = str(tmp_path / "bad2d.pt")
        torch.save(torch.ones(10, 10, dtype=torch.int32), p)
        with pytest.raises(ValueError):
            FixedDataset(p, seq_len=3)

    def test_too_short_rejected(self, tmp_path):
        p = str(tmp_path / "short.pt")
        torch.save(torch.ones(5, dtype=torch.int32), p)
        with pytest.raises(ValueError):
            FixedDataset(p, seq_len=9)

    def test_negative_index(self, token_file):
        ds = FixedDataset(token_file, seq_len=9)
        assert torch.equal(ds[-1]["input_ids"], ds[len(ds) - 1]["input_ids"])

    def test_fingerprint_stable(self, token_file):
        f1 = FixedDataset(token_file, seq_len=9).info.sample_fingerprint
        f2 = FixedDataset(token_file, seq_len=9).info.sample_fingerprint
        assert f1 == f2


class TestRawBin:
    def test_bin_with_sidecar(self, tmp_path):
        tokens = (torch.arange(0, 500, dtype=torch.int32) % 250)
        p = str(tmp_path / "tokens.bin")
        tokens.numpy().tofile(p)
        import json
        with open(p + ".json", "w") as f:
            json.dump({"format": "raw_token_bin_v1", "dtype": "int32", "num_tokens": 500}, f)
        ds = FixedDataset(p, seq_len=9, expected_vocab_size=250)
        assert len(ds) == 50

    def test_bin_size_mismatch_rejected(self, tmp_path):
        tokens = (torch.arange(0, 500, dtype=torch.int32) % 250)
        p = str(tmp_path / "tokens.bin")
        tokens.numpy().tofile(p)
        import json
        with open(p + ".json", "w") as f:
            json.dump({"format": "raw_token_bin_v1", "dtype": "int32", "num_tokens": 400}, f)
        with pytest.raises(ValueError):
            FixedDataset(p, seq_len=9)

    def test_bin_missing_sidecar_rejected(self, tmp_path):
        p = str(tmp_path / "tokens.bin")
        torch.arange(0, 100, dtype=torch.int32).numpy().tofile(p)
        with pytest.raises(FileNotFoundError):
            FixedDataset(p, seq_len=9)


class TestDataloader:
    def test_batch_shape_dtype(self, token_file):
        dl = get_dataloader(token_file, seq_len=9, batch_size=4, drop_last=True,
                            expected_vocab_size=250)
        b = next(iter(dl))
        assert b["input_ids"].shape == (4, 10)
        assert b["input_ids"].dtype == torch.long

    def test_collate_casts_once(self):
        batch = [{"input_ids": torch.ones(10, dtype=torch.int32)} for _ in range(3)]
        out = fixed_token_collate(batch)
        assert out["input_ids"].dtype == torch.long
        assert out["input_ids"].shape == (3, 10)

    def test_order_is_sequential(self, token_file):
        dl = get_dataloader(token_file, seq_len=9, batch_size=2, drop_last=True)
        first = next(iter(dl))["input_ids"]
        assert int(first[0, 0]) == 0
        assert int(first[1, 0]) == 10
