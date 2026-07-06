# Data card

## Dataset
- **Source**: C4 (en) — Common Crawl-derived corpus, tokenized to fixed token
  files by `scripts/prepare_c4_data.py`. (If a different corpus is used for a
  run, it is named in the run's `train_dataset_info.json`.)
- **Tokenizer**: GPT-2 BPE (vocab 50257) via tiktoken.
- **Format**: single 1-D integer tensor (`.pt`) or raw `.bin` + JSON sidecar
  (`raw_token_bin_v1`), dtype int16/int32/int64.
- **Splits**: disjoint train/val files; val is a held-out contiguous slice,
  never overlapping train (verified by `scripts/validate_data_pair.py`).

## Processing guarantees
- Deterministic, non-overlapping sequence slices of length `block_size + 1`;
  no shuffling, no augmentation, no sampling (see `src/data.py` docstring).
- Every run stores `train_dataset_info.json` / `val_dataset_info.json`
  containing: absolute path, file size, total tokens, sequence counts,
  dtype, min/max token id checks, and a content fingerprint (SHA-256 over
  deterministic windows) — sufficient to verify two runs saw identical data.

## Known limitations & risks
- C4 contains web text with the usual biases and quality issues; this project
  uses it for *relative architecture comparison*, not for deployable model
  quality claims.
- Single-corpus results are a stated limitation until the second-corpus
  ablation (EXPERIMENT_CARD run ladder) is complete.
- Token files are not redistributed in this repository; prepare scripts are
  provided for reproduction.

## Licensing
- C4 is distributed under the terms documented by AllenAI/Google; users must
  obtain it from official sources. This repository contains no corpus data.
