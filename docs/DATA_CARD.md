# Data card

## Dataset
- Source: C4 (en) — Common Crawl-derived corpus, tokenized by `scripts/prepare_c4_data.py`.
- Tokenizer: GPT-2 BPE via tiktoken.
- Format: single 1-D integer tensor (`.pt`) or raw `.bin` + JSON sidecar
  (`raw_token_bin_v1`), different datatypes.
- Splits: disjoint train/val files (verified by `scripts/validate_data_pair.py`).

## Known limitations & risks
- C4 contains web text with quality issues; this project
  uses it for relative architecture comparison, not for deployable model
  quality claims.
- Right now, we only have single-corpus data, but that’s just until the second-corpus ablation finishes its run ladder.
- Token files are not redistributed in this repository; prepare scripts are
  provided for reproduction.

## Licensing
- C4 is distributed under the terms documented by AllenAI/Google; users must
  obtain it from official sources.

## Tokenizer backends

The reference tokenizer is tiktoken's GPT-2 encoding (`encode_ordinary` +
explicit EOS). An optional fast backend (`--tokenizer-backend gigatoken`,
marcelroed/gigatoken) is supported for large preparations. Sterility contract:
the sidecar's `content_sha256_stream` fingerprint is backend-independent
ground truth - datasets are interchangeable iff fingerprints match. The
gigatoken path is gated twice: verified bit-identical to tiktoken on a
diverse probe set before any tokens are written, and spot-checked against
tiktoken on one random document per 512-document batch during the run
(any mismatch aborts loudly). The backend used is recorded in the sidecar
(`tokenizer_backend`). Tests: `tests/test_data.py::TestTokenizerBackends`
(cross-backend bit-identity + lying-backend abort).
