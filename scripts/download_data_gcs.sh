#!/usr/bin/env bash
set -euo pipefail

# Download prepared .bin token files from Google Cloud Storage to the TPU VM disk.
# Usage:
#   GCS_PREFIX=gs://your-bucket/hla-v4/data ./scripts/download_data_gcs.sh
# Optional:
#   DATA_DIR=/mnt/disks/data/hla-v4-data ./scripts/download_data_gcs.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${GCS_PREFIX:?Set GCS_PREFIX=gs://bucket/path containing train/val .bin files}"
DATA_DIR="${DATA_DIR:-data}"
mkdir -p "$DATA_DIR"

echo "[download] from $GCS_PREFIX to $DATA_DIR"

gsutil -m cp "$GCS_PREFIX/train_fixed_tokens.bin" "$DATA_DIR/train_fixed_tokens.bin"
gsutil -m cp "$GCS_PREFIX/train_fixed_tokens.bin.json" "$DATA_DIR/train_fixed_tokens.bin.json"
gsutil -m cp "$GCS_PREFIX/val_fixed_tokens.bin" "$DATA_DIR/val_fixed_tokens.bin"
gsutil -m cp "$GCS_PREFIX/val_fixed_tokens.bin.json" "$DATA_DIR/val_fixed_tokens.bin.json"
if gsutil -q stat "$GCS_PREFIX/prepared_dataset_manifest.json"; then
  gsutil cp "$GCS_PREFIX/prepared_dataset_manifest.json" "$DATA_DIR/prepared_dataset_manifest.json"
fi

echo "[download] done"
ls -lh "$DATA_DIR"
