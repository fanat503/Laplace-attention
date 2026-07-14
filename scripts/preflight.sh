#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_CONFIG="${BASE_CONFIG:-configs/800m_base_s42.json}"
HLA_CONFIG="${HLA_CONFIG:-configs/800m_hla_s42.json}"
BASE_INIT="${BASE_INIT:-data/init_base_s42.pt}"
HLA_INIT="${HLA_INIT:-data/init_hla_s42.pt}"

echo "[preflight] static audit"
python scripts/audit_sterility.py --root .

echo "[preflight] environment check"
python scripts/check_environment.py --requirements requirements.txt --require-xla

echo "[preflight] config pair validation"
python scripts/validate_configs.py --base "$BASE_CONFIG" --hla "$HLA_CONFIG"

echo "[preflight] dataset pair validation"
python scripts/validate_data_pair.py --config "$HLA_CONFIG"

echo "[preflight] dataloader structural check"
python scripts/check_dataloader.py --config "$HLA_CONFIG" --batches 4

echo "[preflight] create shared-backbone init if missing"
if [[ ! -f "$BASE_INIT" || ! -f "$HLA_INIT" ]]; then
  python src/make_init.py \
    --shared-backbone \
    --base-config "$BASE_CONFIG" \
    --hla-config "$HLA_CONFIG" \
    --out-base "$BASE_INIT" \
    --out-hla "$HLA_INIT"
else
  echo "[preflight] init checkpoints already exist"
fi

echo "[preflight] create run manifests"
python scripts/create_run_manifest.py --config "$BASE_CONFIG" --out "runs/preflight_base_manifest.json"
python scripts/create_run_manifest.py --config "$HLA_CONFIG" --out "runs/preflight_hla_manifest.json"

echo "[preflight] runtime verifier"
python scripts/verify_run.py \
  --config "$HLA_CONFIG" \
  --init "$HLA_INIT" \
  --base-config "$BASE_CONFIG" \
  --hla-config "$HLA_CONFIG" \
  --base-init "$BASE_INIT" \
  --hla-init "$HLA_INIT" \
  --skip-dataset \
  --skip-forward-backward

echo "[preflight] OK"
