# HLA-v5 Kaggle TPU Smoke

Private Kaggle TPU script kernel for validating the current `pr/hla-v5-repair`
branch end to end.

The kernel:

- clones this branch from the fork
- installs the pinned TPU stack from `HLA-v5/requirements_tpu.txt`
- runs the repo's local verification commands
- generates dummy smoke data and a sterile init checkpoint
- launches `src/train_xla.py` on the 10-step smoke config

Expected output artifacts:

- `hla_v5_kaggle_smoke_result.json`
- `hla_v5_kaggle_smoke_stdout.log`
