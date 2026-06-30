# Configs

## Main pair

- `800m_base_s42.json`
- `800m_hla_s42.json`

These are intentionally matched. The only allowed differences are:

- `variant`
- `run_name`
- `save_dir`
- `init_ckpt`
- `model.phase_mult`
- `model.laplace_alpha`
- `model.baseline_type`

Validate with:

```bash
python scripts/validate_configs.py \
  --base configs/800m_base_s42.json \
  --hla configs/800m_hla_s42.json
```

Current token budget:

```text
batch_size_per_device = 1
num_cores             = 8
block_size            = 1024
grad_accum            = 32
tokens/update         = 262,144
max_steps             = 80,109
total tokens          = 21,000,093,696
```

## Smoke/pilot

- `smoke_hla_s42.json`: small model for TPU/XLA plumbing.
- `pilot_hla_s42.json`: full-size HLA short run.


## Recommended confirmation run

- `700m_base_14b_s42.json`
- `700m_hla_14b_s42.json`

Token budget:

```text
batch_size_per_device = 1
num_cores             = 8
block_size            = 1024
grad_accum            = 32
tokens/update         = 262,144
max_steps             = 53,406
total effective tokens= 14,000,062,464
required stored ids   = 14,013,734,400
params                = 694,046,208
```
