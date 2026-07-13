# Configs

Paired base/HLA configs at 200m, 300m (FLOPs-matched), 600m, 700m
(+ batch-shape ablations), 800m; `pilot`/`smoke` for infrastructure checks;
`*_v2` = aggressive gate envelope + salience recipe.

## Allowed base<->HLA differences

The single source of truth is `ALLOWED_MAIN_DIFFS` in
`scripts/validate_configs.py`. Currently:

- bookkeeping: `variant`, `run_name`, `save_dir`, `init_ckpt`, `_doc`,
  `model.baseline_type`
- mechanism alphas: `model.phase_mult`, `model.laplace_alpha`,
  `model.distance_laplace_alpha`, `model.salience_alpha`, `model.forget_alpha`, `model.qtemp_alpha`
- identity-at-init structural toggles (bit-identical to base when their
  learned params are zero - verified by tests): `model.layer_dependent_gate`,
  `model.learnable_layer_temp`, `model.per_head_phase`,
  `model.layer_dependent_phase`
- `max_steps` ONLY with `--flops-matched` (HLA <= base; e.g. the 300m pair)

## Validate

```bash
python scripts/validate_configs.py \
  --base configs/200m_base_s42.json --hla configs/200m_hla_s42.json

# FLOPs-matched pair:
python scripts/validate_configs.py \
  --base configs/300m_base_s42.json --hla configs/300m_hla_s42.json --flops-matched

# numeric sanity for every config:
python scripts/audit_config_values.py --all
```

If this file ever disagrees with `validate_configs.py`, the script wins -
and please fix this file in the same commit.
