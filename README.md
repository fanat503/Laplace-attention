<div align="center">

# HLA: Holographic Laplace Attention

Separating finding tokens from transmitting them in attention

</div>

Every attention head does two jobs with one set of vectors: retrieval and transmission. They interfere. HLA gives each job its own channel.

This repository has:

1. The mechanism (one modified attention equation) ([`src/model.py`](src/model.py));
2. A sterile comparison;
3. Metrics ([`docs/METRICS.md`](docs/METRICS.md)).



## How to start

```bash
git clone https://github.com/fanat503/Laplace-attention.git
cd Laplace-attention
pip install -r requirements.txt        # yoou can use CPU

python -m pytest tests/ -q             # 290 passed
python scripts/audit_sterility.py      
```

Train base/HLA pair (TPU/XLA; every step below also runs on CPU for smoke test):

```bash

python scripts/validate_configs.py --base configs/200m_base_s42.json --hla configs/200m_hla_s42.json
python src/make_init.py --shared-backbone \
    --base-config configs/200m_base_s42.json --hla-config configs/200m_hla_s42.json \
    --out-base inits/init_200m_base_s42.pt --out-hla inits/init_200m_hla_s42.pt

python scripts/prepare_c4_data.py --train-tokens 5400000000 --val-tokens 20000000 --out-dir data

python src/train_xla.py --config configs/200m_base_s42.json
python src/train_xla.py --config configs/200m_hla_s42.json

python scripts/make_ablation_configs.py \
    --base configs/200m_base_v2_s42.json --hla configs/200m_hla_v2_s42.json \
    --outdir configs/ablations_200m --seeds 42 43 44
```

Our experimental design: [`docs/EXPERIMENT_CARD.md`](docs/EXPERIMENT_CARD.md).


## Repository 

```
├── src/          Model.py (GPT + mechanisms), train_xla.py (TPU trainer), eval.py (probes)
│                 Make_init.py, data.py, manifest.py, utils.py
├── configs/      20 paired base/HLA JSONs, FLOPs-matched
├── scripts/      Validation, experiment, analysis
├── docs/         Theory, metrics, etc.
└── tests/        290 tests
```

## Citation

If you use this code, please cite it via [`CITATION.cff`](CITATION.cff).

**Apache-2.0**
