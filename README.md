# Laplace-attention
HLA-v4 (Holographic Laplace Attention) is a transformer architecture utilizing complex plane geometry to decouple retrieval from content transmission. By integrating complex phase rotation with residual Laplace gating, the model improves expressivity in attention heads. Previous iterations achieved a 0.09 loss reduction on 100M parameter model.

Hla-v4 is an attention modification designed to separate retrieval (Query-Key matching) from transmission (Value mapping). It’s the next iteration after my v3 experiments, which showed a -0.09 loss gap on a 100M model.

How it works
The main idea is to stop positional and semantic signals from interfering with each other in the attention matrix.

Phase rotation (Q and K): Instead of standard embeddings, I use content conditioned rotations. The model generates angles based on the input tokens, bounded by tanh, and applies them to Q and K. This lets the heads rotate into alignment in a complex latent space.
Laplace Gating (K and V): I added a gating path inspired by Laplace distribution priors. It uses a head-specific range modifier to control how much "reach" each head has. It’s implemented as a residual mix:
Mix = (1 - beta) + beta * exp(gate * range)
This means at init (when weights are 0), the model is just a standard Transformer.
I built this repo specifically for sterile experiments on Modal.

Sterile init: I use a separate script (make_init.py) to create a shared starting checkpoint. Both Base and v4 start from the exact same weights, so the 0.09 gap isn't just luck :).
Causal safe: Strict causal masking is applied across all new layers. No future leakage.
Modal: The modal_app.py handles everything: volume uploads, config hashing, and auto-resuming from checkpoints.
Current Status
I'm a 13yo independent researcher. v3 worked well, but I need to see if these scaling laws hold up at 1B+ tokens. I'm currently looking for compute grants to run the full benchmark.

Quick Start
Bash

# 1. Create shared weights locally
python make_init.py

# 2. Sync to Modal
modal run modal_app.py --mode upload --use-init true

# 3. Run a smoke test (Base vs v4)
modal run modal_app.py --mode train --variant base --preset smoke
modal run modal_app.py --mode train --variant v4 --preset smoke
