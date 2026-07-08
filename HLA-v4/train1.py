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



import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import csv
import json
import math
import random
import time
import gc
import shutil
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator, DistributedDataParallelKwargs

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# =========================================================
# HELPERS
# =========================================================

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def free_disk_gb(path: str) -> float:
    p = path if os.path.isdir(path) else os.path.dirname(path)
    if p == "":
        p = "."
    return shutil.disk_usage(p).free / (1024 ** 3)


def save_model_weights(model, path, dtype=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    state = model.state_dict()
    cpu_state = {}
    for k, v in state.items():
        t = v.detach().cpu().contiguous()
        if dtype is not None and torch.is_floating_point(t):
            t = t.to(dtype)
        cpu_state[k] = t

    torch.save(cpu_state, path)


def save_checkpoint(model, optimizer, step, path, dtype=torch.bfloat16, best_val=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    model_state = {}
    for k, v in model.state_dict().items():
        t = v.detach().cpu().contiguous()
        if torch.is_floating_point(t):
            t = t.to(dtype)
        model_state[k] = t

    checkpoint = {
        "model": model_state,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": int(step),
    }

    if best_val is not None:
        checkpoint["best_val"] = float(best_val)

    torch.save(checkpoint, path)
    
def fmt(x, spec=".6f"):
    return format(x, spec) if math.isfinite(x) else "nan"

# =========================================================
# ARCHITECTURE
# =========================================================

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        input_dtype = x.dtype
        x_float = x.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(rms + self.eps)
        return (self.weight * x_norm).to(input_dtype)

class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(8 * config.n_embd / 3)
        hidden_dim = ((hidden_dim + 63) // 64) * 64  # tensor-core friendly
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.w3.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0
        assert (config.n_embd // config.n_head) % 2 == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.phase_mult = config.phase_mult

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.W_phase_q = nn.Parameter(
            torch.zeros(self.n_head, config.n_embd, self.head_dim // 2)
        )
        self.W_phase_k = nn.Parameter(
            torch.zeros(self.n_head, config.n_embd, self.head_dim // 2)
        )

        self.use_laplace = getattr(config, "use_laplace", False)
        self.laplace_alpha = getattr(config, "laplace_alpha", 1.0)
        self.laplace_range_k = getattr(config, "laplace_range_k", 0.35)
        self.laplace_range_v = getattr(config, "laplace_range_v", 0.25)
        self.beta_k = getattr(config, "beta_k", 0.50)
        self.beta_v = getattr(config, "beta_v", 0.30)

        self.W_range_k = nn.Parameter(torch.zeros(self.n_head))
        self.W_range_v = nn.Parameter(torch.zeros(self.n_head))

        self.W_gate_k = nn.Linear(config.n_embd, self.n_head, bias=False)
        self.W_gate_v = nn.Linear(config.n_embd, self.n_head, bias=False)

        nn.init.constant_(self.W_gate_k.weight, 0.0)
        nn.init.constant_(self.W_gate_v.weight, 0.0)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
            .view(1, 1, config.block_size, config.block_size),
            persistent=False
        )

        self.last_entropy = None

    @staticmethod
    def _rotate_pairwise(x, cos, sin):
        x_real, x_imag = x.chunk(2, dim=-1)
        return torch.cat([
            x_real * cos - x_imag * sin,
            x_real * sin + x_imag * cos,
        ], dim=-1)

    def forward(self, x):
        B, T, C = x.size()
        hs = self.head_dim

        assert T <= self.mask.size(-1), (
            f"Sequence length {T} exceeds block_size {self.mask.size(-1)}"
        )

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, H, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        # =====================================================
        # 1) Phase rotation
        # =====================================================
        if self.phase_mult != 0.0:
            x_float = x.float()
            raw_angles_q = torch.einsum("btd,hdk->bhtk", x_float, self.W_phase_q.float())
            raw_angles_k = torch.einsum("btd,hdk->bhtk", x_float, self.W_phase_k.float())

            angles_q = math.pi * self.phase_mult * torch.tanh(raw_angles_q)
            angles_k = math.pi * self.phase_mult * torch.tanh(raw_angles_k)

            cos_q = torch.cos(angles_q).to(q.dtype)
            sin_q = torch.sin(angles_q).to(q.dtype)
            cos_k = torch.cos(angles_k).to(k.dtype)
            sin_k = torch.sin(angles_k).to(k.dtype)

            q = self._rotate_pairwise(q, cos_q, sin_q)
            k_rot = self._rotate_pairwise(k, cos_k, sin_k)
        else:
            k_rot = k

        k = k_rot

        # =====================================================
        # 2) Residual Laplace v4: K + V gating
        # =====================================================
        if self.use_laplace and self.laplace_alpha != 0.0:
            # ---- K gating ----
            gate_k = torch.tanh(self.W_gate_k(x)).float()       # (B, T, H)
            range_k = self.laplace_range_k * (
                1.0 + 0.25 * torch.tanh(self.W_range_k.float())
            )                                                     # (H,)
            range_k = range_k.view(1, 1, self.n_head)

            log_scale_k = torch.clamp(
                self.laplace_alpha * gate_k * range_k,
                min=-1.5, max=1.5
            )
            scale_k = torch.exp(log_scale_k)
            mix_k = (1.0 - self.beta_k) + self.beta_k * scale_k  # (B, T, H)
            mix_k = mix_k.to(k.dtype).transpose(1, 2).unsqueeze(-1)  # (B, H, T, 1)
            k = k_rot * mix_k

            # ---- V gating ----
            gate_v = torch.tanh(self.W_gate_v(x)).float()       # (B, T, H)
            range_v = self.laplace_range_v * (
                1.0 + 0.25 * torch.tanh(self.W_range_v.float())
            )                                                     # (H,)
            range_v = range_v.view(1, 1, self.n_head)

            log_scale_v = torch.clamp(
                self.laplace_alpha * gate_v * range_v,
                min=-1.0, max=1.0
            )
            scale_v = torch.exp(log_scale_v)
            mix_v = (1.0 - self.beta_v) + self.beta_v * scale_v  # (B, T, H)
            mix_v = mix_v.to(v.dtype).transpose(1, 2).unsqueeze(-1)  # (B, H, T, 1)
            v = v * mix_v

        # =====================================================
        # 3) Attention
        # =====================================================
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        mask = self.mask[:, :, :T, :T]
        att = att.masked_fill(~mask, float("-inf"))

        att = att.float()
        att = att - att.amax(dim=-1, keepdim=True)
        att = F.softmax(att, dim=-1).to(q.dtype)

        with torch.no_grad():
            p = att.detach().float().clamp(min=1e-9)
            entropy = -(p * p.log()).sum(dim=-1).mean(dim=(0, 2))  # (H,)
            self.last_entropy = entropy.detach()

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 10
    n_embd: int = 640

    phase_mult: float = 0.15

    # Laplace
    use_laplace: bool = False
    laplace_alpha: float = 1.0
    laplace_range_k: float = 0.35
    laplace_range_v: float = 0.25

    # Residual Laplace v4
    beta_k: float = 0.50
    beta_v: float = 0.30

    gradient_checkpointing: bool = True


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        # CRITICAL FIX (backported from HLA-v5): apply(_init_weights) above
        # re-initializes W_gate_k/W_gate_v (nn.Linear) with noise, silently
        # breaking identity-at-init. Reset all HLA params AFTER generic init.
        self.reset_hla_identity()

        # tie weights AFTER init, so shared weight is not initialized twice
        self.lm_head.weight = self.transformer.wte.weight

        self.gradient_checkpointing = config.gradient_checkpointing

    def reset_hla_identity(self):
        with torch.no_grad():
            for block in self.transformer.h:
                a = block.attn
                a.W_phase_q.zero_(); a.W_phase_k.zero_()
                a.W_range_k.zero_(); a.W_range_v.zero_()
                a.W_gate_k.weight.zero_(); a.W_gate_v.weight.zero_()

    def hla_identity_error(self):
        err = 0.0
        for block in self.transformer.h:
            a = block.attn
            for t in [a.W_phase_q, a.W_phase_k, a.W_range_k, a.W_range_v,
                      a.W_gate_k.weight, a.W_gate_v.weight]:
                err = max(err, float(t.detach().abs().max()))
        return err

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok = self.transformer.wte(idx)                  # (B, T, C)
        pos_emb = self.transformer.wpe(pos)[None, :, :] # (1, T, C)
        x = tok + pos_emb

        for block in self.transformer.h:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100
            )

        return logits, loss


# =========================================================
# DATA
# =========================================================

class FixedDataset(Dataset):
    """
    Deterministic, non-shuffled dataset for matched-baseline experiments.

    Design decisions (NeurIPS reproducibility):
    - No randomness: idx -> token slice is a pure function.
    - mmap=True: avoids loading entire dataset into RAM.
    - weights_only=True: safe deserialization.
    - Explicit dtype check: prevents silent float->long cast bugs.
    - Drop incomplete final sequence (via __len__).

    The file at `path` must contain a single 1-D torch.Tensor
    of integer token ids (int16 / int32 / int64).
    """

    VALID_DTYPES = (torch.int16, torch.int32, torch.int64)

    def __init__(self, path: str, seq_len: int):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        file_size_mb = os.path.getsize(path) / (1024 ** 2)

        self.tokens = torch.load(path, mmap=True, weights_only=True)
        self.seq_len = seq_len
        self.block = seq_len + 1  # input + target

        # --- Validate tensor ---
        if not isinstance(self.tokens, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor in {path}, "
                f"got {type(self.tokens).__name__}"
            )

        if self.tokens.ndim != 1:
            raise ValueError(
                f"Expected 1-D tensor in {path}, "
                f"got {self.tokens.ndim}-D with shape {self.tokens.shape}"
            )

        if self.tokens.dtype not in self.VALID_DTYPES:
            raise TypeError(
                f"Expected integer tensor ({self.VALID_DTYPES}), "
                f"got dtype={self.tokens.dtype} in {path}"
            )

        if len(self.tokens) < self.block:
            raise ValueError(
                f"Dataset {path} has {len(self.tokens):,} tokens, "
                f"need at least {self.block:,} for seq_len={seq_len}"
            )

        # --- Stats ---
        self.n_sequences = len(self.tokens) // self.block
        self.n_tokens_used = self.n_sequences * self.block
        self.n_tokens_dropped = len(self.tokens) - self.n_tokens_used

        # Optional: value range check (catches corrupted files)
        token_min = self.tokens[:1000].min().item()
        token_max = self.tokens[:1000].max().item()

        if token_min < 0:
            raise ValueError(
                f"Negative token ids in {path}: min={token_min}. "
                f"Likely corrupted or wrong format."
            )

        # --- Log ---
        self._path = path
        self._file_size_mb = file_size_mb
        self._token_min = token_min
        self._token_max = token_max

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.block
        chunk = self.tokens.narrow(0, start, self.block).to(torch.long)
        return {"input_ids": chunk}

    def summary(self) -> str:
        return (
            f"FixedDataset(\n"
            f"  path          = {self._path}\n"
            f"  file_size     = {self._file_size_mb:.1f} MB\n"
            f"  total_tokens  = {len(self.tokens):,}\n"
            f"  seq_len       = {self.seq_len}\n"
            f"  block_size    = {self.block} (seq_len + 1)\n"
            f"  n_sequences   = {self.n_sequences:,}\n"
            f"  tokens_used   = {self.n_tokens_used:,}\n"
            f"  tokens_dropped= {self.n_tokens_dropped:,}\n"
            f"  dtype         = {self.tokens.dtype}\n"
            f"  token_range   = [{self._token_min}, {self._token_max}] (first 1000)\n"
            f")"
        )


def _worker_init_fn(worker_id: int):
    """Ensure reproducibility if num_workers > 0 in the future."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(
    path: str,
    seq_len: int,
    batch_size: int,
    drop_last: bool,
    seed: int = 0,
    print_summary: bool = False,
) -> DataLoader:
    ds = FixedDataset(path, seq_len)

    if print_summary:
        print(ds.summary())

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,            # critical for matched data order
        pin_memory=True,
        drop_last=drop_last,
        num_workers=0,            # deterministic; change requires worker_init_fn
        worker_init_fn=_worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
    )

# =========================================================
# PROBES / EVAL
# =========================================================

@torch.no_grad()
def evaluate_induction(model, device, seed=42, batch_size=32):
    was_training = model.training
    model.eval()

    try:
        T = min(256, model.config.block_size)
        assert model.config.vocab_size > 46000 + batch_size, (
            "vocab_size too small for synthetic induction token ids"
        )
        assert T >= 8, "block_size too small for induction eval"

        pos_a1 = T // 3
        pos_b1 = pos_a1 + 1
        pos_a2 = (2 * T) // 3
        assert pos_a2 < T - 1, "Need room so logits at pos_a2 predict next token"

        g = torch.Generator()
        g.manual_seed(seed)

        tokens = torch.randint(
            low=100,
            high=min(20000, model.config.vocab_size - 1),
            size=(batch_size, T),
            generator=g
        ).to(device)

        target_ids = []
        for i in range(batch_size):
            tok_A = 45000 + i
            tok_B = 46000 + i
            tokens[i, pos_a1] = tok_A
            tokens[i, pos_b1] = tok_B
            tokens[i, pos_a2] = tok_A
            target_ids.append(tok_B)

        target_ids = torch.tensor(target_ids, device=device, dtype=torch.long)

        logits, _ = model(tokens)

        if not torch.isfinite(logits).all():
            return float("nan")

        probs = torch.softmax(logits[:, pos_a2, :].float(), dim=-1)
        score = probs.gather(1, target_ids[:, None]).mean().item()

        return score
    finally:
        model.train(was_training)


@torch.no_grad()
def measure_attention_entropy(model, device, seed=42, batch_size=4):
    was_training = model.training
    model.eval()

    try:
        g = torch.Generator()
        g.manual_seed(seed)

        T = min(256, model.config.block_size)
        tokens = torch.randint(
            0, model.config.vocab_size, (batch_size, T), generator=g
        ).to(device)

        _ = model(tokens)

        entropies = []
        for block in model.transformer.h:
            if block.attn.last_entropy is not None:
                entropies.append(block.attn.last_entropy.float())

        if len(entropies) == 0:
            return float("nan")

        return torch.stack(entropies).mean().item()
    finally:
        model.train(was_training)


@torch.no_grad()
def phase_statistics(model):
    all_norms = []

    for block in model.transformer.h:
        if not hasattr(block.attn, "W_phase_q"):
            return None, float("nan")

        Wq = block.attn.W_phase_q.float()
        Wk = block.attn.W_phase_k.float()

        norm_q = torch.linalg.norm(Wq, dim=(1, 2))
        norm_k = torch.linalg.norm(Wk, dim=(1, 2))
        norms = 0.5 * (norm_q + norm_k)
        all_norms.append(norms)

    if len(all_norms) == 0:
        return None, float("nan")

    stacked = torch.stack(all_norms, dim=0)
    mean_per_head = stacked.mean(dim=0)
    mean_global = stacked.mean()

    return mean_per_head.detach().cpu(), mean_global.item()


def get_autocast_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif device.type == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    else:
        return nullcontext()


@torch.no_grad()
def validation_loss(model, device, val_loader, max_batches=25):
    was_training = model.training
    model.eval()

    total_loss = 0.0
    count = 0

    try:
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break

            x = batch["input_ids"][:, :-1].to(device, non_blocking=True)
            y = batch["input_ids"][:, 1:].to(device, non_blocking=True)

            with get_autocast_context(device):
                _, loss = model(x, y)
                loss = loss.mean()

            if not torch.isfinite(loss):
                continue  # пропускаем битый батч вместо того чтобы портить среднее

            total_loss += loss.item()
            count += 1

        return total_loss / count if count > 0 else float("nan")
    finally:
        model.train(was_training)

# =========================================================
# TRAIN
# =========================================================

def train_worker(config):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    accelerator = Accelerator(
        mixed_precision=config.get("mixed_precision", "bf16"),
        kwargs_handlers=[ddp_kwargs],
    )

    seed_everything(config["seed"])

    save_dir = config["save_dir"]
    run_name = config.get("run_name", "run")

    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "train_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    accelerator.wait_for_everyone()

    model_config = GPTConfig(**config["model"])
    model = GPT(model_config)

    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=config.get("fused_adamw", False),
    )

    # =====================================================
    # CHECKPOINT LOADING: resume > init > random
    # =====================================================
    completed_step = 0
    best_val = float("inf")

    resume_ckpt = config.get("resume_ckpt", None)
    init_ckpt = config.get("init_ckpt", None)
    init_strict = config.get("init_strict", True)

    if resume_ckpt is not None:
        if not os.path.exists(resume_ckpt):
            raise FileNotFoundError(f"resume_ckpt not found: {resume_ckpt}")

        if accelerator.is_main_process:
            print(f"[Resume] Loading full checkpoint: {resume_ckpt}")

        ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)

        if "model" not in ckpt:
            raise KeyError(
                f"resume_ckpt={resume_ckpt} does not contain key 'model'. "
                f"Expected full checkpoint with keys ['model', 'optimizer', 'step']."
            )

        incompat = model.load_state_dict(ckpt["model"], strict=True)
        if len(incompat.missing_keys) != 0 or len(incompat.unexpected_keys) != 0:
            raise RuntimeError(
                f"Strict resume load failed.\n"
                f"Missing keys: {incompat.missing_keys}\n"
                f"Unexpected keys: {incompat.unexpected_keys}"
            )

        if "optimizer" not in ckpt or ckpt["optimizer"] is None:
            raise KeyError(
                f"resume_ckpt={resume_ckpt} does not contain optimizer state."
            )

        optimizer.load_state_dict(ckpt["optimizer"])
        completed_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val", float("inf")))

        if accelerator.is_main_process:
            print(f"[Resume] Loaded successfully from step={completed_step}")

    elif init_ckpt is not None:
        if not os.path.exists(init_ckpt):
            raise FileNotFoundError(f"init_ckpt not found: {init_ckpt}")

        if accelerator.is_main_process:
            print(f"[Init] Loading init checkpoint: {init_ckpt} (strict={init_strict})")

        state = torch.load(init_ckpt, map_location="cpu", weights_only=True)
        incompat = model.load_state_dict(state, strict=init_strict)

        if accelerator.is_main_process:
            print(f"[Init] Missing keys:    {len(incompat.missing_keys)}")
            print(f"[Init] Unexpected keys: {len(incompat.unexpected_keys)}")

            if len(incompat.missing_keys) > 0:
                print(f"[Init] First missing keys: {incompat.missing_keys[:8]}")
            if len(incompat.unexpected_keys) > 0:
                print(f"[Init] First unexpected keys: {incompat.unexpected_keys[:8]}")

        if init_strict and (
            len(incompat.missing_keys) != 0 or len(incompat.unexpected_keys) != 0
        ):
            raise RuntimeError(
                f"Strict init load failed.\n"
                f"Missing keys: {incompat.missing_keys}\n"
                f"Unexpected keys: {incompat.unexpected_keys}"
            )

    else:
        if accelerator.is_main_process:
            print("[Init] No checkpoint specified. Using fresh random initialization.")

    # =====================================================
    # DATA LOADERS
    # =====================================================
    train_loader_raw = get_dataloader(
        path=config["train_path"],
        seq_len=model_config.block_size,
        batch_size=config["batch_size_per_device"],
        drop_last=True,
        seed=config["seed"],
        print_summary=accelerator.is_main_process,
    )

    val_loader = None
    if accelerator.is_main_process:
        val_loader = get_dataloader(
            path=config["val_path"],
            seq_len=model_config.block_size,
            batch_size=config["eval_batch_size_per_device"],
            drop_last=False,
            seed=config["seed"] + 1,
            print_summary=False,
        )

    # =====================================================
    # DATASET / BATCH STATS
    # =====================================================
    n_train_sequences = len(train_loader_raw.dataset)
    n_val_sequences = len(val_loader.dataset) if val_loader is not None else 0

    train_tokens_stored = n_train_sequences * (model_config.block_size + 1)
    val_tokens_stored = n_val_sequences * (model_config.block_size + 1)

    train_tokens_effective = n_train_sequences * model_config.block_size
    val_tokens_effective = n_val_sequences * model_config.block_size

    n_train_batches_raw = len(train_loader_raw)
    n_val_batches_raw = len(val_loader) if val_loader is not None else 0

    if accelerator.is_main_process:
        print(f"Train sequences:           {n_train_sequences:,}")
        print(f"Val sequences:             {n_val_sequences:,}")
        print(f"Train batches (raw):       {n_train_batches_raw:,}")
        print(f"Val batches (raw):         {n_val_batches_raw:,}")
        print(f"Train tokens stored:       {train_tokens_stored:,}")
        print(f"Val tokens stored:         {val_tokens_stored:,}")
        print(f"Train tokens effective:    {train_tokens_effective:,}")
        print(f"Val tokens effective:      {val_tokens_effective:,}")

    # =====================================================
    # TRAINING BUDGET STATS
    # =====================================================
    tokens_per_update = (
        config["batch_size_per_device"]
        * accelerator.num_processes
        * model_config.block_size
        * config["grad_accum"]
    )

    remaining_steps = max(0, config["max_steps"] - completed_step)


    required_global_micro_batches_remaining = (
        remaining_steps
        * config["grad_accum"]
        * accelerator.num_processes
    )

    planned_total_tokens = config["max_steps"] * tokens_per_update
    remaining_tokens = remaining_steps * tokens_per_update

    if accelerator.is_main_process:
        print(f"Completed steps already:   {completed_step:,}")
        print(f"Remaining steps:           {remaining_steps:,}")
        print(f"Tokens/update:             {tokens_per_update:,}")
        print(f"Planned total tokens:      {planned_total_tokens:,}")
        print(f"Remaining tokens:          {remaining_tokens:,}")
        print(f"Required micro-batches:    {required_global_micro_batches_remaining:,} (remaining)")

    if (
        len(train_loader_raw) < required_global_micro_batches_remaining
        and accelerator.is_main_process
    ):
        print(
            f"[WARNING] Dataset has {len(train_loader_raw):,} raw batches, "
            f"but {required_global_micro_batches_remaining:,} are needed "
            f"to finish the remaining training schedule."
        )
        print("[WARNING] Training will naturally stop when dataset ends.")

    # =====================================================
    # PREPARE FOR DISTRIBUTED / MIXED PRECISION
    # =====================================================
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader_raw
    )

    # =====================================================
    # SKIP ALREADY CONSUMED BATCHES IF RESUMING
    # NOTE:
    # after accelerator.prepare(), train_loader is local to the process,
    # so we skip LOCAL micro-batches = completed_step * grad_accum
    # =====================================================
    local_micro_batches_to_skip = completed_step * config["grad_accum"]

    if local_micro_batches_to_skip > 0:
        train_loader = accelerator.skip_first_batches(
            train_loader,
            local_micro_batches_to_skip
        )
        if accelerator.is_main_process:
            print(
                f"[Resume] Skipped {local_micro_batches_to_skip:,} local micro-batches "
                f"to match completed_step={completed_step}"
            )

    def get_lr(step):
        warmup = config["warmup"]
        max_steps = config["max_steps"]
        base_lr = config["lr"]
        min_lr = config["min_lr"]

        if step < warmup:
            return base_lr * (step + 1) / max(1, warmup)

        progress = (step - warmup) / max(1, (max_steps - warmup))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + cosine * (base_lr - min_lr)

    csv_file = None
    writer = None

    if accelerator.is_main_process:
        csv_path = os.path.join(save_dir, f"train_log_{run_name}.csv")
        append_csv = completed_step > 0 and os.path.exists(csv_path)
        csv_file = open(csv_path, "a" if append_csv else "w", newline="")
        writer = csv.writer(csv_file)

        if not append_csv:
            # Метаданные эксперимента — первые строки файла
            writer.writerow(["# experiment", run_name])
            writer.writerow(["# variant", config.get("variant", "unknown")])
            writer.writerow(["# seed", config["seed"]])
            writer.writerow(["# config_hash", config.get("config_hash", "unknown")])
            writer.writerow(["# phase_mult", config["model"].get("phase_mult", 0.0)])
            writer.writerow(["# laplace_alpha", config["model"].get("laplace_alpha", 0.0)])
            writer.writerow(["# baseline_type", config["model"].get("baseline_type", "unknown")])
            writer.writerow(["# tokens_planned", tokens_per_update * config["max_steps"]])
            writer.writerow([])  # пустая строка-разделитель
    
            # Основной header
            writer.writerow([
                "step",
                "tokens_seen",
                "train_loss",
                "val_loss",
                "val_ppl",
                "induction",
                "entropy",
                "phase_norm",
                "grad_norm",
                "steps_per_sec",
                "tokens_per_sec",
                "wall_time_sec"
            ])
            csv_file.flush()
        else:
            writer.writerow([])
            writer.writerow(["# resumed_run", time.strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow(["# resumed_from_step", completed_step])
            csv_file.flush()
            print(f"[Resume] Appending to existing CSV: {csv_path}")

        print(f"Using {accelerator.num_processes} GPU(s)")
        print(f"Mixed precision: {accelerator.mixed_precision}")
        print(f"Parameters: {n_params:,}")
        print(f"Tokens/update: {tokens_per_update:,}")
        print(f"Planned total tokens: {tokens_per_update * config['max_steps']:,}")
        print(f"Save dir: {save_dir}")

    micro_in_update = 0
    running_micro_loss = 0.0
    log_loss_accum = 0.0
    log_steps_accum = 0
    log_grad_norm_accum = 0.0

    train_start_time = time.time()
    last_log_time = train_start_time
    last_log_step = completed_step

    optimizer.zero_grad(set_to_none=True)
    model.train()
    
    resume_every = config.get("resume_every", config["log_every"])

    try:
        for batch in train_loader:
            if completed_step >= config["max_steps"]:
                break

            # LR update for current step
            lr = get_lr(completed_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            x = batch["input_ids"][:, :-1]
            y = batch["input_ids"][:, 1:]

            should_sync = (micro_in_update + 1 == config["grad_accum"])
            sync_context = (
                nullcontext() if should_sync
                else accelerator.no_sync(model)
            )

            with sync_context:
                with accelerator.autocast():
                    _, loss = model(x, y)
                    loss = loss.mean()

                if not torch.isfinite(loss):
                    accelerator.print(
                        f"[ERROR] Non-finite loss at "
                        f"step={completed_step}, micro={micro_in_update}: {loss.item()}"
                    )
                    accelerator.end_training()
                    raise RuntimeError(f"Non-finite loss at step={completed_step}")

                running_micro_loss += loss.detach().float().item()
                accelerator.backward(loss / config["grad_accum"])

            micro_in_update += 1

            if should_sync:
                grad_clip = config.get("grad_clip", 1.0)
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                log_grad_norm_accum += grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                completed_step += 1
                update_loss = running_micro_loss / config["grad_accum"]
                running_micro_loss = 0.0
                micro_in_update = 0

                log_loss_accum += update_loss
                log_steps_accum += 1

                if completed_step % config["log_every"] == 0:
                    accelerator.wait_for_everyone()

                    # ---------------------------------------------
                    # Aggregate window stats BEFORE resetting them
                    # ---------------------------------------------
                    steps_in_window = max(log_steps_accum, 1)
                    smooth_loss = log_loss_accum / steps_in_window
                    smooth_grad_norm = log_grad_norm_accum / steps_in_window

                    now = time.time()
                    elapsed = now - last_log_time
                    steps_since_log = completed_step - last_log_step
                    steps_per_sec = steps_since_log / max(elapsed, 1e-8)
                    tokens_per_sec = steps_per_sec * tokens_per_update
                    wall_time_sec = now - train_start_time

                    last_log_time = now
                    last_log_step = completed_step

                    # Reset accumulators on ALL processes
                    log_loss_accum = 0.0
                    log_grad_norm_accum = 0.0
                    log_steps_accum = 0

                    if accelerator.is_main_process:
                        base_model = accelerator.unwrap_model(model)
                        tokens_seen = completed_step * tokens_per_update

                        val_loss_val = validation_loss(
                            base_model,
                            device=accelerator.device,
                            val_loader=val_loader,
                            max_batches=config["val_batches"]
                        )

                        if math.isfinite(val_loss_val):
                            val_ppl = math.exp(val_loss_val) if val_loss_val < 20 else float("inf")
                        else:
                            val_ppl = float("nan")

                        induction = evaluate_induction(
                            base_model,
                            device=accelerator.device
                        )
                        entropy = measure_attention_entropy(
                            base_model,
                            device=accelerator.device
                        )
                        _, phase_norm = phase_statistics(base_model)

                        # Extra safety: sanitize non-finite metrics
                        if not math.isfinite(induction):
                            induction = float("nan")
                        if not math.isfinite(entropy):
                            entropy = float("nan")
                        if not math.isfinite(phase_norm):
                            phase_norm = float("nan")
                        if not math.isfinite(smooth_grad_norm):
                            smooth_grad_norm = float("nan")
                        if not math.isfinite(smooth_loss):
                            smooth_loss = float("nan")
                        if not math.isfinite(steps_per_sec):
                            steps_per_sec = float("nan")
                        if not math.isfinite(tokens_per_sec):
                            tokens_per_sec = float("nan")

                        writer.writerow([
                            completed_step,
                            tokens_seen,
                            f"{smooth_loss:.6f}",
                            f"{val_loss_val:.6f}" if math.isfinite(val_loss_val) else "nan",
                            f"{val_ppl:.3f}" if math.isfinite(val_ppl) else "nan",
                            f"{induction:.6f}" if math.isfinite(induction) else "nan",
                            f"{entropy:.6f}" if math.isfinite(entropy) else "nan",
                            f"{phase_norm:.6f}" if math.isfinite(phase_norm) else "nan",
                            f"{smooth_grad_norm:.6f}" if math.isfinite(smooth_grad_norm) else "nan",
                            f"{steps_per_sec:.4f}" if math.isfinite(steps_per_sec) else "nan",
                            f"{tokens_per_sec:.2f}" if math.isfinite(tokens_per_sec) else "nan",
                            f"{wall_time_sec:.1f}",
                        ])
                        csv_file.flush()

                        if completed_step % resume_every == 0:
                            save_checkpoint(
                                model=base_model,
                                optimizer=optimizer,
                                step=completed_step,
                                path=os.path.join(save_dir, f"latest_{run_name}_resume.pt"),
                                dtype=torch.bfloat16
                            )

                        print(
                            f"\nStep {completed_step} | "
                            f"tokens_seen={tokens_seen:,} | "
                            f"lr={lr:.2e}\n"
                            f"  Train Loss : {smooth_loss:.6f}\n"
                            f"  Val Loss   : {val_loss_val:.6f}\n"
                            f"  Val PPL    : {val_ppl:.3f}\n"
                            f"  Induction  : {induction:.6f}\n"
                            f"  Entropy    : {entropy:.6f}\n"
                            f"  Phase Norm : {phase_norm:.6f}\n"
                            f"  Grad Norm  : {smooth_grad_norm:.6f}\n"
                            f"  Steps/sec  : {steps_per_sec:.4f}\n"
                            f"  Tokens/sec : {tokens_per_sec:.2f}\n"
                            f"  Wall time  : {wall_time_sec:.1f}s"
                        )

                        # ---------------------------------------------
                        # Save best checkpoint
                        # ---------------------------------------------
                        if math.isfinite(val_loss_val) and val_loss_val < best_val:
                            if free_disk_gb(save_dir) < config["min_free_gb_best"]:
                                raise RuntimeError("Not enough free disk space for best checkpoint")

                            best_val = val_loss_val

                            # Light weights for eval / paper plots / lm-eval
                            save_model_weights(
                                model=base_model,
                                path=os.path.join(save_dir, f"best_val_{run_name}.pt"),
                                dtype=torch.bfloat16
                            )

                            # Full checkpoint for resume
                            save_checkpoint(
                                model=base_model,
                                optimizer=optimizer,
                                step=completed_step,
                                path=os.path.join(save_dir, f"best_val_{run_name}_resume.pt"),
                                dtype=torch.bfloat16
                            )

                            print(f"  [Saved best] val_loss={best_val:.6f}")

                        # ---------------------------------------------
                        # Periodic checkpoint
                        # ---------------------------------------------
                        if completed_step % config["save_every"] == 0:
                            if free_disk_gb(save_dir) >= config["min_free_gb_best"]:
                                save_model_weights(
                                    model=base_model,
                                    path=os.path.join(save_dir, f"step_{completed_step}_{run_name}.pt"),
                                    dtype=torch.bfloat16
                                )

                                save_checkpoint(
                                    model=base_model,
                                    optimizer=optimizer,
                                    step=completed_step,
                                    path=os.path.join(save_dir, f"step_{completed_step}_{run_name}_resume.pt"),
                                    dtype=torch.bfloat16
                                )

                                print(f"  [Saved ckpt] step_{completed_step}_{run_name}.pt")
                            else:
                                print("  [Skip save] not enough free disk space for periodic checkpoint")

                    model.train()
                    accelerator.wait_for_everyone()

    finally:
        if accelerator.is_main_process:
            try:
                base_model = accelerator.unwrap_model(model)

                if free_disk_gb(save_dir) >= config["min_free_gb_final"]:
                    # 1) Финальные fp32 веса модели
                    save_model_weights(
                        model=base_model,
                        path=os.path.join(save_dir, f"final_{run_name}_fp32.pt"),
                        dtype=torch.float32
                    )

                    # 2) Финальный полный resume-checkpoint
                    save_checkpoint(
                        model=base_model,
                        optimizer=optimizer,
                        step=completed_step,
                        path=os.path.join(save_dir, f"final_{run_name}_resume.pt"),
                        dtype=torch.bfloat16
                    )

                    print(f"Saved final fp32 checkpoint: final_{run_name}_fp32.pt")
                else:
                    print("WARNING: not enough disk space to save final fp32 checkpoint")

            except Exception as e:
                print(f"WARNING: failed to save final checkpoint: {e}")

            if csv_file is not None:
                csv_file.close()

if __name__ == "__main__":
    raise RuntimeError("Run this through modal_app.py, not directly.")
