"""
Model definition for HLA-v4-safe-scale experiments.

This file intentionally preserves the logic of the provided HLA-v4 model:
  - learned absolute positional embeddings (`wpe`);
  - RMSNorm pre-norm blocks;
  - SwiGLU MLP with hidden_dim = round_up(8*n_embd/3, 64);
  - content-conditioned pairwise phase rotation of Q/K;
  - residual exponential K/V gating when `use_laplace and laplace_alpha != 0`;
  - manual causal attention with a fixed causal mask;
  - tied token embedding / LM head.

The main improvements are engineering/sterility only:
  - valid Python, type hints, clearer errors;
  - HLA identity reset after generic module initialization, so W_gate_k/v remain zero;
  - diagnostics fields for entropy/phase/gate statistics;
  - helper methods used by make_init.py and train_xla.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, T: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_rope:
            return q, k
        pos = torch.arange(T, device=q.device, dtype=self.rope_inv_freq.dtype)
        angles = torch.einsum("t,d->td", pos, self.rope_inv_freq)
        cos = torch.cos(angles)[None, None, :, :].to(q.dtype)
        sin = torch.sin(angles)[None, None, :, :].to(q.dtype)
        return self._rotate_pairwise(q, cos, sin), self._rotate_pairwise(k, cos, sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(rms + self.eps)
        return (self.weight * x_norm).to(input_dtype)


class SwiGLU(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        hidden_dim = int(8 * config.n_embd / 3)
        multiple = int(getattr(config, "ffn_hidden_multiple_of", 64))
        hidden_dim = ((hidden_dim + multiple - 1) // multiple) * multiple  # TPU-friendly
        self.hidden_dim = hidden_dim
        self.fused = bool(getattr(config, "fused_swiglu", True))
        if self.fused:
            # Same math as two independent projections, but one larger matmul.
            self.w_gate_hidden = nn.Linear(config.n_embd, 2 * hidden_dim, bias=False)
        else:
            self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
            self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.w3.NANOGPT_SCALE_INIT = 1
        self.capture_diagnostics: bool = False
        self.last_hidden: Optional[torch.Tensor] = None
        self.last_mlp_out_norm: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            gate_hidden = self.w_gate_hidden(x)
            gate, hidden = torch.split(gate_hidden, self.hidden_dim, dim=-1)
            h = F.silu(gate) * hidden
        else:
            h = F.silu(self.w1(x)) * self.w2(x)
        out = self.w3(h)
        if self.capture_diagnostics:
            with torch.no_grad():
                self.last_hidden = h.detach()
                self.last_mlp_out_norm = out.detach().float().pow(2).mean(dim=-1).sqrt().mean()
        return out


class CausalSelfAttention(nn.Module):
    def __init__(self, config: "GPTConfig", layer_idx: int = 0):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if (config.n_embd // config.n_head) % 2 != 0:
            raise ValueError("head_dim must be even for pairwise rotation")

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.layer_idx = int(layer_idx)
        self.n_layer = int(config.n_layer)
        self.phase_mult = config.phase_mult
        self.use_rope = bool(getattr(config, "use_rope", False))
        self.rope_theta = float(getattr(config, "rope_theta", 10000.0))
        self.attention_backend = getattr(config, "attention_backend", "manual")
        self.layer_dependent_gate = bool(getattr(config, "layer_dependent_gate", False))
        self.layer_gate_multiplier = (
            1.0 + float(self.layer_idx) / float(max(1, self.n_layer))
            if self.layer_dependent_gate
            else 1.0
        )
        self.k_log_clip = float(getattr(config, "k_log_clip", 1.5))
        self.v_log_clip = float(getattr(config, "v_log_clip", 1.0))
        self.use_distance_laplace = bool(getattr(config, "use_distance_laplace", False))
        self.distance_laplace_alpha = float(getattr(config, "distance_laplace_alpha", 0.0))
        self.distance_laplace_range = float(getattr(config, "distance_laplace_range", 1.0))
        self.distance_laplace_clip = float(getattr(config, "distance_laplace_clip", 1.0))

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.W_phase_q = nn.Parameter(torch.zeros(self.n_head, config.n_embd, self.head_dim // 2))
        self.W_phase_k = nn.Parameter(torch.zeros(self.n_head, config.n_embd, self.head_dim // 2))

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
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim // 2).float() / (self.head_dim // 2)))
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

        self.last_entropy: Optional[torch.Tensor] = None
        self.last_angle_q_abs_mean: Optional[torch.Tensor] = None
        self.last_angle_k_abs_mean: Optional[torch.Tensor] = None
        self.last_gate_k_mean: Optional[torch.Tensor] = None
        self.last_gate_v_mean: Optional[torch.Tensor] = None
        self.last_mix_k_mean: Optional[torch.Tensor] = None
        self.last_mix_v_mean: Optional[torch.Tensor] = None
        self.capture_diagnostics: bool = False
        self.capture_attention: bool = False
        self.last_attn: Optional[torch.Tensor] = None
        self.last_gate_k: Optional[torch.Tensor] = None
        self.last_gate_v: Optional[torch.Tensor] = None
        self.last_mix_k: Optional[torch.Tensor] = None
        self.last_mix_v: Optional[torch.Tensor] = None
        self.last_attn_out_norm: Optional[torch.Tensor] = None
        self.last_distance_bias_mean: Optional[torch.Tensor] = None
        self.last_distance_bias_abs_mean: Optional[torch.Tensor] = None

    def reset_hla_identity(self) -> None:
        """Reset HLA-specific parameters to the exact identity state."""
        with torch.no_grad():
            self.W_phase_q.zero_()
            self.W_phase_k.zero_()
            self.W_range_k.zero_()
            self.W_range_v.zero_()
            self.W_gate_k.weight.zero_()
            self.W_gate_v.weight.zero_()

    @staticmethod
    def _rotate_pairwise(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_real, x_imag = x.chunk(2, dim=-1)
        return torch.cat(
            [
                x_real * cos - x_imag * sin,
                x_real * sin + x_imag * cos,
            ],
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        hs = self.head_dim

        if T > self.mask.size(-1):
            raise ValueError(f"block_size {self.mask.size(-1)}")

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, H, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        q, k = self._apply_rope(q, k, T)


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

            with torch.no_grad():
                self.last_angle_q_abs_mean = angles_q.detach().abs().mean()
                self.last_angle_k_abs_mean = angles_k.detach().abs().mean()
        else:
            k_rot = k
            with torch.no_grad():
                zero = torch.zeros((), device=x.device)
                self.last_angle_q_abs_mean = zero
                self.last_angle_k_abs_mean = zero

        k = k_rot
        gate_k_for_distance: Optional[torch.Tensor] = None


        if self.use_laplace and self.laplace_alpha != 0.0:
            
            gate_k = torch.tanh(self.W_gate_k(x)).float()  # (B, T, H)
            gate_k_for_distance = gate_k
            range_k = (self.laplace_range_k * self.layer_gate_multiplier) * (
                1.0 + 0.25 * torch.tanh(self.W_range_k.float())
            )  # (H,)
            range_k = range_k.view(1, 1, self.n_head)

            log_scale_k = torch.clamp(
                self.laplace_alpha * gate_k * range_k,
                min=-(self.k_log_clip * self.layer_gate_multiplier),
                max=(self.k_log_clip * self.layer_gate_multiplier),
            )
            scale_k = torch.exp(log_scale_k)
            mix_k = (1.0 - self.beta_k) + self.beta_k * scale_k  # (B, T, H)
            mix_k = mix_k.to(k.dtype).transpose(1, 2).unsqueeze(-1)  # (B, H, T, 1)
            k = k_rot * mix_k

            gate_v = torch.tanh(self.W_gate_v(x)).float()  # (B, T, H)
            range_v = (self.laplace_range_v * self.layer_gate_multiplier) * (
                1.0 + 0.25 * torch.tanh(self.W_range_v.float())
            )  # (H,)
            range_v = range_v.view(1, 1, self.n_head)

            log_scale_v = torch.clamp(
                self.laplace_alpha * gate_v * range_v,
                min=-(self.v_log_clip * self.layer_gate_multiplier),
                max=(self.v_log_clip * self.layer_gate_multiplier),
            )
            scale_v = torch.exp(log_scale_v)
            mix_v = (1.0 - self.beta_v) + self.beta_v * scale_v  # (B, T, H)
            mix_v = mix_v.to(v.dtype).transpose(1, 2).unsqueeze(-1)  # (B, H, T, 1)
            v = v * mix_v

            with torch.no_grad():
                self.last_gate_k_mean = gate_k.detach().mean()
                self.last_gate_v_mean = gate_v.detach().mean()
                self.last_mix_k_mean = mix_k.detach().float().mean()
                self.last_mix_v_mean = mix_v.detach().float().mean()
                if self.capture_diagnostics:
                    self.last_gate_k = gate_k.detach()
                    self.last_gate_v = gate_v.detach()
                    self.last_mix_k = mix_k.detach()
                    self.last_mix_v = mix_v.detach()
        else:
            with torch.no_grad():
                zero = torch.zeros((), device=x.device)
                one = torch.ones((), device=x.device)
                self.last_gate_k_mean = zero
                self.last_gate_v_mean = zero
                self.last_mix_k_mean = one
                self.last_mix_v_mean = one
                if self.capture_diagnostics:
                    self.last_gate_k = None
                    self.last_gate_v = None
                    self.last_mix_k = None
                    self.last_mix_v = None

        # =====================================================
        # 3) Attention
        # =====================================================
        if self.attention_backend == "sdpa" and not self.capture_diagnostics and not self.capture_attention:

            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.c_proj(y)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))


        if self.use_distance_laplace and self.distance_laplace_alpha != 0.0:
            if gate_k_for_distance is None:
                gate_k_for_distance = torch.tanh(self.W_gate_k(x)).float()
            pos = torch.arange(T, device=x.device)
            dist = (pos[:, None] - pos[None, :]).clamp_min(0).float() / float(max(1, T - 1))
            key_gate = gate_k_for_distance.transpose(1, 2).unsqueeze(2)  # (B,H,1,T_key)
            dist_bias = (
                self.distance_laplace_alpha
                * self.distance_laplace_range
                * self.layer_gate_multiplier
                * dist[None, None, :, :]
                * key_gate
            )
            dist_bias = torch.clamp(
                dist_bias,
                min=-(self.distance_laplace_clip * self.layer_gate_multiplier),
                max=(self.distance_laplace_clip * self.layer_gate_multiplier),
            )
            att = att + dist_bias.to(att.dtype)
            with torch.no_grad():
                self.last_distance_bias_mean = dist_bias.detach().float().mean()
                self.last_distance_bias_abs_mean = dist_bias.detach().float().abs().mean()
        elif self.capture_diagnostics:
            with torch.no_grad():
                zero = torch.zeros((), device=x.device)
                self.last_distance_bias_mean = zero
                self.last_distance_bias_abs_mean = zero

        mask = self.mask[:, :, :T, :T]
        att = att.masked_fill(~mask, float("-inf"))

        att = att.float()
        att = att - att.amax(dim=-1, keepdim=True)
        att = F.softmax(att, dim=-1).to(q.dtype)

        with torch.no_grad():
            p = att.detach().float().clamp(min=1e-9)
            entropy = -(p * p.log()).sum(dim=-1).mean(dim=(0, 2))  # (H,)
            self.last_entropy = entropy.detach()
            if self.capture_attention:
                self.last_attn = att.detach().float()
            elif self.capture_diagnostics:
                self.last_attn = None

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(y)
        if self.capture_diagnostics:
            with torch.no_grad():
                self.last_attn_out_norm = out.detach().float().pow(2).mean(dim=-1).sqrt().mean()
        return out


class Block(nn.Module):
    def __init__(self, config: "GPTConfig", layer_idx: int):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, layer_idx=layer_idx)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.ln_1(x))
        x = x + attn_out
        mlp_out = self.mlp(self.ln_2(x))
        x = x + mlp_out
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257  # true tokenizer vocabulary size
    padded_vocab_size: int | None = None  # optional embedding/output dim rounded for TPU MXU
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1536
    dropout: float = 0.0  # kept for config compatibility; original model does not use dropout
    bias: bool = False    # kept for config compatibility; original model uses bias=False

    phase_mult: float = 0.0
    use_rope: bool = False
    rope_theta: float = 10000.0

    use_laplace: bool = True
    laplace_alpha: float = 0.0
    laplace_range_k: float = 0.35
    laplace_range_v: float = 0.20
    beta_k: float = 0.50
    beta_v: float = 0.25
    k_log_clip: float = 1.5
    v_log_clip: float = 1.0
    layer_dependent_gate: bool = False

    # Optional true distance-aware Laplace log-bias. This is parameter-free and
    # identity-initialized because it is proportional to gate activations, which
    # are exactly zero at init. Negative gate values decay distant tokens; positive
    # gate values preserve/boost distant important tokens.
    use_distance_laplace: bool = False
    distance_laplace_alpha: float = 0.0
    distance_laplace_range: float = 1.0
    distance_laplace_clip: float = 1.0

    gradient_checkpointing: bool = True
    fused_swiglu: bool = True
    ffn_hidden_multiple_of: int = 64
    attention_backend: str = "manual"  # "manual" or optional "sdpa" speed-probe backend

    baseline_type: str = "parameter_matched_ablated"

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.padded_vocab_size is not None and self.padded_vocab_size < self.vocab_size:
            raise ValueError("padded_vocab_size must be >= vocab_size")
        if self.n_layer <= 0 or self.n_head <= 0 or self.n_embd <= 0:
            raise ValueError("n_layer, n_head and n_embd must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("head_dim must be even for pairwise rotation")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.beta_k < 0 or self.beta_v < 0:
            raise ValueError("beta_k and beta_v must be non-negative")
        if self.k_log_clip <= 0 or self.v_log_clip <= 0:
            raise ValueError("k_log_clip and v_log_clip must be positive")
        if self.distance_laplace_range < 0:
            raise ValueError("distance_laplace_range must be non-negative")
        if self.distance_laplace_clip <= 0:
            raise ValueError("distance_laplace_clip must be positive")
        if self.ffn_hidden_multiple_of <= 0:
            raise ValueError("ffn_hidden_multiple_of must be positive")
        if self.attention_backend not in {"manual", "sdpa"}:
            raise ValueError("attention_backend must be 'manual' or 'sdpa'")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.padded_vocab_size = int(config.padded_vocab_size or config.vocab_size)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(self.padded_vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, self.padded_vocab_size, bias=False)

        self.apply(self._init_weights)

        # Critical: generic _init_weights reinitializes W_gate_k/v because they are
        # Linear modules. Reset HLA branches *after* apply() to preserve exact
        # identity initialization intended by the architecture.
        self.reset_hla_identity()

        # tie weights AFTER init, so shared weight is not initialized twice
        self.lm_head.weight = self.transformer.wte.weight

        self.gradient_checkpointing = config.gradient_checkpointing

    def reset_hla_identity(self) -> None:
        for block in self.transformer.h:
            block.attn.reset_hla_identity()

    def _init_weights(self, module: nn.Module) -> None:
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

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.config.block_size}")

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok = self.transformer.wte(idx)  # (B, T, C)
        pos_emb = self.transformer.wpe(pos)[None, :, :]  # (1, T, C)
        x = tok + pos_emb

        for block in self.transformer.h:
            if self.gradient_checkpointing and self.training:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_for_loss = logits.float()
            if self.padded_vocab_size != self.config.vocab_size:
                # Exclude padding-only output classes from the likelihood. This keeps
                # the objective identical to vocab_size while allowing MXU-friendly
                # padded embedding/output matrices.
                logits_for_loss = logits_for_loss.clone()
                logits_for_loss[..., self.config.vocab_size :] = torch.finfo(logits_for_loss.dtype).min
            loss = F.cross_entropy(
                logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss

    def set_diagnostics(self, *, enabled: bool, capture_attention: bool = False) -> None:
        for block in self.transformer.h:
            block.attn.capture_diagnostics = bool(enabled)
            block.attn.capture_attention = bool(enabled and capture_attention)
            block.mlp.capture_diagnostics = bool(enabled)

    def hla_identity_error(self) -> float:
        """Max absolute value of HLA identity-initialized tensors."""
        err = 0.0
        for block in self.transformer.h:
            attn = block.attn
            for tensor in [
                attn.W_phase_q,
                attn.W_phase_k,
                attn.W_range_k,
                attn.W_range_v,
                attn.W_gate_k.weight,
                attn.W_gate_v.weight,
            ]:
                err = max(err, float(tensor.detach().float().abs().max().cpu().item()))
        return err

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["RMSNorm", "SwiGLU", "CausalSelfAttention", "Block", "GPTConfig", "GPT"]
