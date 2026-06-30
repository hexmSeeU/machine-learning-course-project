
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int
    max_seq_len: int = 128
    n_layers: int = 6
    n_heads: int = 6
    d_model: int = 384
    d_ff: Optional[int] = None
    dropout: float = 0.1
    bias: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("head dimension must be even for RoPE")
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, seq_len, seq_len)


def apply_rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Apply rotary positional embedding to (..., seq_len, head_dim) tensors."""
    *prefix, seq_len, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")

    device = x.device
    dtype = x.dtype
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    angles = torch.outer(positions, inv_freq)
    cos = angles.cos().to(dtype=dtype)
    sin = angles.sin().to(dtype=dtype)

    for _ in prefix:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.empty_like(x)
    rotated[..., 0::2] = x_even * cos - x_odd * sin
    rotated[..., 1::2] = x_even * sin + x_odd * cos
    return rotated


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        h = self.config.n_heads
        hd = self.config.head_dim

        q = self.q_proj(x).view(batch, seq_len, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, h, hd).transpose(1, 2)
        q = apply_rope(q)
        k = apply_rope(k)

        att = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
        att = att + build_causal_mask(seq_len, x.device)
        weights = F.softmax(att, dim=-1)
        weights = self.attn_dropout(weights)
        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.resid_dropout(self.out_proj(y))


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, elementwise_affine=True)
        self.attn = MultiHeadCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, elementwise_affine=True)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class ShakespeareTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_ln = nn.LayerNorm(config.d_model, elementwise_affine=True)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        _, seq_len = idx.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}")
        x = self.token_embedding(idx)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


def theoretical_parameter_count(config: TransformerConfig) -> dict[str, int]:
    d = config.d_model
    f = int(config.d_ff)
    v = config.vocab_size
    bias = 1 if config.bias else 0

    token_embedding = v * d
    attn = 4 * (d * d + bias * d)
    norms = 4 * d
    ffn = d * f + bias * f + f * d + bias * d
    block = attn + norms + ffn
    final_ln = 2 * d
    lm_head = d * v
    total = token_embedding + config.n_layers * block + final_ln + lm_head
    return {
        "token_embedding": token_embedding,
        "attention_per_block": attn,
        "layernorms_per_block": norms,
        "ffn_per_block": ffn,
        "block_total": block,
        "all_blocks": config.n_layers * block,
        "final_layernorm": final_ln,
        "lm_head": lm_head,
        "total": total,
    }


def experimental_parameter_count(model: nn.Module) -> list[dict[str, int]]:
    rows = []
    for name, module in model.named_modules():
        if name == "":
            continue
        own = sum(p.numel() for p in module.parameters(recurse=False))
        if own:
            rows.append({"module": name, "parameters": own})
    rows.append({"module": "total", "parameters": sum(p.numel() for p in model.parameters())})
    return rows


def estimate_flops_by_layer(config: TransformerConfig, seq_len: int, batch_size: int = 1) -> list[dict[str, int]]:
    d = config.d_model
    f = int(config.d_ff)
    h = config.n_heads
    hd = config.head_dim
    b = batch_size
    t = seq_len

    rows: list[dict[str, int]] = [{"layer": "token_embedding_lookup", "flops": 0}]
    for layer_idx in range(1, config.n_layers + 1):
        qkv_out = 3 * 2 * b * t * d * d
        attn_scores = 2 * b * h * t * t * hd
        attn_values = 2 * b * h * t * t * hd
        out_proj = 2 * b * t * d * d
        ffn = 2 * b * t * d * f + 2 * b * t * f * d
        layernorm = 2 * 5 * b * t * d
        rows.append(
            {
                "layer": f"transformer_block_{layer_idx}",
                "flops": qkv_out + attn_scores + attn_values + out_proj + ffn + layernorm,
            }
        )
    rows.append({"layer": "final_layernorm", "flops": 5 * b * t * d})
    rows.append({"layer": "lm_head", "flops": 2 * b * t * d * config.vocab_size})
    rows.append({"layer": "total", "flops": sum(row["flops"] for row in rows)})
    return rows
