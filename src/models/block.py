"""Pre-norm Transformer-блок; stochastic depth пока не поддерживается."""

import torch.nn as nn

from .attention import Attention
from .mlp import MLP


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0,
                 qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = Attention(
            embed_dim, num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MLP(embed_dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        """[B,N,D] -> [B,N,D]."""
        attention_out, _ = self.attn(self.norm1(x))
        x = x + attention_out
        x = x + self.mlp(self.norm2(x))
        return x
