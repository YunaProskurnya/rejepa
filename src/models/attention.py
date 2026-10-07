"""Multi-head self-attention для encoder и predictor I-JEPA."""

import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be positive and divisible by positive num_heads")

        self.num_heads = num_headsw
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """[B,N,D] -> (выход [B,N,D], веса после dropout [B,H,N,N])."""
        B, N, D = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) * self.scale
        weights = scores.softmax(dim=-1)
        weights = self.attn_drop(weights)

        out = weights @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, weights
