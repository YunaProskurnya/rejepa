"""I-JEPA predictor для одного набора context/target-индексов на изображение."""

import math

import torch
import torch.nn as nn

from .block import Block
from .positional_embedding import get_2d_sincos_pos_embed
from .token_selection import select_tokens


class VisionTransformerPredictor(nn.Module):
    def __init__(self, num_patches, embed_dim=768, predictor_embed_dim=384,
                 depth=6, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0):
        super().__init__()
        if num_patches <= 0 or math.isqrt(num_patches) ** 2 != num_patches:
            raise ValueError("num_patches must describe a positive square grid")
        if embed_dim <= 0 or depth <= 0:
            raise ValueError("embed_dim and depth must be positive")
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        positions = get_2d_sincos_pos_embed(predictor_embed_dim, math.isqrt(num_patches))
        self.predictor_pos_embed = nn.Parameter(positions.unsqueeze(0), requires_grad=False)
        self.predictor_blocks = nn.ModuleList([
            Block(predictor_embed_dim, num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop)
            for _ in range(depth)
        ])
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim, eps=1e-6)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim)

        nn.init.trunc_normal_(self.mask_token, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.apply(self._init_weights)
        self._rescale_residual_weights()

    @torch.no_grad()
    def _rescale_residual_weights(self):
        # Та же глубинная поправка, что в авторском predictor и context encoder.
        for layer_id, block in enumerate(self.predictor_blocks, start=1):
            scale = math.sqrt(2.0 * layer_id)
            block.attn.proj.weight.div_(scale)
            block.mlp.fc2.weight.div_(scale)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, context, context_indices, target_indices):
        """[B,Nc,D], [B,Nc], [B,Nt] -> предсказания [B,Nt,D]."""
        if isinstance(target_indices, list):
            if not target_indices:
                raise ValueError("target_indices must contain at least one block")
            return [self.forward(context, context_indices, indices) for indices in target_indices]

        if context.ndim != 3 or context.shape[-1] != self.embed_dim:
            raise ValueError("context must have shape [B,Nc,embed_dim]")
        B, Nc, _ = context.shape
        if tuple(context_indices.shape) != (B, Nc):
            raise ValueError("context_indices must match context shape [B,Nc]")
        if target_indices.ndim != 2 or target_indices.shape[0] != B:
            raise ValueError("target_indices must have shape [B,Nt]")
        if B == 0 or Nc == 0 or target_indices.shape[1] == 0:
            raise ValueError("batch, context and target blocks must be nonempty")

        positions = self.predictor_pos_embed.expand(B, -1, -1)
        x = self.predictor_embed(context)
        x = x + select_tokens(positions, context_indices)

        target_positions = select_tokens(positions, target_indices)
        target_tokens = self.mask_token.expand(B, target_indices.shape[1], -1)
        target_tokens = target_tokens + target_positions
        x = torch.cat([x, target_tokens], dim=1)

        for block in self.predictor_blocks:
            x = block(x)
        x = self.predictor_norm(x)
        x = x[:, Nc:, :]
        return self.predictor_proj(x)
