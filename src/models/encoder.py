"""ViT-энкодер I-JEPA для квадратного размера изображения.

Пока поддерживается один набор индексов [B,K], без stochastic depth.
"""

import math

import torch
import torch.nn as nn

from .block import Block
from .patch_embed import PatchEmbed
from .positional_embedding import get_2d_sincos_pos_embed
from .token_selection import select_tokens


class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size != 0:
            raise ValueError("image_size must be positive and divisible by positive patch_size")
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.image_size = image_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embed = PatchEmbed(patch_size, in_channels, embed_dim)
        positions = get_2d_sincos_pos_embed(embed_dim, image_size // patch_size)
        self.pos_embed = nn.Parameter(positions.unsqueeze(0), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop, attn_drop=attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.apply(self._init_weights)
        self._rescale_residual_weights()

    @torch.no_grad()
    def _rescale_residual_weights(self):
        # Авторская инициализация I-JEPA уменьшает вклад глубоких residual-ветвей.
        for layer_id, block in enumerate(self.blocks, start=1):
            scale = math.sqrt(2.0 * layer_id)
            block.attn.proj.weight.div_(scale)
            block.mlp.fc2.weight.div_(scale)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images, indices=None):
        """[B,C,S,S] -> [B,N,D] или [B,K,D] при indices [B,K]."""
        return self.get_intermediate_layers(images, n=1, indices=indices)[0]

    def get_intermediate_layers(self, images, n=4, indices=None):
        """Return the last n block outputs, each passed through the final encoder LayerNorm.

        Used by linear evaluation; normal forward still returns only the final block.
        The shared final norm for intermediate blocks is an explicit evaluation choice.
        """
        if not 1 <= n <= len(self.blocks):
            raise ValueError("n must be between 1 and encoder depth")
        if images.ndim != 4 or tuple(images.shape[1:]) != (
            self.in_channels, self.image_size, self.image_size
        ):
            raise ValueError("images must have shape [B, in_channels, image_size, image_size]")
        x = self.patch_embed(images)
        x = x + self.pos_embed
        if indices is not None:
            if indices.ndim != 2 or indices.shape[0] != images.shape[0]:
                raise ValueError("indices must have shape [B,K] with the same batch size as images")
            x = select_tokens(x, indices)
        outputs = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index >= len(self.blocks) - n:
                outputs.append(self.norm(x))
        return outputs
