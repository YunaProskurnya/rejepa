"""Преобразование неперекрывающихся патчей в последовательность векторов."""

import torch.nn as nn


class PatchEmbed(nn.Module):
    """Вход [B, C, H, W], выход [B, N, D].

    H и W должны делиться на patch_size без остатка.
    N = (H / patch_size) * (W / patch_size), D = embed_dim.
    Порядок токенов: слева направо, затем сверху вниз.
    """

    def __init__(self, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

    def forward(self, images):
        tokens = self.proj(images)       # [B, D, H/P, W/P]
        tokens = tokens.flatten(2)       # [B, D, N]
        tokens = tokens.transpose(1, 2)  # [B, N, D]
        return tokens
