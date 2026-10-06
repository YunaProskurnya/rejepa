"""Выбор токенов по готовым индексам патчей без генерации масок."""

import torch


def select_tokens(tokens, indices):
    """[B, N, D] и индексы [B, K] -> выбранные токены [B, K, D]."""

    embed_dim = tokens.shape[-1]
    gather_indices = indices.unsqueeze(-1)
    gather_indices = gather_indices.expand(-1, -1, embed_dim)
    return torch.gather(tokens, dim=1, index=gather_indices)
