"""Фиксированные 2D sin/cos-кодировки позиций патчей для I-JEPA."""

import torch


def _sincos_1d(positions, embed_dim):
    """Кодировать список координат: [N] -> [N, embed_dim]."""
    # Первую половину признаков занимают sin, вторую - cos.
    half_dim = embed_dim // 2
    frequency_indices = torch.arange(half_dim, dtype=torch.float64)
    frequencies = 1.0 / (10000.0 ** (frequency_indices / half_dim))

    angles = positions.reshape(-1, 1) * frequencies.reshape(1, -1)
    return torch.cat([angles.sin(), angles.cos()], dim=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Вернуть float32-тензор [grid_size**2, embed_dim] на CPU.

    grid_size — сторона квадратной сетки патчей.
    Порядок признаков: [sin(x), cos(x), sin(y), cos(y)].
    """
    if embed_dim <= 0 or embed_dim % 4 != 0:
        raise ValueError("embed_dim must be positive and divisible by 4")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    coordinates = torch.arange(grid_size, dtype=torch.float64)
    x, y = torch.meshgrid(coordinates, coordinates, indexing="xy")
    x_embedding = _sincos_1d(x.flatten(), embed_dim // 2)
    y_embedding = _sincos_1d(y.flatten(), embed_dim // 2)
    return torch.cat([x_embedding, y_embedding], dim=1).float()
