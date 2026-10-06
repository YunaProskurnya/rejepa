"""Проверки координат, порядка признаков и сложения с patch embeddings."""

import math
import unittest

import torch

from src.models.patch_embed import PatchEmbed
from src.models.positional_embedding import get_2d_sincos_pos_embed


class TestPositionalEmbedding(unittest.TestCase):
    def test_known_two_by_two_grid(self):
        # D=8: на каждую координату четыре признака, частоты 1 и 0.01.
        at_zero = [0.0, 0.0, 1.0, 1.0]
        at_one = [math.sin(1.0), math.sin(0.01), math.cos(1.0), math.cos(0.01)]
        expected = torch.tensor([
            at_zero + at_zero,  # (x=0, y=0), индекс 0
            at_one + at_zero,   # (x=1, y=0), индекс 1
            at_zero + at_one,   # (x=0, y=1), индекс 2
            at_one + at_one,    # (x=1, y=1), индекс 3
        ])
        actual = get_2d_sincos_pos_embed(embed_dim=8, grid_size=2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)

    def test_shared_positions_preserve_patch_gradients(self):
        patch_embed = PatchEmbed(patch_size=4, embed_dim=8)
        images = torch.zeros(3, 3, 8, 8)
        tokens = patch_embed(images)
        positions = get_2d_sincos_pos_embed(embed_dim=8, grid_size=2)
        encoded = tokens + positions.unsqueeze(0)

        self.assertEqual(tuple(encoded.shape), (3, 4, 8))
        self.assertEqual(positions.dtype, torch.float32)
        self.assertFalse(positions.requires_grad)
        torch.testing.assert_close(encoded[0] - tokens[0], positions)
        torch.testing.assert_close(encoded[2] - tokens[2], positions)

        encoded.sum().backward()
        # Фиксированные позиции не мешают обучению patch projection.
        torch.testing.assert_close(
            patch_embed.proj.bias.grad, torch.full_like(patch_embed.proj.bias, 12.0)
        )

    def test_invalid_dimensions_are_rejected(self):
        for dim, grid in [(6, 2), (0, 2), (-4, 2), (8, 0), (8, -1)]:
            with self.subTest(dim=dim, grid=grid):
                with self.assertRaises(ValueError):
                    get_2d_sincos_pos_embed(dim, grid)


if __name__ == "__main__":
    unittest.main()
