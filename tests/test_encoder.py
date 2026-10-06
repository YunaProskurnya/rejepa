"""Проверки информационного и градиентного пути целого энкодера."""

import unittest

import torch

from src.models.encoder import VisionTransformer


class TestEncoder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.model = VisionTransformer(
            image_size=8, patch_size=4, embed_dim=12, depth=2, num_heads=3
        )

    def test_full_and_context_shapes_and_independent_blocks(self):
        images = torch.randn(2, 3, 8, 8)
        self.assertEqual(tuple(self.model(images).shape), (2, 4, 12))
        indices = torch.tensor([[3, 0], [1, 2]])
        self.assertEqual(tuple(self.model(images, indices).shape), (2, 2, 12))
        first = self.model.blocks[0].attn.qkv.weight
        second = self.model.blocks[1].attn.qkv.weight
        self.assertIsNot(first, second)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertFalse(self.model.pos_embed.requires_grad)
        torch.testing.assert_close(self.model.state_dict()["pos_embed"], self.model.pos_embed)

    def test_hidden_pixels_cannot_affect_context(self):
        images = torch.randn(2, 3, 8, 8)
        # Патчи 0 и 1 — верхняя половина изображения.
        indices = torch.tensor([[0, 1], [1, 0]])
        changed = images.clone()
        changed[:, :, 4:, :] = torch.randn_like(changed[:, :, 4:, :]) * 10
        torch.testing.assert_close(
            self.model(images, indices), self.model(changed, indices), rtol=0, atol=0
        )
        self.assertFalse(torch.allclose(self.model(images), self.model(changed)))

    def test_context_backprop_reaches_projection_but_not_hidden_pixels(self):
        images = torch.randn(2, 3, 8, 8, requires_grad=True)
        indices = torch.tensor([[0, 1], [1, 0]])
        out = self.model(images, indices)
        # Сумма после LayerNorm вырожденна; используем разные веса координат.
        (out * torch.randn_like(out)).sum().backward()
        self.assertGreater(self.model.patch_embed.proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(images.grad[:, :, :4, :].abs().sum().item(), 0)
        torch.testing.assert_close(
            images.grad[:, :, 4:, :], torch.zeros_like(images.grad[:, :, 4:, :]),
            rtol=0, atol=0,
        )
        self.assertIsNone(self.model.pos_embed.grad)


if __name__ == "__main__":
    unittest.main()
