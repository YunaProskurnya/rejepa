"""Проверки геометрии патчей и участия проекции в обучении.

Запуск из корня проекта: python -m unittest discover -s tests -v
Нужно окружение с установленным PyTorch; GPU не требуется.
"""

import unittest

import torch

from src.models.patch_embed import PatchEmbed


class TestPatchEmbed(unittest.TestCase):
    def test_shapes_for_different_batches_and_patch_sizes(self):
        # B, H, W, размер патча, D, ожидаемое число токенов.
        cases = [(4, 32, 48, 8, 12, 24), (1, 28, 28, 14, 8, 4)]
        for batch, height, width, patch_size, dim, count in cases:
            with self.subTest(batch=batch, patch_size=patch_size):
                model = PatchEmbed(patch_size=patch_size, embed_dim=dim)
                images = torch.zeros(batch, 3, height, width)
                self.assertEqual(tuple(model(images).shape), (batch, count, dim))

    def test_pixel_change_affects_only_its_patch_and_image(self):
        model = PatchEmbed(patch_size=4, embed_dim=5)
        # Известные веса делают результат независимым от случайной инициализации.
        with torch.no_grad():
            model.proj.weight.fill_(1.0)
            model.proj.bias.zero_()

            images = torch.zeros(2, 3, 8, 12)
            original = model(images)
            images[0, 0, 5, 9] = 1.0
            changed = model(images)

        # Сетка 2 x 3: пиксель (5, 9) принадлежит патчу (1, 2), индексу 5.
        expected_difference = torch.zeros_like(original)
        expected_difference[0, 5, :] = 1.0
        torch.testing.assert_close(changed - original, expected_difference)

    def test_projection_weights_receive_gradients(self):
        model = PatchEmbed(patch_size=4, embed_dim=5)
        images = torch.ones(2, 3, 8, 12)
        loss = model(images).sum()
        loss.backward()

        # Каждый вес использован в 2 изображениях x 6 патчах с входом 1.
        self.assertIsNotNone(model.proj.weight.grad)
        self.assertIsNotNone(model.proj.bias.grad)
        torch.testing.assert_close(
            model.proj.weight.grad, torch.full_like(model.proj.weight, 12.0)
        )
        torch.testing.assert_close(
            model.proj.bias.grad, torch.full_like(model.proj.bias, 12.0)
        )


if __name__ == "__main__":
    unittest.main()
