"""Проверки порядка целевых позиций и градиентного пути predictor."""

import unittest

import torch

from src.models.encoder import VisionTransformer
from src.models.predictor import VisionTransformerPredictor


class TestPredictor(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.predictor = VisionTransformerPredictor(
            num_patches=4, embed_dim=12, predictor_embed_dim=8,
            depth=2, num_heads=2,
        )
        self.context = torch.randn(2, 2, 12)
        self.context_indices = torch.tensor([[0, 2], [1, 3]])
        self.target_indices = torch.tensor([[3, 1], [2, 0]])

    def test_target_order_and_context_order(self):
        out = self.predictor(self.context, self.context_indices, self.target_indices)
        self.assertEqual(tuple(out.shape), (2, 2, 12))
        reordered_targets = self.predictor(
            self.context, self.context_indices, self.target_indices.flip(1)
        )
        torch.testing.assert_close(reordered_targets, out.flip(1), rtol=1e-5, atol=1e-6)
        reordered_context = self.predictor(
            self.context.flip(1), self.context_indices.flip(1), self.target_indices
        )
        torch.testing.assert_close(reordered_context, out, rtol=1e-5, atol=1e-6)

    def test_predictions_depend_on_context(self):
        before = self.predictor(self.context, self.context_indices, self.target_indices)
        changed = self.context.clone()
        changed[0] = torch.randn_like(changed[0])
        after = self.predictor(changed, self.context_indices, self.target_indices)
        self.assertFalse(torch.allclose(before[0], after[0]))
        torch.testing.assert_close(before[1], after[1], rtol=0, atol=0)

    def test_gradients_reach_mask_token_and_encoder(self):
        encoder = VisionTransformer(
            image_size=8, patch_size=4, embed_dim=12, depth=2, num_heads=3
        )
        images = torch.randn(2, 3, 8, 8)
        context = encoder(images, self.context_indices)
        out = self.predictor(context, self.context_indices, self.target_indices)
        # Произвольные цели проверяют autograd; это не обучающий loss I-JEPA.
        loss = (out - torch.randn_like(out)).square().mean()
        loss.backward()
        for parameter in [self.predictor.mask_token,
                          self.predictor.predictor_embed.weight,
                          self.predictor.predictor_proj.weight,
                          encoder.patch_embed.proj.weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertIsNone(self.predictor.predictor_pos_embed.grad)
        self.assertIsNone(encoder.pos_embed.grad)


if __name__ == "__main__":
    unittest.main()
