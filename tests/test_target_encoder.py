"""Проверки независимости target encoder и обновления EMA."""

import unittest

import torch

from src.models.encoder import VisionTransformer
from src.models.target_encoder import create_target_encoder, prepare_targets, update_target_encoder


class TestTargetEncoder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.encoder = VisionTransformer(
            image_size=8, patch_size=4, embed_dim=12, depth=2, num_heads=3
        )
        self.target = create_target_encoder(self.encoder)

    def test_copy_is_equal_independent_and_frozen(self):
        source_params = dict(self.encoder.named_parameters())
        for name, target in self.target.named_parameters():
            source = source_params[name]
            torch.testing.assert_close(target, source, rtol=0, atol=0)
            self.assertNotEqual(target.data_ptr(), source.data_ptr())
            self.assertFalse(target.requires_grad)
        self.assertTrue(self.encoder.patch_embed.proj.weight.requires_grad)
        images = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            torch.testing.assert_close(self.target(images), self.encoder(images), rtol=0, atol=0)

    def test_optimizer_only_changes_source_then_ema_changes_target(self):
        old_source = self.encoder.patch_embed.proj.weight.detach().clone()
        old_target = {name: p.detach().clone() for name, p in self.target.named_parameters()}
        images = torch.randn(2, 3, 8, 8, requires_grad=True)
        with torch.no_grad():
            teacher_output = self.target(images)
        self.assertFalse(teacher_output.requires_grad)
        prediction = self.encoder(images, torch.tensor([[0, 1], [2, 3]]))
        optimizer = torch.optim.SGD(self.encoder.parameters(), lr=0.01)
        (prediction * torch.randn_like(prediction)).sum().backward()
        optimizer.step()
        self.assertFalse(torch.equal(old_source, self.encoder.patch_embed.proj.weight))
        for name, p in self.target.named_parameters():
            self.assertIsNone(p.grad)
            torch.testing.assert_close(p, old_target[name], rtol=0, atol=0)

        source_after_step = {name: p.detach().clone() for name, p in self.encoder.named_parameters()}
        update_target_encoder(self.encoder, self.target, momentum=0.9)
        for name, p in self.target.named_parameters():
            source = dict(self.encoder.named_parameters())[name]
            expected = 0.9 * old_target[name] + 0.1 * source_after_step[name] if source.requires_grad else old_target[name]
            torch.testing.assert_close(p, expected)
            self.assertFalse(p.requires_grad)
            self.assertIsNone(p.grad_fn)
            torch.testing.assert_close(source, source_after_step[name], rtol=0, atol=0)

    def test_known_ema_values_and_endpoints(self):
        source = self.encoder.patch_embed.proj.weight
        target = self.target.patch_embed.proj.weight
        original_positions = self.target.pos_embed.clone()
        with torch.no_grad():
            source.fill_(10.0)
            target.fill_(2.0)
        update_target_encoder(self.encoder, self.target, momentum=0.75)
        torch.testing.assert_close(target, torch.full_like(target, 4.0), rtol=0, atol=0)
        update_target_encoder(self.encoder, self.target, momentum=0.75)
        torch.testing.assert_close(target, torch.full_like(target, 5.5), rtol=0, atol=0)
        update_target_encoder(self.encoder, self.target, momentum=1.0)
        torch.testing.assert_close(target, torch.full_like(target, 5.5), rtol=0, atol=0)
        update_target_encoder(self.encoder, self.target, momentum=0.0)
        torch.testing.assert_close(target, source, rtol=0, atol=0)
        torch.testing.assert_close(self.target.pos_embed, original_positions, rtol=0, atol=0)

    def test_invalid_momentum_and_alias_are_rejected(self):
        for momentum in [-0.1, 1.1, float("nan")]:
            with self.assertRaises(ValueError):
                update_target_encoder(self.encoder, self.target, momentum)
        with self.assertRaises(ValueError):
            update_target_encoder(self.encoder, self.encoder, 0.9)

    def test_targets_are_normalized_in_requested_order_without_graph(self):
        images = torch.randn(2, 3, 8, 8, requires_grad=True)
        indices = torch.tensor([[3, 1], [0, 2]])
        with torch.no_grad():
            # Изменяем affine LayerNorm, чтобы повторная нормализация была существенна.
            self.target.norm.weight.copy_(torch.linspace(0.5, 2.0, 12))
            self.target.norm.bias.copy_(torch.linspace(-1.0, 1.0, 12))
            full = self.target(images)
            chosen = torch.stack([full[0, [3, 1]], full[1, [0, 2]]])
            mean = chosen.mean(dim=-1, keepdim=True)
            variance = ((chosen - mean) ** 2).mean(dim=-1, keepdim=True)
            expected = (chosen - mean) / torch.sqrt(variance + 1e-5)

        targets = prepare_targets(self.target, images, indices)
        torch.testing.assert_close(targets, expected)
        self.assertEqual(tuple(targets.shape), (2, 2, 12))
        self.assertFalse(targets.requires_grad)
        self.assertIsNone(targets.grad_fn)


if __name__ == "__main__":
    unittest.main()
