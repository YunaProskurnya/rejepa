"""Сравнение выхода и градиентов с независимой реализацией PyTorch."""

import copy
import unittest

import torch
import torch.nn as nn

from src.models.attention import Attention


class TestAttention(unittest.TestCase):
    def test_matches_pytorch_outputs_and_gradients(self):
        torch.manual_seed(42)
        model = Attention(embed_dim=12, num_heads=3).double()
        reference = nn.MultiheadAttention(12, 3, batch_first=True).double()
        with torch.no_grad():
            reference.in_proj_weight.copy_(model.qkv.weight)
            reference.in_proj_bias.copy_(model.qkv.bias)
            reference.out_proj.weight.copy_(model.proj.weight)
            reference.out_proj.bias.copy_(model.proj.bias)

        x = torch.randn(2, 5, 12, dtype=torch.float64, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_(True)
        actual, weights = model(x)
        expected, reference_weights = reference(
            reference_x, reference_x, reference_x, average_attn_weights=False
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(weights, reference_weights)
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 3, 5).double())

        # Неодинаковый вклад выходных координат в loss проверяет маршрут градиентов.
        upstream = torch.randn_like(actual)
        (actual * upstream).sum().backward()
        (expected * upstream).sum().backward()
        torch.testing.assert_close(x.grad, reference_x.grad)
        for ours, theirs in [
            (model.qkv.weight, reference.in_proj_weight),
            (model.qkv.bias, reference.in_proj_bias),
            (model.proj.weight, reference.out_proj.weight),
            (model.proj.bias, reference.out_proj.bias),
        ]:
            torch.testing.assert_close(ours.grad, theirs.grad)

    def test_sdpa_matches_explicit_attention_outputs_and_gradients(self):
        torch.manual_seed(17)
        explicit = Attention(embed_dim=12, num_heads=3).double()
        sdpa = copy.deepcopy(explicit)
        x = torch.randn(2, 5, 12, dtype=torch.float64, requires_grad=True)
        sdpa_x = x.detach().clone().requires_grad_(True)

        expected, _ = explicit(x)
        actual, weights = sdpa(sdpa_x, need_weights=False)
        self.assertIsNone(weights)
        torch.testing.assert_close(actual, expected)

        upstream = torch.randn_like(expected)
        (expected * upstream).sum().backward()
        (actual * upstream).sum().backward()
        torch.testing.assert_close(sdpa_x.grad, x.grad)
        for actual_parameter, expected_parameter in zip(sdpa.parameters(), explicit.parameters()):
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)

    def test_sdpa_dropout_is_disabled_in_eval(self):
        model = Attention(embed_dim=12, num_heads=3, attn_drop=1.0).double()
        x = torch.randn(2, 5, 12, dtype=torch.float64)
        training_out, _ = model(x, need_weights=False)
        torch.testing.assert_close(training_out, model.proj.bias.expand_as(training_out))

        model.eval()
        expected, _ = model(x)
        actual, _ = model(x, need_weights=False)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.equal(actual, training_out))


if __name__ == "__main__":
    unittest.main()
