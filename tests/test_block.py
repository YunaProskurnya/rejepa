"""Проверки pre-norm блока и прямого residual-пути."""

import unittest

import torch
import torch.nn as nn

from src.models.block import Block


class TestBlock(unittest.TestCase):
    def test_zero_branches_preserve_input_and_gradient(self):
        block = Block(12, 3).double()
        with torch.no_grad():
            block.attn.proj.weight.zero_()
            block.attn.proj.bias.zero_()
            block.mlp.fc2.weight.zero_()
            block.mlp.fc2.bias.zero_()
        x = torch.randn(2, 5, 12, dtype=torch.float64, requires_grad=True)
        out = block(x)
        torch.testing.assert_close(out, x, rtol=0, atol=0)
        upstream = torch.randn_like(out)
        (out * upstream).sum().backward()
        torch.testing.assert_close(x.grad, upstream, rtol=0, atol=0)

    def test_matches_pytorch_pre_norm_layer(self):
        torch.manual_seed(17)
        block = Block(12, 3).double()
        reference = nn.TransformerEncoderLayer(
            d_model=12, nhead=3, dim_feedforward=48, dropout=0.0,
            activation="gelu", layer_norm_eps=1e-6,
            batch_first=True, norm_first=True,
        ).double()
        parameter_pairs = [
            (block.attn.qkv.weight, reference.self_attn.in_proj_weight),
            (block.attn.qkv.bias, reference.self_attn.in_proj_bias),
            (block.attn.proj.weight, reference.self_attn.out_proj.weight),
            (block.attn.proj.bias, reference.self_attn.out_proj.bias),
        ]
        for ours, theirs in [
            (block.norm1, reference.norm1), (block.norm2, reference.norm2),
            (block.mlp.fc1, reference.linear1), (block.mlp.fc2, reference.linear2),
        ]:
            parameter_pairs.extend([(ours.weight, theirs.weight), (ours.bias, theirs.bias)])
        with torch.no_grad():
            for ours, theirs in parameter_pairs:
                theirs.copy_(ours)
        x = torch.randn(2, 5, 12, dtype=torch.float64, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_(True)
        out = block(x)
        expected = reference(reference_x)
        torch.testing.assert_close(out, expected)
        upstream = torch.randn_like(out)
        (out * upstream).sum().backward()
        (expected * upstream).sum().backward()
        torch.testing.assert_close(x.grad, reference_x.grad)
        for ours, theirs in parameter_pairs:
            torch.testing.assert_close(ours.grad, theirs.grad)


if __name__ == "__main__":
    unittest.main()
