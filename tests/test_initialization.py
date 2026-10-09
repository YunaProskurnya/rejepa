"""Соответствие depth scaling авторскому fix_init_weight (commit 52c1ae95)."""

import math
import unittest
from unittest.mock import patch

import torch

from src.models.encoder import VisionTransformer
from src.models.predictor import VisionTransformerPredictor


class TestReferenceInitialization(unittest.TestCase):
    def check_reference_scaling(self, factory, blocks_name):
        # Запоминаем реальные случайные веса до глубинной поправки: проверяем
        # формулу для каждого элемента и отсутствие изменений в других слоях.
        before_scaling = {}
        original_init = torch.nn.init.trunc_normal_

        def record_initialization(tensor, *args, **kwargs):
            result = original_init(tensor, *args, **kwargs)
            before_scaling[id(tensor)] = tensor.detach().clone()
            return result

        torch.manual_seed(71)
        with patch("torch.nn.init.trunc_normal_", side_effect=record_initialization):
            model = factory()

        divisors = {}
        for layer_id, block in enumerate(getattr(model, blocks_name), start=1):
            divisors[id(block.attn.proj.weight)] = math.sqrt(2.0 * layer_id)
            divisors[id(block.mlp.fc2.weight)] = math.sqrt(2.0 * layer_id)
        self.assertEqual(len(divisors), 6)
        for name, parameter in model.named_parameters():
            if id(parameter) in before_scaling:
                expected = before_scaling[id(parameter)] / divisors.get(id(parameter), 1.0)
                with self.subTest(parameter=name):
                    torch.testing.assert_close(parameter, expected, rtol=0, atol=0)

        # Поправка применяется при создании, а загрузка готового checkpoint
        # сохраняет веса буквально, без повторного масштабирования.
        restored = factory()
        restored.load_state_dict(model.state_dict())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)

    def test_encoder_residual_initialization(self):
        self.check_reference_scaling(
            lambda: VisionTransformer(image_size=8, patch_size=4, embed_dim=12,
                                      depth=3, num_heads=3),
            "blocks",
        )

    def test_predictor_residual_initialization(self):
        self.check_reference_scaling(
            lambda: VisionTransformerPredictor(num_patches=4, embed_dim=12,
                                               predictor_embed_dim=8, depth=3, num_heads=2),
            "predictor_blocks",
        )


if __name__ == "__main__":
    unittest.main()
