import math
import unittest
from unittest.mock import patch

import torch

from src.models.encoder import VisionTransformer
from src.models.predictor import VisionTransformerPredictor
from src.optimizer import create_optimizer
from src.train_step import train_step


class TestJointTraining(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.encoder = VisionTransformer(
            image_size=8, patch_size=4, embed_dim=12, depth=1, num_heads=3,
        )
        self.predictor = VisionTransformerPredictor(
            4, embed_dim=12, predictor_embed_dim=8, depth=1, num_heads=2,
        )
        self.optimizer = create_optimizer(self.encoder, self.predictor, lr=0.001, weight_decay=0.04)
        self.images = torch.randn(2, 3, 8, 8)
        self.context = torch.tensor([[0, 1], [1, 2]])
        self.targets = [torch.tensor([[2, 3], [0, 3]])]

    def step(self, regularizer="none", weight=0.02):
        return train_step(
            self.encoder, self.predictor, None, self.optimizer,
            self.images, self.context, self.targets, None,
            mode="joint", regularizer=regularizer, regularizer_weight=weight,
        )

    def test_prediction_trains_both_calls_to_the_same_encoder(self):
        outputs = []

        def collect(_, __, output):
            output.retain_grad()
            outputs.append(output)

        hook = self.encoder.register_forward_hook(collect)
        before = self.encoder.patch_embed.proj.weight.detach().clone()
        try:
            metrics = self.step(weight=0)
        finally:
            hook.remove()

        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape, (2, 4, 12))
        self.assertEqual(outputs[1].shape, (2, 2, 12))
        for output in outputs:
            self.assertIsNotNone(output.grad)
            self.assertGreater(output.grad.abs().sum().item(), 0)
        self.assertFalse(torch.equal(before, self.encoder.patch_embed.proj.weight))
        self.assertAlmostEqual(metrics["loss"], metrics["prediction_loss"])
        self.assertEqual(metrics["regularization_loss"], 0)

    def test_regularization_receives_all_full_image_patch_positions(self):
        received = []

        def regularize(features, name):
            received.append((features, name))
            return features.square().mean()

        with patch("src.train_step.apply_regularization", side_effect=regularize):
            metrics = self.step(regularizer="sigreg", weight=0.5)

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0][0].shape, (2, 4, 12))
        self.assertEqual(received[0][1], "sigreg")
        self.assertAlmostEqual(
            metrics["loss"], metrics["prediction_loss"] + 0.5 * metrics["regularization_loss"],
            places=6,
        )

    def test_both_regularizers_produce_finite_updates(self):
        for regularizer in ("sigreg", "visreg"):
            with self.subTest(regularizer=regularizer):
                metrics = self.step(regularizer=regularizer)
                self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
                self.assertGreater(metrics["regularization_loss"], 0)
                self.assertGreater(metrics["grad_norm"], 0)

    def test_joint_rejects_a_second_encoder(self):
        with self.assertRaisesRegex(ValueError, "without a separate target encoder"):
            train_step(
                self.encoder, self.predictor, self.encoder, self.optimizer,
                self.images, self.context, self.targets, None, mode="joint",
            )


if __name__ == "__main__":
    unittest.main()
