import unittest
from unittest.mock import patch

import torch

from src.regularizers import SIGReg
from src.regularizers.apply_regularization import apply_regularization


class TestApplyRegularization(unittest.TestCase):
    def test_each_patch_position_uses_independent_images(self):
        features = torch.tensor([[[1.0], [4.0]], [[3.0], [8.0]]], requires_grad=True)
        samples = []

        def regularizer(_, z):
            samples.append(z.detach().clone())
            return z.mean().square()

        with patch.object(SIGReg, "forward", regularizer):
            loss = apply_regularization(features, "sigreg")

        torch.testing.assert_close(samples[0], torch.tensor([[1.0], [3.0]]))
        torch.testing.assert_close(samples[1], torch.tensor([[4.0], [8.0]]))
        torch.testing.assert_close(loss, torch.tensor(20.0))

        loss.backward()
        torch.testing.assert_close(
            features.grad,
            torch.tensor([[[1.0], [3.0]], [[1.0], [3.0]]]),
        )

    def test_none_returns_zero_without_features(self):
        self.assertEqual(apply_regularization(None, "none"), 0)

    def test_visreg_returns_differentiable_loss(self):
        features = torch.randn(3, 2, 4, requires_grad=True)
        loss = apply_regularization(features, "visreg")

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(features.grad)

    def test_unknown_regularizer_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_regularization(None, "unknown")


if __name__ == "__main__":
    unittest.main()
