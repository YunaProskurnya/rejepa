import unittest

import torch

from src.regularizers.visreg import VISReg


class TestVISReg(unittest.TestCase):
    def test_center_term_uses_mean_squared_value(self):
        z = torch.tensor([[1., -2.], [3., -4.]], requires_grad=True)
        regularizer = VISReg(num_slices=1, scale_weight=0, shape_weight=0)

        loss = regularizer(z)
        torch.testing.assert_close(loss, torch.tensor(6.5))

        loss.backward()
        torch.testing.assert_close(z.grad, torch.tensor([[1., -1.5], [1., -1.5]]))

    def test_shape_term_averages_over_batch(self):
        z = torch.zeros(3, 2)
        regularizer = VISReg(num_slices=4, scale_weight=0, center_weight=0)
        probabilities = torch.tensor([0.25, 0.5, 0.75])
        quantiles = torch.erfinv(2 * probabilities - 1) * (2 ** 0.5)

        torch.testing.assert_close(regularizer(z), quantiles.square().mean())

    def test_epsilon_does_not_change_scale_loss(self):
        z = torch.zeros(4, 2)
        regularizer = VISReg(num_slices=1, shape_weight=0, center_weight=0, eps=0.5)

        torch.testing.assert_close(regularizer(z), torch.tensor(1.0))

    def test_gaussian_has_lower_loss_than_collapse(self):
        torch.manual_seed(41)
        gaussian = torch.randn(256, 4)
        regularizer = VISReg(num_slices=64)

        torch.manual_seed(43)
        gaussian_loss = regularizer(gaussian)
        torch.manual_seed(43)
        collapsed_loss = regularizer(torch.zeros_like(gaussian))

        self.assertLess(gaussian_loss.item(), collapsed_loss.item())

    def test_shape_term_is_nearly_scale_invariant_away_from_zero(self):
        torch.manual_seed(47)
        z = torch.randn(128, 4)
        regularizer = VISReg(num_slices=32, scale_weight=0, center_weight=0, eps=1e-7)

        torch.manual_seed(53)
        original = regularizer(z)
        torch.manual_seed(53)
        scaled = regularizer(2 * z)

        torch.testing.assert_close(original, scaled, rtol=1e-4, atol=1e-4)

    def test_exact_collapse_has_finite_loss_and_gradients(self):
        torch.manual_seed(59)
        z = torch.zeros(16, 4, requires_grad=True)
        loss = VISReg(num_slices=16)(z)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad.abs().sum().item(), 0)

        with self.assertRaises(ValueError):
            VISReg()(torch.randn(1, 4))


if __name__ == "__main__":
    unittest.main()
