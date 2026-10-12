import unittest

import torch

from src.regularizers.sigreg import SIGReg


class TestSIGReg(unittest.TestCase):
    def test_gaussian_has_lower_loss_than_collapse_or_shift(self):
        torch.manual_seed(17)
        gaussian = torch.randn(512, 8)
        regularizer = SIGReg(num_slices=128)

        torch.manual_seed(23)
        gaussian_loss = regularizer(gaussian)
        torch.manual_seed(23)
        collapsed_loss = regularizer(torch.zeros_like(gaussian))
        torch.manual_seed(23)
        shifted_loss = regularizer(gaussian + 3)

        self.assertLess(gaussian_loss.item(), collapsed_loss.item())
        self.assertLess(gaussian_loss.item(), shifted_loss.item())

    def test_gradients_and_input_validation(self):
        torch.manual_seed(31)
        z = torch.randn(32, 4, requires_grad=True)
        loss = SIGReg(num_slices=16)(z)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad.abs().sum().item(), 0)

        with self.assertRaises(ValueError):
            SIGReg()(torch.randn(1, 4))


if __name__ == "__main__":
    unittest.main()
