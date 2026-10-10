import unittest

import torch

from src.reference.lars import LARS


class TestLARS(unittest.TestCase):
    def test_weight_decay_trust_ratio_and_two_step_momentum(self):
        weight = torch.nn.Parameter(torch.tensor([[3.0, 4.0]], dtype=torch.float64))
        optimizer = LARS([weight], lr=0.2, weight_decay=0.5)

        # Decay turns [2.5, 1] into [4, 3]; its norm and the parameter
        # norm are both 5, giving the default trust ratio of 0.001.
        weight.grad = torch.tensor([[2.5, 1.0]], dtype=torch.float64)
        optimizer.step()
        torch.testing.assert_close(
            weight, torch.tensor([[2.9992, 3.9994]], dtype=torch.float64),
            rtol=0, atol=1e-12,
        )
        torch.testing.assert_close(
            optimizer.state[weight]['mu'],
            torch.tensor([[0.004, 0.003]], dtype=torch.float64),
            rtol=0, atol=1e-12,
        )

        # Cancel the decay on step two. With a zero fresh update, the
        # existing buffer must still move the weight using momentum 0.9.
        weight.grad = -0.5 * weight.detach().clone()
        optimizer.step()
        torch.testing.assert_close(
            weight, torch.tensor([[2.99848, 3.99886]], dtype=torch.float64),
            rtol=0, atol=1e-12,
        )
        self.assertTrue(torch.isfinite(weight).all())

    def test_scalar_and_vector_biases_skip_decay_and_adaptation(self):
        bias = torch.nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
        scalar = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        optimizer = LARS([bias, scalar], lr=0.2, weight_decay=100.0)
        bias.grad = torch.tensor([2.0, -1.0], dtype=torch.float64)
        scalar.grad = torch.tensor(0.5, dtype=torch.float64)
        optimizer.step()
        torch.testing.assert_close(
            bias, torch.tensor([2.6, 4.2], dtype=torch.float64),
            rtol=0, atol=1e-12,
        )
        torch.testing.assert_close(
            scalar, torch.tensor(1.9, dtype=torch.float64), rtol=0, atol=1e-12,
        )

        bias.grad = torch.zeros_like(bias)
        scalar.grad = torch.zeros_like(scalar)
        optimizer.step()
        torch.testing.assert_close(
            bias, torch.tensor([2.24, 4.38], dtype=torch.float64),
            rtol=0, atol=1e-12,
        )
        torch.testing.assert_close(
            scalar, torch.tensor(1.81, dtype=torch.float64), rtol=0, atol=1e-12,
        )

    def test_zero_parameter_norm_uses_unscaled_update(self):
        weight = torch.nn.Parameter(torch.zeros((1, 2), dtype=torch.float64))
        optimizer = LARS([weight], lr=0.2, weight_decay=0.5)
        weight.grad = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
        optimizer.step()
        torch.testing.assert_close(
            weight, torch.tensor([[-0.6, -0.8]], dtype=torch.float64),
            rtol=0, atol=1e-12,
        )
        self.assertTrue(torch.isfinite(weight).all())

    def test_zero_gradient_and_missing_gradient_are_safe(self):
        stationary = torch.nn.Parameter(torch.tensor([[3.0, 4.0]]))
        missing = torch.nn.Parameter(torch.tensor([[5.0, 6.0]]))
        optimizer = LARS([stationary, missing], lr=0.2, weight_decay=0.0)
        stationary.grad = torch.zeros_like(stationary)
        optimizer.step()
        torch.testing.assert_close(stationary, torch.tensor([[3.0, 4.0]]))
        torch.testing.assert_close(missing, torch.tensor([[5.0, 6.0]]))
        self.assertNotIn(missing, optimizer.state)
        self.assertTrue(torch.isfinite(stationary).all())


if __name__ == '__main__':
    unittest.main()
