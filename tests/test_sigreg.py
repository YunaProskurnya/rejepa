import unittest

import torch

from src.regularizers.sigreg import SIGReg


class TestSIGReg(unittest.TestCase):
    def test_matches_upstream_minimal_loss_and_gradients(self):
        torch.manual_seed(17)
        z = torch.randn(128, 768, requires_grad=True)
        reference = z.detach().clone().requires_grad_()
        t = torch.linspace(0, 3, 17, dtype=torch.float32)
        dt = 3 / 16
        weights = torch.full((17,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2)

        torch.manual_seed(23)
        directions = torch.randn(reference.size(-1), 256)
        directions = directions.div_(directions.norm(p=2, dim=0))
        angles = (reference @ directions).unsqueeze(-1) * t
        error = (angles.cos().mean(-3) - window).square() + angles.sin().mean(-3).square()
        expected = ((error @ (weights * window)) * reference.size(-2)).mean()
        expected.backward()

        torch.manual_seed(23)
        actual = SIGReg()(z)
        actual.backward()

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(z.grad, reference.grad, rtol=0, atol=0)

    def test_quadrature_buffers_follow_module_dtype_and_state(self):
        regularizer = SIGReg().double()

        self.assertEqual(set(regularizer.state_dict()), {"t", "phi", "weights"})
        for buffer in regularizer.buffers():
            self.assertEqual(buffer.dtype, torch.float64)

        z = torch.randn(16, 4, dtype=torch.float64, requires_grad=True)
        loss = regularizer(z)
        loss.backward()
        self.assertEqual(loss.dtype, torch.float64)
        self.assertTrue(torch.isfinite(z.grad).all())

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

    def test_autocast_matches_float32_loss_and_gradients(self):
        for device in ("cpu", "cuda"):
            if device == "cuda" and not torch.cuda.is_available():
                continue
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(device=device, dtype=dtype):
                    torch.manual_seed(37)
                    z = torch.randn(128, 768, device=device).to(dtype).requires_grad_()
                    reference = z.detach().float().requires_grad_()
                    regularizer = SIGReg()

                    torch.manual_seed(41)
                    expected = regularizer(reference)
                    expected.backward()
                    torch.manual_seed(41)
                    with torch.autocast(device, dtype=torch.bfloat16):
                        actual = regularizer(z)
                    actual.backward()

                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(z.grad, reference.grad.to(dtype), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
