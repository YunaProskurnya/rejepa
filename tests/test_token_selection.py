"""Проверка выбора патчей и маршрута градиентов."""

import unittest

import torch

from src.models.token_selection import select_tokens


class TestTokenSelection(unittest.TestCase):
    def test_per_image_selection_and_gradients(self):
        tokens = torch.tensor([
            [[10., 11.], [20., 21.], [30., 31.], [40., 41.]],
            [[50., 51.], [60., 61.], [70., 71.], [80., 81.]],
        ], requires_grad=True)
        indices = torch.tensor([[3, 1], [0, 2]], dtype=torch.long)

        selected = select_tokens(tokens, indices)
        expected = torch.tensor([
            [[40., 41.], [20., 21.]],
            [[50., 51.], [70., 71.]],
        ])
        torch.testing.assert_close(selected, expected)

        selected.sum().backward()
        expected_grad = torch.tensor([
            [[0., 0.], [1., 1.], [0., 0.], [1., 1.]],
            [[1., 1.], [0., 0.], [1., 1.], [0., 0.]],
        ])
        torch.testing.assert_close(tokens.grad, expected_grad)


if __name__ == "__main__":
    unittest.main()
