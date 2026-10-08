import copy
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from src.losses import prediction_loss
from src.models.encoder import VisionTransformer
from src.models.predictor import VisionTransformerPredictor
from src.models.target_encoder import create_target_encoder, prepare_targets


class TestMultiblock(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81)
        self.encoder = VisionTransformer(image_size=8, patch_size=4, embed_dim=12, depth=2, num_heads=3)
        self.predictor = VisionTransformerPredictor(4, embed_dim=12, predictor_embed_dim=8, depth=2, num_heads=2)
        self.teacher = create_target_encoder(self.encoder)
        self.images = torch.randn(2, 3, 8, 8)
        self.context_indices = torch.tensor([[0, 1], [2, 3]])
        self.blocks = [torch.tensor([[2, 3], [0, 1]]), torch.tensor([[3], [0]])]

    def test_different_block_lengths_and_one_teacher_pass(self):
        context = self.encoder(self.images, self.context_indices)
        predictions = self.predictor(context, self.context_indices, self.blocks)
        with patch.object(self.teacher, "forward", wraps=self.teacher.forward) as forward:
            targets = prepare_targets(self.teacher, self.images, self.blocks)
        self.assertEqual(forward.call_count, 1)
        self.assertEqual([tuple(p.shape) for p in predictions], [(2, 2, 12), (2, 1, 12)])
        for predicted, target, indices in zip(predictions, targets, self.blocks):
            torch.testing.assert_close(predicted, self.predictor(context, self.context_indices, indices))
            torch.testing.assert_close(target, prepare_targets(self.teacher, self.images, indices))
            self.assertFalse(target.requires_grad)
        altered = [self.blocks[0], torch.tensor([[2], [1]])]
        torch.testing.assert_close(
            predictions[0], self.predictor(context, self.context_indices, altered)[0]
        )
        prediction_loss(predictions, targets).backward()
        for parameter in [self.predictor.mask_token, self.encoder.patch_embed.proj.weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))

    def test_equal_blocks_match_batch_packing_outputs_loss_and_gradients(self):
        reference = copy.deepcopy(self.predictor)
        context = torch.randn(2, 2, 12, requires_grad=True)
        reference_context = context.detach().clone().requires_grad_(True)
        blocks = [self.blocks[0], self.blocks[0].flip(1)]
        predictions = self.predictor(context, self.context_indices, blocks)
        packed = reference(reference_context.repeat(2, 1, 1), self.context_indices.repeat(2, 1), torch.cat(blocks))
        torch.testing.assert_close(torch.cat(predictions), packed)
        targets = prepare_targets(self.teacher, self.images, blocks)
        loss = prediction_loss(predictions, targets)
        reference_loss = F.smooth_l1_loss(packed, torch.cat(targets))
        torch.testing.assert_close(loss, reference_loss)
        loss.backward()
        reference_loss.backward()
        torch.testing.assert_close(context.grad, reference_context.grad)
        for ours, theirs in zip(self.predictor.parameters(), reference.parameters()):
            if ours.requires_grad:
                torch.testing.assert_close(ours.grad, theirs.grad, rtol=1e-4, atol=1e-6)

    def test_empty_target_lists_are_rejected(self):
        with self.assertRaises(ValueError):
            self.predictor(torch.zeros(2, 2, 12), self.context_indices, [])
        with self.assertRaises(ValueError):
            prepare_targets(self.teacher, self.images, [])


class TestPredictionLoss(unittest.TestCase):
    def test_known_values_and_gradient(self):
        prediction = torch.tensor([[[0.5, 3.0]]], requires_grad=True)
        target = torch.tensor([[[0.0, 1.0]]])
        loss = prediction_loss(prediction, target)
        torch.testing.assert_close(loss, torch.tensor(0.8125))
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([[[0.25, 0.5]]]))

    def test_different_lengths_have_equal_block_weights(self):
        predictions = [torch.ones(1, 1, 1), torch.full((1, 3, 1), 2.0)]
        targets = [torch.zeros_like(p) for p in predictions]
        torch.testing.assert_close(prediction_loss(predictions, targets), torch.tensor(1.0))

    def test_mismatched_and_empty_blocks_are_rejected(self):
        cases = [([], []), ([torch.zeros(1, 1, 2)], []),
                 (torch.zeros(2, 3, 4), torch.zeros(1, 3, 4)),
                 (torch.zeros(1, 0, 4), torch.zeros(1, 0, 4))]
        for predictions, targets in cases:
            with self.assertRaises(ValueError):
                prediction_loss(predictions, targets)


if __name__ == "__main__":
    unittest.main()
