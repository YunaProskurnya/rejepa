import copy
import math
import unittest
from unittest import mock

import torch

from src.models.encoder import VisionTransformer
from src.models.predictor import VisionTransformerPredictor
from src.models.target_encoder import create_target_encoder, update_target_encoder
from src.optimizer import create_optimizer
from src.train_step import train_step


class TestTrainStep(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.encoder = VisionTransformer(
            image_size=8, patch_size=4, embed_dim=12, depth=1, num_heads=3,
        )
        self.predictor = VisionTransformerPredictor(
            4, embed_dim=12, predictor_embed_dim=8, depth=1, num_heads=2,
        )
        self.teacher = create_target_encoder(self.encoder)
        self.optimizer = create_optimizer(self.encoder, self.predictor, lr=0.001, weight_decay=0.04)
        self.images = torch.randn(2, 3, 8, 8)
        self.context = torch.tensor([[0, 1], [2, 3]])
        self.targets = [torch.tensor([[2, 3], [0, 1]]), torch.tensor([[3], [1]])]

    def test_two_steps_update_models_then_teacher_and_preserve_optimizer_state(self):
        source = self.encoder.patch_embed.proj.weight
        prediction = self.predictor.predictor_proj.weight
        positions = self.encoder.pos_embed.detach().clone()

        for step in (1, 2):
            old_source = source.detach().clone()
            old_prediction = prediction.detach().clone()
            old_teacher = {name: parameter.detach().clone()
                           for name, parameter in self.teacher.named_parameters()}
            for group in self.optimizer.param_groups:
                for parameter in group["params"]:
                    parameter.grad = torch.full_like(parameter, float("nan"))

            metrics = train_step(
                self.encoder, self.predictor, self.teacher, self.optimizer,
                self.images, self.context, self.targets, momentum=0.9,
            )

            self.assertEqual(set(metrics), {"loss", "grad_norm"})
            for value in metrics.values():
                self.assertIsInstance(value, float)
                self.assertTrue(math.isfinite(value))
            self.assertGreater(metrics["grad_norm"], 0)
            self.assertFalse(torch.equal(source, old_source))
            self.assertFalse(torch.equal(prediction, old_prediction))
            self.assertEqual(self.optimizer.state[source]["step"].item(), step)
            torch.testing.assert_close(self.encoder.pos_embed, positions, rtol=0, atol=0)

            source_parameters = dict(self.encoder.named_parameters())
            for name, parameter in self.teacher.named_parameters():
                expected = old_teacher[name]
                if source_parameters[name].requires_grad:
                    expected = 0.9 * expected + 0.1 * source_parameters[name].detach()
                torch.testing.assert_close(parameter, expected)
                self.assertIsNone(parameter.grad)
                self.assertFalse(parameter.requires_grad)

    def test_nonfinite_loss_stops_before_parameter_updates(self):
        source_before = self.encoder.patch_embed.proj.weight.detach().clone()
        teacher_before = self.teacher.patch_embed.proj.weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError, "Non-finite loss"):
            train_step(
                self.encoder, self.predictor, self.teacher, self.optimizer,
                torch.full_like(self.images, float("nan")),
                self.context, self.targets, momentum=0.9,
            )
        torch.testing.assert_close(self.encoder.patch_embed.proj.weight, source_before, rtol=0, atol=0)
        torch.testing.assert_close(self.teacher.patch_embed.proj.weight, teacher_before, rtol=0, atol=0)
        self.assertEqual(len(self.optimizer.state), 0)

    def test_accumulation_matches_full_batch_and_updates_teacher_once(self):
        self.encoder.double()
        self.predictor.double()
        self.teacher.double()
        images = self.images.double()
        reference_encoder = copy.deepcopy(self.encoder)
        reference_predictor = copy.deepcopy(self.predictor)
        reference_teacher = copy.deepcopy(self.teacher)
        reference_optimizer = create_optimizer(
            reference_encoder, reference_predictor, lr=0.001, weight_decay=0.04,
        )
        expected = train_step(
            reference_encoder, reference_predictor, reference_teacher, reference_optimizer,
            images, self.context, self.targets, momentum=0.9,
        )
        source_before = self.encoder.patch_embed.proj.weight.detach().clone()
        teacher_before = self.teacher.patch_embed.proj.weight.detach().clone()

        with mock.patch("src.train_step.update_target_encoder", wraps=update_target_encoder) as ema:
            first = train_step(
                self.encoder, self.predictor, self.teacher, self.optimizer,
                images[:1], self.context[:1], [indices[:1] for indices in self.targets],
                momentum=0.9, accumulation_steps=2, update=False,
            )
            self.assertIsNone(first["grad_norm"])
            self.assertEqual(len(self.optimizer.state), 0)
            ema.assert_not_called()
            torch.testing.assert_close(self.encoder.patch_embed.proj.weight, source_before, rtol=0, atol=0)
            torch.testing.assert_close(self.teacher.patch_embed.proj.weight, teacher_before, rtol=0, atol=0)

            last = train_step(
                self.encoder, self.predictor, self.teacher, self.optimizer,
                images[1:], self.context[1:], [indices[1:] for indices in self.targets],
                momentum=0.9, accumulation_steps=2, zero_grad=False,
            )
            ema.assert_called_once()

        self.assertAlmostEqual((first["loss"] + last["loss"]) / 2, expected["loss"], places=12)
        self.assertAlmostEqual(last["grad_norm"], expected["grad_norm"], places=10)
        for actual_model, expected_model in (
            (self.encoder, reference_encoder), (self.predictor, reference_predictor),
            (self.teacher, reference_teacher),
        ):
            for actual, expected_parameter in zip(actual_model.parameters(), expected_model.parameters()):
                torch.testing.assert_close(actual, expected_parameter, rtol=1e-8, atol=1e-10)
                if actual.grad is not None:
                    torch.testing.assert_close(actual.grad, expected_parameter.grad, rtol=1e-8, atol=1e-10)
        for state in self.optimizer.state.values():
            self.assertEqual(state["step"].item(), 1)

    def test_bf16_autocast_preserves_fp32_parameters_and_finite_gradients(self):
        output_dtypes = []
        hook = self.encoder.blocks[0].attn.qkv.register_forward_hook(
            lambda module, inputs, output: output_dtypes.append(output.dtype)
        )
        try:
            metrics = train_step(
                self.encoder, self.predictor, self.teacher, self.optimizer,
                self.images, self.context, self.targets, momentum=0.9, precision="bf16",
            )
        finally:
            hook.remove()

        self.assertEqual(output_dtypes, [torch.bfloat16])
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        for model in (self.encoder, self.predictor, self.teacher):
            for parameter in model.parameters():
                self.assertEqual(parameter.dtype, torch.float32)
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertTrue(all(parameter.grad is None for parameter in self.teacher.parameters()))

    def test_invalid_precision_or_accumulation_does_not_update_parameters(self):
        for options in ({"precision": "fp16"}, {"accumulation_steps": 0}, {"accumulation_steps": 1.5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                train_step(
                    self.encoder, self.predictor, self.teacher, self.optimizer,
                    self.images, self.context, self.targets, momentum=0.9, **options,
                )
        self.assertEqual(len(self.optimizer.state), 0)


if __name__ == "__main__":
    unittest.main()
