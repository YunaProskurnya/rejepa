import unittest

import torch

from src.models.encoder import VisionTransformer
from src.models.predictor import VisionTransformerPredictor
from src.models.target_encoder import create_target_encoder
from src.optimizer import create_optimizer
from src.reference.schedulers import CosineWDSchedule


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.encoder = VisionTransformer(
            image_size=8, patch_size=4, embed_dim=12, depth=1, num_heads=3,
        )
        self.predictor = VisionTransformerPredictor(
            4, embed_dim=12, predictor_embed_dim=8, depth=1, num_heads=2,
        )
        self.optimizer = create_optimizer(self.encoder, self.predictor, lr=0.1, weight_decay=0.2)

    def test_groups_cover_trainable_parameters_once(self):
        groups = self.optimizer.param_groups
        self.assertEqual(len(groups), 2)
        self.assertEqual([group["weight_decay"] for group in groups], [0.2, 0.0])
        self.assertEqual([group["lr"] for group in groups], [0.1, 0.1])
        ids = [id(parameter) for group in groups for parameter in group["params"]]
        expected = {
            id(parameter) for module in (self.encoder, self.predictor)
            for parameter in module.parameters() if parameter.requires_grad
        }
        self.assertEqual(set(ids), expected)
        self.assertEqual(len(ids), len(expected))

        decay_ids = {id(parameter) for parameter in groups[0]["params"]}
        no_decay_ids = {id(parameter) for parameter in groups[1]["params"]}
        for parameter in (self.encoder.patch_embed.proj.weight, self.predictor.mask_token,
                          self.predictor.predictor_proj.weight):
            self.assertIn(id(parameter), decay_ids)
        for parameter in (self.encoder.norm.weight, self.encoder.norm.bias,
                          self.encoder.patch_embed.proj.bias, self.predictor.predictor_norm.weight):
            self.assertIn(id(parameter), no_decay_ids)
        self.assertNotIn(id(self.encoder.pos_embed), ids)
        self.assertNotIn(id(self.predictor.predictor_pos_embed), ids)
        teacher = create_target_encoder(self.encoder)
        self.assertTrue(set(ids).isdisjoint(id(parameter) for parameter in teacher.parameters()))

    def test_decay_only_affects_first_group_but_both_groups_learn(self):
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                with torch.no_grad():
                    parameter.fill_(1.0)
                parameter.grad = torch.zeros_like(parameter)

        self.optimizer.step()

        for group, expected in zip(self.optimizer.param_groups, (0.98, 1.0)):
            for parameter in group["params"]:
                torch.testing.assert_close(parameter, torch.full_like(parameter, expected))
                parameter.grad = torch.ones_like(parameter)

        self.optimizer.step()

        for group, previous in zip(self.optimizer.param_groups, (0.98, 1.0)):
            for parameter in group["params"]:
                self.assertTrue(torch.isfinite(parameter).all())
                self.assertTrue((parameter < previous).all())

    def test_author_schedule_preserves_no_decay_group(self):
        schedule = CosineWDSchedule(self.optimizer, ref_wd=0.2, final_wd=0.4, T_max=4)
        for _ in range(4):
            value = schedule.step()
            self.assertEqual(self.optimizer.param_groups[0]["weight_decay"], value)
            self.assertEqual(self.optimizer.param_groups[1]["weight_decay"], 0.0)
        self.assertAlmostEqual(value, 0.4)


if __name__ == "__main__":
    unittest.main()
