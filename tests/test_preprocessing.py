from pathlib import Path
import subprocess
import sys
import unittest

import torch

from src.masks import pick_context_patches, pick_target_patches
from src.transforms import process_image


ROOT = Path(__file__).resolve().parents[1]


class TestPreprocessing(unittest.TestCase):
    def test_scale_is_area_fraction(self):
        targets = pick_target_patches(16, scale=(0.25, 0.25), aspect_ratio=(1.0, 1.0))
        self.assertEqual([indices.numel() for indices in targets], [64] * 4)
        context = pick_context_patches(16, [], scale=(0.5625, 0.5625))
        self.assertEqual(context.numel(), 144)

    def test_image_and_mask_contract_and_seed(self):
        path = str(ROOT / "example" / "lenna.png")
        for seed in (0, 42):
            image, targets, context = process_image(path, seed=seed)
            repeated_image, repeated_targets, repeated_context = process_image(path, seed=seed)
            self.assertEqual(tuple(image.shape), (3, 224, 224))
            self.assertEqual(image.dtype, torch.float32)
            self.assertTrue(torch.isfinite(image).all())
            self.assertEqual(len(targets), 4)
            self.assertGreater(context.numel(), 10)
            self.assertEqual(len({indices.numel() for indices in targets}), 1)
            for indices in [context, *targets]:
                self.assertEqual(indices.ndim, 1)
                self.assertEqual(indices.dtype, torch.long)
                self.assertGreater(indices.numel(), 0)
                self.assertTrue(((indices >= 0) & (indices < 196)).all())
                self.assertEqual(indices.unique().numel(), indices.numel())
            self.assertFalse(torch.isin(context, torch.cat(targets)).any())
            torch.testing.assert_close(image, repeated_image, rtol=0, atol=0)
            torch.testing.assert_close(context, repeated_context)
            for indices, repeated in zip(targets, repeated_targets):
                torch.testing.assert_close(indices, repeated)

    def test_training_script_respects_seed(self):
        outputs = []
        for seed in (0, 42, 42):
            result = subprocess.run(
                [sys.executable, str(ROOT / "example" / "mini_experiment.py"),
                 "--steps", "2", "--seed", str(seed)],
                cwd=ROOT, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("step=002 loss=", result.stdout)
            outputs.append(result.stdout)
        self.assertNotEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])


if __name__ == "__main__":
    unittest.main()
