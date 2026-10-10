import csv
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image
import torch

from src.models.encoder import VisionTransformer
from src.representation_diagnostics import spectrum_statistics


ROOT = Path(__file__).resolve().parents[1]


class TestRepresentationDiagnostics(unittest.TestCase):
    def test_covariance_and_singular_entropy(self):
        features = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        statistics = spectrum_statistics(features)
        torch.testing.assert_close(statistics["covariance"], torch.eye(2) * (2 / 3))
        torch.testing.assert_close(statistics["singular_values"], torch.full((2,), math.sqrt(2 / 3)))
        self.assertAlmostEqual(statistics["singular_entropy"], math.log(2))
        self.assertAlmostEqual(statistics["effective_rank"], 2)
        self.assertAlmostEqual(statistics["mean_variance"], 2 / 3)

        collapsed = spectrum_statistics(torch.ones(4, 2))
        self.assertEqual(collapsed["singular_entropy"], 0)
        self.assertEqual(collapsed["effective_rank"], 0)

    def test_analyzes_finished_checkpoints_and_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "val" / "class_a"
            data.mkdir(parents=True)
            for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
                Image.new("RGB", (40, 40), color).save(data / f"{index}.png")

            run = root / "run"
            run.mkdir()
            options = dict(image_size=32, patch_size=8, in_channels=3, embed_dim=12,
                           depth=1, num_heads=3)
            encoder = VisionTransformer(**options)
            for epoch in range(4):
                weights = {name: value.clone() for name, value in encoder.state_dict().items()}
                if epoch == 0:
                    weights["norm.weight"].zero_()
                torch.save(dict(config={"encoder": options}, target_encoder=weights,
                                step=(epoch + 1) * 10), run / f"checkpoint_{epoch}.pt")

            command = [sys.executable, str(ROOT / "scripts/analyze_checkpoints.py"),
                       "--checkpoints", str(run), "--data", str(data.parent),
                       "--samples", "3", "--batch-size", "2", "--every", "2"]
            for _ in range(2):
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            output = run / "representation_analysis"
            with (output / "representations.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["epoch"]) for row in rows], [0, 1, 3])
            self.assertEqual([int(row["step"]) for row in rows], [10, 20, 40])
            self.assertEqual(float(rows[0]["effective_rank"]), 0)
            self.assertGreater(float(rows[1]["effective_rank"]), 0)
            for row in rows:
                statistics = torch.load(output / row["artifact"], weights_only=True)
                self.assertEqual(tuple(statistics["covariance"].shape), (12, 12))
                self.assertEqual(statistics["images"], 3)
                self.assertTrue(torch.isfinite(statistics["singular_values"]).all())


if __name__ == "__main__":
    unittest.main()
