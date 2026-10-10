import csv
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from src.representation_diagnostics import spectrum_statistics


ROOT = Path(__file__).resolve().parents[1]


class TestRepresentationDiagnostics(unittest.TestCase):
    def test_covariance_and_singular_entropy(self):
        features = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        statistics = spectrum_statistics(features)
        torch.testing.assert_close(statistics["covariance"], torch.eye(2) * (2 / 3))
        self.assertAlmostEqual(statistics["singular_entropy"], math.log(2))
        self.assertAlmostEqual(statistics["effective_rank"], 2)

        collapsed = spectrum_statistics(torch.ones(4, 2))
        self.assertEqual(collapsed["singular_entropy"], 0)
        self.assertEqual(collapsed["effective_rank"], 0)

    def test_training_writes_baseline_and_updated_covariance(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/train.py"), "--smoke-test",
                 "--representation-every", "1", "--representation-samples", "4",
                 "--max-steps", "1", "--output", str(output)],
                cwd=ROOT, capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with (output / "representations.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["step"]) for row in rows], [0, 1])
            for row in rows:
                statistics = torch.load(output / row["artifact"], weights_only=True)
                self.assertEqual(tuple(statistics["covariance"].shape), (48, 48))
                self.assertEqual(statistics["images"], 4)
                self.assertTrue(torch.isfinite(statistics["singular_values"]).all())

            resumed = subprocess.run(
                [sys.executable, str(ROOT / "scripts/train.py"), "--smoke-test",
                 "--representation-every", "1", "--representation-samples", "4",
                 "--max-steps", "2", "--output", str(output),
                 "--resume", str(output / "checkpoint.pt")],
                cwd=ROOT, capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            with (output / "representations.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["step"]) for row in rows], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
