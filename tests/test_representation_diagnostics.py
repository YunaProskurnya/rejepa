import csv
import json
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
    def test_spectrum_and_variance_without_covariance_storage(self):
        features = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        statistics = spectrum_statistics(features)
        self.assertNotIn("covariance", statistics)
        torch.testing.assert_close(statistics["singular_values"], torch.full((2,), math.sqrt(2 / 3)))
        self.assertAlmostEqual(statistics["singular_entropy"], math.log(2))
        self.assertAlmostEqual(statistics["effective_rank"], 2)
        self.assertAlmostEqual(statistics["mean_variance"], 2 / 3)

        collapsed = spectrum_statistics(torch.ones(4, 2))
        self.assertEqual(collapsed["singular_entropy"], 0)
        self.assertEqual(collapsed["effective_rank"], 0)

    def test_training_records_selected_epochs_and_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "val" / "class_a"
            data.mkdir(parents=True)
            for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
                Image.new("RGB", (40, 40), color).save(data / f"{index}.png")

            run = root / "run"
            command = [sys.executable, str(ROOT / "tests/run_training.py"), "--smoke-test",
                       "--epochs", "3", "--representation-data", str(data.parent),
                       "--representation-samples", "3", "--output", str(run)]
            for steps, expected in ((3, []), (4, [0]), (8, [0]), (12, [0, 2])):
                arguments = [*command, "--max-steps", str(steps)]
                if steps > 3:
                    arguments.extend(("--resume", str(run / "checkpoint.pt")))
                result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

                output = run / "representation_analysis"
                if not expected:
                    self.assertFalse((output / "representations.csv").exists())
                    continue
                with (output / "representations.csv").open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([int(row["epoch"]) for row in rows], expected)
                self.assertEqual([int(row["step"]) for row in rows], [4 * (epoch + 1) for epoch in expected])
                for row in rows:
                    statistics = torch.load(output / row["artifact"], weights_only=True)
                    self.assertEqual(len(statistics["singular_values"]), 2)
                    self.assertNotIn("covariance", statistics)
                    self.assertTrue(math.isfinite(float(row["mean_variance"])))

            settings = json.loads((run / "representation_analysis/settings.json").read_text(encoding="utf-8"))
            self.assertEqual(len(settings["paths"]), 3)
            self.assertEqual(settings["every"], 25)

            baseline = root / "baseline"
            result = subprocess.run([sys.executable, str(ROOT / "tests/run_training.py"), "--smoke-test",
                                     "--epochs", "3", "--max-steps", "12", "--output", str(baseline)],
                                    cwd=ROOT, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            checkpoint = torch.load(run / "checkpoint.pt", weights_only=True)
            reference = torch.load(baseline / "checkpoint.pt", weights_only=True)
            for component in ("encoder", "predictor", "target_encoder"):
                for name, value in checkpoint[component].items():
                    torch.testing.assert_close(value, reference[component][name], rtol=0, atol=0)

            joint = root / "joint"
            result = subprocess.run([sys.executable, str(ROOT / "tests/run_training.py"), "--smoke-test",
                                     "--mode", "joint", "--regularizer", "visreg", "--epochs", "1",
                                     "--representation-data", str(data.parent), "--output", str(joint)],
                                    cwd=ROOT, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            joint_checkpoint = torch.load(joint / "checkpoint.pt", weights_only=True)
            self.assertNotIn("target_encoder", joint_checkpoint)
            joint_settings = json.loads((joint / "representation_analysis/settings.json").read_text(encoding="utf-8"))
            self.assertEqual(joint_settings["encoder_source"], "encoder")
            self.assertEqual(joint_settings["regularizer"], "visreg")
            self.assertTrue((joint / "representation_analysis/representations.csv").exists())

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
                self.assertNotIn("covariance", statistics)
                self.assertEqual(statistics["images"], 3)
                self.assertEqual(len(statistics["singular_values"]), 2)
                self.assertTrue(torch.isfinite(statistics["singular_values"]).all())

            joint = root / "joint"
            joint.mkdir()
            torch.save(dict(config={"encoder": options, "mode": "joint", "regularizer": "sigreg",
                                    "regularizer_weight": 0.02}, encoder=encoder.state_dict(), step=10),
                       joint / "checkpoint_0.pt")
            result = subprocess.run([*command[:2], "--checkpoints", str(joint), "--data", str(data.parent),
                                     "--samples", "3"], cwd=ROOT, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            joint_statistics = torch.load(joint / "representation_analysis/spectrum_0.pt", weights_only=True)
            settings = json.loads((joint / "representation_analysis/settings.json").read_text(encoding="utf-8"))
            self.assertEqual(settings["encoder_source"], "encoder")
            self.assertEqual(settings["regularizer"], "sigreg")
            self.assertNotIn("covariance", joint_statistics)
            self.assertTrue(torch.isfinite(joint_statistics["singular_values"]).all())


if __name__ == "__main__":
    unittest.main()
