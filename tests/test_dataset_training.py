import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch

from src.models.encoder import VisionTransformer


ROOT = Path(__file__).resolve().parents[1]


class TestDatasetTraining(unittest.TestCase):
    def test_smoke_run_crosses_epoch_and_saves_usable_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/train.py"), "--config", str(ROOT / "configs/smoke.json"),
                 "--epochs", "2", "--warmup-epochs", "1", "--max-steps", "5",
                 "--output", str(output)],
                cwd=ROOT, capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["total_steps"], 8)
            self.assertEqual(config["collator"], "custom")
            self.assertIn("collator=custom", result.stdout)
            with (output / "metrics.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["epoch"]) for row in rows], [1, 1, 1, 1, 2])
            for row in rows:
                self.assertTrue(math.isfinite(float(row["loss"])))
                self.assertGreater(float(row["grad_norm"]), 0)
                self.assertGreater(int(row["context_tokens"]), 10)
            self.assertGreater(len({row["context_tokens"] for row in rows}), 1)
            self.assertGreater(float(rows[-1]["momentum"]), float(rows[0]["momentum"]))

            checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["config"], config)
            self.assertEqual(config["encoder"]["mlp_ratio"], 4.0)
            self.assertEqual(config["optimizer"]["defaults"]["betas"], [0.9, 0.999])
            self.assertEqual(config["preprocessing"]["random_resized_crop"]["scale"], [0.3, 1.0])
            self.assertEqual(config["objective"]["beta"], 1.0)
            self.assertEqual(config["schedules"]["warmup_steps"], 4)
            self.assertEqual(checkpoint["step"], 5)
            self.assertEqual(checkpoint["config"]["collator"], "custom")
            self.assertEqual(checkpoint["optimizer"]["param_groups"][1]["weight_decay"], 0.0)
            self.assertTrue(all(state["step"].item() == 5 for state in checkpoint["optimizer"]["state"].values()))
            encoder = VisionTransformer(**config["encoder"])
            encoder.load_state_dict(checkpoint["encoder"])
            with torch.no_grad():
                features = encoder(torch.zeros(1, 3, 224, 224))
            self.assertEqual(tuple(features.shape), (1, 196, 48))
            self.assertTrue(torch.isfinite(features).all())

    def test_imagefolder_run_with_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images" / "unlabelled"
            images.mkdir(parents=True)
            for index in range(2):
                shutil.copyfile(ROOT / "example/lenna.png", images / f"{index}.png")
            for collator in ("reference", "custom"):
                with self.subTest(collator=collator):
                    output = root / collator
                    result = subprocess.run(
                        [sys.executable, str(ROOT / "scripts/train.py"),
                         "--data", str(images.parent), "--collator", collator, "--num-workers", "1", "--max-steps", "1",
                         "--output", str(output)],
                        cwd=ROOT, capture_output=True, text=True, timeout=90,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertTrue((output / "checkpoint.pt").is_file())
                    self.assertIn("images=2", result.stdout)


if __name__ == "__main__":
    unittest.main()
