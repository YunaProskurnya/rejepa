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


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/train.py"


class TestGPULauncher(unittest.TestCase):
    def launch(self, *arguments, success=True):
        result = subprocess.run(
            [sys.executable, str(LAUNCHER), *map(str, arguments)],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_accumulation_resume_matches_uninterrupted_cpu_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resumed = root / "resumed"
            whole = root / "whole"
            arguments = ["--smoke-test", "--epochs", "2", "--accumulation-steps", "3",
                         "--checkpoint-every", "1", "--benchmark-warmup", "1"]
            self.launch(*arguments, "--max-steps", "3", "--output", resumed)
            before = torch.load(resumed / "checkpoint.pt", weights_only=True)
            self.assertEqual(before["step"], 3)
            self.assertEqual(before["next_epoch"], 1)
            self.assertEqual(before["next_batch"], 3)
            self.assertEqual(before["microbatches_seen"], 7)
            self.assertEqual(before["lr_schedule_step"], 3)
            self.assertEqual(before["wd_schedule_step"], 3)
            self.launch(*arguments, "--max-steps", "4", "--output", resumed,
                        "--resume", resumed / "checkpoint.pt")
            self.launch(*arguments, "--max-steps", "4", "--output", whole)
            checkpoint = torch.load(resumed / "checkpoint.pt", weights_only=True)
            reference = torch.load(whole / "checkpoint.pt", weights_only=True)
            self.assertEqual(checkpoint["step"], 4)
            self.assertEqual(checkpoint["next_epoch"], 2)
            self.assertEqual(checkpoint["next_batch"], 0)
            for component in ("encoder", "predictor", "target_encoder"):
                for name, value in checkpoint[component].items():
                    torch.testing.assert_close(value, reference[component][name], rtol=0, atol=0)
            self.assertTrue(all(state["step"].item() == 4 for state in checkpoint["optimizer"]["state"].values()))
            with (resumed / "metrics.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["step"]) for row in rows], [1, 2, 3, 4])
            self.assertEqual([int(row["images"]) for row in rows], [6, 2, 6, 2])
            self.assertEqual(rows[0]["steady_images_per_second"], "")
            self.assertGreater(float(rows[1]["steady_images_per_second"]), 0)
            self.assertEqual(float(rows[-1]["weight_decay"]), 0.4)
            for row in rows:
                self.assertTrue(math.isfinite(float(row["loss"])))
                self.assertGreater(float(row["grad_norm"]), 0)

    def test_resume_rejects_changed_batch_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            experiment = root / "experiment.json"
            settings = dict(smoke_test=True, max_steps=1, output=str(output), batch_size=2)
            experiment.write_text(json.dumps(settings), encoding="utf-8")
            self.launch("--config", experiment)
            settings["batch_size"] = 4
            experiment.write_text(json.dumps(settings), encoding="utf-8")
            result = self.launch("--config", experiment,
                                 "--resume", output / "checkpoint.pt", success=False)
            self.assertIn("resume configuration differs", result.stderr)
            self.assertIn("batch_size", result.stderr)
            saved = torch.load(output / "checkpoint.pt", weights_only=True)
            self.assertEqual(saved["config"]["batch_size"], 2)

    def test_time_budget_stops_and_saves_restartable_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            result = self.launch("--smoke-test", "--max-seconds", "0.000001", "--output", output)
            self.assertIn("max-seconds reached", result.stdout)
            checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
            self.assertEqual(checkpoint["step"], 0)
            self.assertEqual(checkpoint["next_epoch"], 0)
            self.assertEqual(checkpoint["next_batch"], 0)
            self.launch("--smoke-test", "--max-steps", "1", "--output", output,
                        "--resume", output / "checkpoint.pt")
            self.assertEqual(torch.load(output / "checkpoint.pt", weights_only=True)["step"], 1)

    def test_class_subset_and_cpu_bfloat16(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for class_name in ("ant", "bird", "cat"):
                folder = root / "images" / class_name
                folder.mkdir(parents=True)
                for index in range(2):
                    shutil.copyfile(ROOT / "example/lenna.png", folder / f"{index}.png")
            output = root / "run"
            self.launch("--data", root / "images", "--class-subset-count", "2", "--precision", "bf16",
                        "--max-steps", "1", "--output", output)
            config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["images"], 4)
            expected_indices = torch.randperm(3, generator=torch.Generator().manual_seed(0))[:2].tolist()
            expected = sorted(("ant", "bird", "cat")[index] for index in expected_indices)
            self.assertEqual(config["selected_classes"], expected)
            self.assertEqual(config["precision"], "bf16")
            checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
            self.assertEqual(checkpoint["encoder"]["patch_embed.proj.weight"].dtype, torch.float32)

    def test_joint_checkpoint_and_resume_use_one_encoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "joint"
            options = ("--smoke-test", "--mode", "joint", "--output", output)
            self.launch(*options, "--max-steps", "1")
            first = torch.load(output / "checkpoint.pt", weights_only=True)
            self.assertEqual(first["config"]["mode"], "joint")
            self.assertEqual(first["config"]["regularizer"], "sigreg")
            self.assertNotIn("target_encoder", first)
            self.assertIn("encoder", first)
            with (output / "metrics.csv").open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["momentum"], "")
            self.assertGreater(float(row["regularization_loss"]), 0)

            self.launch(*options, "--max-steps", "2", "--resume", output / "checkpoint.pt")
            resumed = torch.load(output / "checkpoint.pt", weights_only=True)
            self.assertEqual(resumed["step"], 2)
            self.assertNotIn("target_encoder", resumed)

    def test_joint_visreg_bfloat16_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "joint-visreg"
            self.launch("--smoke-test", "--mode", "joint", "--regularizer", "visreg",
                        "--precision", "bf16", "--max-steps", "1", "--output", output)
            with (output / "metrics.csv").open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertTrue(math.isfinite(float(row["loss"])))
            self.assertGreater(float(row["regularization_loss"]), 0)

    def test_reference_collator_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "reference"
            self.launch("--smoke-test", "--collator", "reference", "--max-steps", "1",
                        "--output", output)
            checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
            self.assertEqual(checkpoint["config"]["collator"], "reference")

    def test_resume_accepts_previous_ema_checkpoint_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "ema"
            self.launch("--smoke-test", "--max-steps", "1", "--output", output)
            checkpoint_path = output / "checkpoint.pt"
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            config_path = output / "config.json"
            saved_config = json.loads(config_path.read_text(encoding="utf-8"))
            for config in (checkpoint["config"], saved_config):
                for key in ("collator", "mode", "regularizer", "regularizer_weight"):
                    del config[key]
                masks = config["masks"]
                masks["M"] = masks.pop("npred")
                del masks["nenc"]
                del masks["allow_overlap"]
            torch.save(checkpoint, checkpoint_path)
            config_path.write_text(json.dumps(saved_config), encoding="utf-8")

            self.launch("--smoke-test", "--max-steps", "2", "--output", output,
                        "--resume", checkpoint_path)
            resumed = torch.load(checkpoint_path, weights_only=True)
            self.assertEqual(resumed["step"], 2)


if __name__ == "__main__":
    unittest.main()
