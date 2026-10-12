"""Exercise the real launchers with an offline, subprocess-local Comet SDK."""

import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
FAKE_KEY = "integration-test-placeholder"
FAKE_SDK = r'''
import json
import os
from pathlib import Path


def record(kind, **values):
    with Path(os.environ["FAKE_COMET_EVENTS"]).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(kind=kind, **values)) + "\n")


class ExperimentConfig:
    def __init__(self, **options):
        self.options = options


class Experiment:
    def __init__(self, key):
        self.key = key
        self.disabled = os.environ.get("FAKE_COMET_DISABLED") == "1"
        self.url = "https://example.test/experiment/" + key

    def get_key(self):
        return self.key

    def log_parameters(self, parameters):
        record("parameters", key=self.key, values=parameters)

    def log_metrics(self, metrics, *, step, epoch):
        record("metrics", key=self.key, values=metrics, step=step, epoch=epoch)

    def log_asset(self, path, *, file_name, overwrite):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        record("asset", key=self.key, path=str(path), file_name=file_name,
               overwrite=overwrite)

    def end(self):
        record("end", key=self.key)

    def log_model(self, name, file_or_folder, *, overwrite, metadata):
        import torch
        path = Path(file_or_folder)
        weights = torch.load(path, map_location="cpu", weights_only=True)
        record("model", key=self.key, name=name, path=str(path), overwrite=overwrite,
               metadata=metadata, components=sorted(weights), step=weights["step"],
               completed=weights["completed"])


def start(*, api_key, experiment_config, **options):
    # Validate only the supplied test credential; never record credentials.
    if api_key != "integration-test-placeholder":
        raise RuntimeError("unexpected test credential")
    record("start", config=experiment_config.options, **options)
    return Experiment(options["experiment_key"])
'''


class TestCometIntegration(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.events_path = self.root / "comet-events.jsonl"
        sdk = self.root / "fake-sdk"
        sdk.mkdir()
        (sdk / "comet_ml.py").write_text(textwrap.dedent(FAKE_SDK), encoding="utf-8")
        # Inherit ordinary process settings without reading any Comet credentials.
        self.env = {name: os.environ[name] for name in os.environ
                    if not name.upper().startswith("COMET_")}
        self.env.update(COMET_API_KEY=FAKE_KEY,
                        FAKE_COMET_EVENTS=str(self.events_path),
                        PYTHONPATH=os.pathsep.join((str(sdk), str(ROOT))))

    def launch(self, *arguments, expected=0):
        result = subprocess.run(
            [sys.executable, *map(str, arguments)], cwd=ROOT, env=self.env,
            capture_output=True, text=True, timeout=120,
        )
        self.assertNotIn(FAKE_KEY, result.stdout + result.stderr)
        if expected == 0:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def events(self):
        if not self.events_path.exists():
            return []
        return [json.loads(line) for line in self.events_path.read_text(encoding="utf-8").splitlines()]

    def make_images(self):
        generator = np.random.default_rng(12)
        for split in ("train", "val"):
            for name in ("ant", "bird"):
                directory = self.root / split / name
                directory.mkdir(parents=True)
                for index in range(2):
                    pixels = generator.integers(0, 256, (240, 260, 3), dtype=np.uint8)
                    Image.fromarray(pixels).save(directory / f"{index}.png")

    def test_training_resume_and_final_evaluation_share_experiment(self):
        self.make_images()
        plain, tracked = self.root / "plain", self.root / "tracked"
        base = ["scripts/train.py", "--data", self.root / "train", "--epochs", "1",
                "--batch-size", "2", "--seed", "17", "--linear-probe",
                "--val-data", self.root / "val", "--probe-epochs", "1",
                "--probe-batch-size", "2", "--checkpoint-every-epochs", "1"]
        comet = ["--comet", "--comet-workspace", "test-team", "--comet-project", "test-project"]
        self.launch(*base, "--output", plain, "--no-comet")
        self.assertEqual(self.events(), [])
        self.assertFalse((plain / "comet.json").exists())
        self.assertFalse((plain / "final_weights.pt").exists())

        self.launch(*base, "--output", tracked, *comet, "--max-steps", "1")
        identity = json.loads((tracked / "comet.json").read_text(encoding="utf-8"))
        self.assertEqual(identity["workspace"], "test-team")
        self.assertEqual(identity["project"], "test-project")
        self.assertFalse((tracked / "linear_probe.csv").exists())
        self.assertEqual([event["kind"] for event in self.events()].count("end"), 1)
        self.launch(*base, "--output", tracked, *comet, "--resume", tracked / "checkpoint.pt")
        self.assertEqual(json.loads((tracked / "comet.json").read_text(encoding="utf-8")), identity)

        expected = torch.load(plain / "checkpoint.pt", map_location="cpu", weights_only=True)
        actual = torch.load(tracked / "checkpoint.pt", map_location="cpu", weights_only=True)
        self.assertEqual(actual["step"], 2)
        self.assertEqual((tracked / "checkpoint_epoch_0001.pt").read_bytes(),
                         (tracked / "checkpoint.pt").read_bytes())
        self.assertEqual(actual["config"]["checkpoint_every_epochs"], 1)
        for component in ("encoder", "predictor", "target_encoder"):
            for name, weights in expected[component].items():
                torch.testing.assert_close(weights, actual[component][name], rtol=0, atol=0)

        events = self.events()
        starts = [event for event in events if event["kind"] == "start"]
        self.assertEqual([event["mode"] for event in starts], ["create", "get"])
        self.assertEqual({event["experiment_key"] for event in starts}, {identity["experiment_key"]})
        training = [event for event in events
                    if event["kind"] == "metrics" and "train/loss" in event["values"]]
        self.assertEqual([(event["step"], event["epoch"]) for event in training], [(1, 1), (2, 1)])
        with (tracked / "metrics.csv").open(newline="", encoding="utf-8") as stream:
            training_rows = list(csv.DictReader(stream))
        self.assertEqual(len(training_rows), 2)
        for event, row in zip(training, training_rows):
            self.assertEqual(event["values"]["train/loss"], float(row["loss"]))
            self.assertTrue(all(key.startswith("train/") for key in event["values"]))

        with (tracked / "linear_probe.csv").open(newline="", encoding="utf-8") as stream:
            probe_rows = list(csv.DictReader(stream))
        probes = [event for event in events
                  if event["kind"] == "metrics" and "linear_probe/val_top1" in event["values"]]
        self.assertEqual(len(probe_rows), 1)
        self.assertEqual(len(probes), 1)
        self.assertEqual((probes[0]["step"], probes[0]["epoch"]), (2, 1))
        for name in ("train_loss", "val_loss", "train_top1", "val_top1",
                     "train_images", "val_images", "probe_seconds"):
            self.assertEqual(probes[0]["values"]["linear_probe/" + name], float(probe_rows[0][name]))

        output = tracked / "linear_final.json"
        self.launch("scripts/evaluate_linear.py", "--mode", "final",
                    "--checkpoint", tracked / "checkpoint.pt", "--train", self.root / "train",
                    "--val", self.root / "val", "--output", output, "--probe-epochs", "1",
                    "--probe-batch-size", "2", "--batch-size", "2", "--seed", "17", *comet)
        final_result = json.loads(output.read_text(encoding="utf-8"))
        trials = final_result["metrics"]["trials"]
        self.assertEqual(len(trials), 12)  # Tiny encoder has no last-four representation.
        events = self.events()
        starts = [event for event in events if event["kind"] == "start"]
        self.assertEqual([event["mode"] for event in starts], ["create", "get", "get"])
        self.assertEqual({event["experiment_key"] for event in starts}, {identity["experiment_key"]})
        self.assertEqual([event["kind"] for event in events].count("end"), 3)
        final_metrics = [event for event in events
                         if event["kind"] == "metrics" and "linear_final/val_top1" in event["values"]]
        self.assertEqual(len(final_metrics), 1)
        self.assertEqual((final_metrics[0]["step"], final_metrics[0]["epoch"]), (2, 1))
        self.assertEqual(final_metrics[0]["values"]["linear_final/val_top1"],
                         final_result["metrics"]["best"]["val_top1"])
        for index, trial in enumerate(trials):
            name = f"linear_final/trial_{index}/val_top1"
            logged = [event for event in events if event["kind"] == "metrics" and name in event["values"]]
            self.assertEqual(len(logged), 1)
            self.assertEqual(logged[0]["values"][name], trial["val_top1"])
            self.assertEqual((logged[0]["step"], logged[0]["epoch"]), (2, 1))

        assets = {Path(event["path"]).name for event in events if event["kind"] == "asset"}
        self.assertTrue({"config.json", "metrics.csv", "linear_probe.csv", "linear_final.json"} <= assets)
        self.assertFalse(any(name.endswith(".pt") for name in assets))
        for event in events:
            if event["kind"] == "metrics":
                self.assertTrue(all(math.isfinite(value) for value in event["values"].values()))
            if event["kind"] == "parameters":
                self.assertNotIn(FAKE_KEY, json.dumps(event["values"]))
        self.assertNotIn(FAKE_KEY, (tracked / "config.json").read_text(encoding="utf-8"))
        self.assertNotIn(FAKE_KEY, (tracked / "comet.json").read_text(encoding="utf-8"))
        models = [event for event in events if event["kind"] == "model"]
        self.assertEqual([event["step"] for event in models], [1, 2])
        self.assertEqual([event["completed"] for event in models], [False, True])
        self.assertTrue(all(event["name"] == "rejepa" for event in models))
        self.assertTrue(all(event["key"] == identity["experiment_key"] for event in models))
        self.assertTrue(all(not event["overwrite"] for event in models))
        for event in models:
            self.assertEqual(event["step"], event["metadata"]["step"])
            self.assertEqual(event["completed"], event["metadata"]["completed"])
            self.assertIn("target_encoder", event["components"])
            self.assertNotIn("optimizer", event["components"])
        weights = torch.load(tracked / "final_weights.pt", weights_only=True)
        self.assertEqual(weights["config"], actual["config"])
        for component in ("encoder", "predictor", "target_encoder"):
            for name, tensor in actual[component].items():
                torch.testing.assert_close(weights[component][name], tensor, rtol=0, atol=0)
                self.assertEqual(weights[component][name].device.type, "cpu")

        selected_output = tracked / "linear_selected.json"
        selected_config = self.root / "probe_config.json"
        recipe = {name: final_result['metrics']['best'][name]
                  for name in ('representation', 'head', 'reference_lr', 'weight_decay')}
        selected_config.write_text(json.dumps(dict(
            checkpoint=str(tracked / 'final_weights.pt'), train=str(self.root / 'train'),
            val=str(self.root / 'val'), output=str(selected_output),
            probe_epochs=1, probe_batch_size=2, batch_size=2, seed=17, **recipe,
        )))
        self.launch("scripts/evaluate_linear_selected.py", "--config", selected_config, *comet)
        selected = json.loads(selected_output.read_text())
        self.assertEqual(len(selected["metrics"]["trials"]), 1)
        for name in ("representation", "head", "reference_lr", "weight_decay"):
            self.assertEqual(selected["metrics"]["best"][name], final_result["metrics"]["best"][name])
        self.assertEqual(selected["config"]["protocol"]["epochs"], 1)
        self.assertEqual(selected["config"]["protocol"]["effective_batch_size"], 2)
        self.assertEqual(selected["metrics"]["best"]["val_images"], 4)
        self.assertEqual([event for event in self.events() if event["kind"] == "start"][-1]["mode"], "get")
        self.assertTrue(any(event["kind"] == "asset" and event["path"] == str(selected_output)
                            for event in self.events()))

    def test_inactive_sdk_fails_before_training_and_closes_experiment(self):
        self.env["FAKE_COMET_DISABLED"] = "1"
        output = self.root / "failed"
        result = self.launch("scripts/train.py", "--smoke-test", "--max-steps", "1",
                             "--output", output, "--comet", expected=1)
        self.assertIn("Comet startup failed", result.stderr)
        self.assertEqual([event["kind"] for event in self.events()], ["start", "end"])
        for name in ("checkpoint.pt", "metrics.csv", "config.json", "comet.json"):
            self.assertFalse((output / name).exists())


if __name__ == "__main__":
    unittest.main()
