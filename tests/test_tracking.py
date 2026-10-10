import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings
from zipfile import ZipFile

from src.tracking import start_tracker


class TestTracking(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.experiment = Mock(disabled=False, url="https://example.test/experiment")
        self.sdk = Mock()
        self.sdk.ExperimentConfig.side_effect = lambda **options: options

        def start(**options):
            self.experiment.get_key.return_value = options["experiment_key"]
            return self.experiment

        self.sdk.start.side_effect = start
        self.import_sdk = patch("src.tracking.import_module", return_value=self.sdk).start()
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {"COMET_API_KEY": "test-only-placeholder"}).start()

    def start(self, config=None, **options):
        return start_tracker(self.output, config or {}, enabled=True, **options)

    def metadata(self):
        return json.loads((self.output / "comet.json").read_text(encoding="utf-8"))

    def test_disabled_does_not_import_read_metadata_or_write_files(self):
        with patch("src.tracking.Path", side_effect=AssertionError("should not access paths")):
            with start_tracker(self.output, {}, enabled=False) as tracker:
                tracker.log_metrics({"loss": float("nan")}, prefix="train", step=1, epoch=0)
                tracker.log_asset(self.output / "missing.csv")
            tracker.end()
        self.import_sdk.assert_not_called()
        self.sdk.start.assert_not_called()
        self.assertEqual(list(self.output.iterdir()), [])
        self.assertIsNone(tracker.experiment_key)
        self.assertIsNone(tracker.url)

    def test_missing_key_fails_before_import_or_start(self):
        with patch.dict(os.environ, {"COMET_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "COMET_API_KEY"):
                self.start()
        self.import_sdk.assert_not_called()

    def test_missing_sdk_has_clear_error(self):
        self.import_sdk.side_effect = ImportError
        with self.assertRaisesRegex(RuntimeError, "requires comet-ml"):
            self.start()

    def test_new_run_disables_auto_logging_and_persists_only_identity(self):
        config = {"batch_size": 8, "evaluation": {"lr": 0.1, "classes": ["a", "b"]},
                  "data": Path("data/train"), "max_steps": None, "empty": {}}
        tracker = self.start(config)
        options = self.sdk.start.call_args.kwargs
        self.assertEqual(options["mode"], "create")
        self.assertEqual(options["workspace"], "superustam")
        self.assertEqual(options["project_name"], "rejepa")
        self.assertTrue(options["online"])
        self.assertEqual(len(options["experiment_key"]), 32)
        self.assertEqual(self.metadata(), {"experiment_key": tracker.experiment_key,
                                          "workspace": "superustam", "project": "rejepa"})
        self.assertEqual(tracker.url, self.experiment.url)
        for option, value in options["experiment_config"].items():
            with self.subTest(option=option):
                self.assertFalse(value)
        self.experiment.log_parameters.assert_called_once_with({
            "batch_size": 8, "evaluation.lr": 0.1, "evaluation.classes": '["a", "b"]',
            "data": str(Path("data/train")), "max_steps": "null", "empty": "{}",
        })
        self.experiment.log_asset.assert_not_called()

    def test_new_run_uses_distinct_key_even_with_existing_metadata_and_env_key(self):
        first = self.start()
        with patch.dict(os.environ, {"COMET_EXPERIMENT_KEY": first.experiment_key}):
            second = self.start()
        self.assertNotEqual(first.experiment_key, second.experiment_key)
        self.assertEqual(self.sdk.start.call_args.kwargs["mode"], "create")
        self.assertEqual(self.metadata()["experiment_key"], second.experiment_key)

    def test_resume_reuses_persisted_key(self):
        first = self.start(workspace="team", project="project")
        first.end()
        resumed = self.start(resume=True, workspace="team", project="project")
        self.assertEqual(resumed.experiment_key, first.experiment_key)
        self.assertEqual(self.sdk.start.call_args.kwargs["mode"], "get")

    def test_resume_without_metadata_warns_and_creates(self):
        with self.assertWarnsRegex(RuntimeWarning, "No comet.json"):
            self.start(resume=True)
        self.assertEqual(self.sdk.start.call_args.kwargs["mode"], "create")

    def test_resume_target_mismatch_fails_before_start(self):
        self.start()
        self.sdk.start.reset_mock()
        for options in ({"workspace": "other"}, {"project": "other"}):
            with self.subTest(options=options):
                with self.assertRaisesRegex(RuntimeError, "workspace/project differs"):
                    self.start(resume=True, **options)
        self.sdk.start.assert_not_called()

    def test_invalid_resume_metadata_is_not_overwritten(self):
        path = self.output / "comet.json"
        for contents in ("not json", "[]", json.dumps({"workspace": "superustam",
                         "project": "rejepa", "experiment_key": "invalid"})):
            with self.subTest(contents=contents):
                path.write_text(contents, encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    self.start(resume=True)
                self.assertEqual(path.read_text(encoding="utf-8"), contents)
        self.sdk.start.assert_not_called()

    def test_namespaced_metrics_keep_step_and_epoch_and_skip_invalid_values(self):
        tracker = self.start()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tracker.log_metrics({"loss": 0.25, "count": 3, "missing": None,
                                 "nan": float("nan"), "inf": float("inf"),
                                 "negative_inf": -float("inf"), "text": "3", "array": [1, 2]},
                                prefix="train/", step=7, epoch=2)
        self.assertEqual(len(caught), 6)
        self.experiment.log_metrics.assert_called_once_with(
            {"train/loss": 0.25, "train/count": 3.0}, step=7, epoch=2)

    def test_empty_or_all_invalid_metrics_are_not_sent(self):
        tracker = self.start()
        tracker.log_metrics({}, prefix="monitor", step=1, epoch=1)
        with self.assertWarns(RuntimeWarning):
            tracker.log_metrics({"loss": None}, prefix="monitor", step=1, epoch=1)
        self.experiment.log_metrics.assert_not_called()

    def test_assets_are_only_uploaded_when_explicitly_requested(self):
        tracker = self.start()
        path = self.output / "summary.json"
        tracker.log_asset(path, name="summary.json")
        self.experiment.log_asset.assert_called_once_with(
            str(path), file_name="summary.json", overwrite=False)

    def test_logging_failures_warn_without_exposing_sdk_error_details(self):
        tracker = self.start()
        self.experiment.log_metrics.side_effect = RuntimeError("private-request-detail")
        self.experiment.log_asset.side_effect = OSError("private-request-detail")
        self.experiment.end.side_effect = RuntimeError("private-request-detail")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tracker.log_metrics({"loss": 1}, prefix="train", step=1, epoch=0)
            tracker.log_asset("summary.json")
            tracker.end()
            tracker.end()
        self.assertEqual(len(caught), 3)
        for warning in caught:
            self.assertIn("remote data may be incomplete", str(warning.message))
            self.assertNotIn("private-request-detail", str(warning.message))
        self.experiment.end.assert_called_once()

    def test_config_logging_failure_keeps_tracker_usable(self):
        self.experiment.log_parameters.side_effect = OSError("offline")
        with self.assertWarnsRegex(RuntimeWarning, "configuration logging failed"):
            tracker = self.start({"batch_size": 8})
        tracker.log_metrics({"loss": 1}, prefix="train", step=1, epoch=0)
        self.experiment.log_metrics.assert_called_once()

    def test_context_manager_ends_on_training_error_without_swallowing_it(self):
        with self.assertRaisesRegex(ValueError, "training failed"):
            with self.start():
                raise ValueError("training failed")
        self.experiment.end.assert_called_once()

    def test_startup_error_fails_without_metadata_or_secret_details(self):
        self.sdk.start.side_effect = RuntimeError("private-request-detail")
        with self.assertRaisesRegex(RuntimeError, "Comet startup failed") as caught:
            self.start()
        self.assertNotIn("private-request-detail", str(caught.exception))
        self.assertFalse((self.output / "comet.json").exists())

    def test_metadata_write_failure_closes_experiment(self):
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(RuntimeError, "output directory"):
                self.start()
        self.experiment.end.assert_called_once()

    def test_inactive_experiment_is_rejected(self):
        self.experiment.disabled = True
        with self.assertRaisesRegex(RuntimeError, "Comet startup failed"):
            self.start()
        self.experiment.end.assert_called_once()
        self.assertFalse((self.output / "comet.json").exists())


class TestCometSdkOffline(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("comet_ml"), "optional comet-ml SDK not installed")
    def test_real_sdk_records_metrics_and_asset_in_offline_archive(self):
        # A fresh process exercises the first SDK import independently of suite
        # order, SDK global state, and any previously initialized telemetry.
        result = subprocess.run(
            [sys.executable, "-c", "from tests.test_tracking import TestCometSdkOffline; "
             "TestCometSdkOffline()._check_offline_sdk()"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def _check_offline_sdk(self):
        # Exercise the production wrapper and real SDK with network access blocked.
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(os.environ, {"COMET_API_KEY": "test-only-placeholder"}), \
                patch("socket.socket.connect", side_effect=AssertionError("network forbidden")) as connect:
            import numpy as np
            import torch

            output = Path(temporary)
            archives = output / "offline"

            def import_offline_sdk(name):
                comet = importlib.import_module(name)

                def offline_start(**options):
                    inspect.signature(comet.start).bind(**options)
                    self.assertTrue(options["online"])
                    options["online"] = False
                    options["experiment_config"].offline_directory = str(archives)
                    return comet.start(**options)

                return SimpleNamespace(start=offline_start, ExperimentConfig=comet.ExperimentConfig)

            asset = output / "summary.json"
            asset_content = b'{"top1": 75.0}\n'
            asset.write_bytes(asset_content)
            random.seed(0)  # This seed enables Comet 3.49's randomized Sentry path.
            python_state, numpy_state = random.getstate(), np.random.get_state()
            torch_state = torch.get_rng_state().clone()

            def assert_rng_unchanged():
                self.assertEqual(random.getstate(), python_state)
                self.assertEqual(np.random.get_state()[0], numpy_state[0])
                np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
                self.assertEqual(np.random.get_state()[2:], numpy_state[2:])
                self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

            with patch("src.tracking.import_module", side_effect=import_offline_sdk):
                with start_tracker(output, {"evaluation": {"batch_size": 8}}, enabled=True) as tracker:
                    assert_rng_unchanged()
                    tracker.log_metrics({"loss": 0.25}, prefix="train", step=7, epoch=2)
                    assert_rng_unchanged()
                    tracker.log_asset(asset, name="summary.json")
            assert_rng_unchanged()

            archive_paths = list(archives.glob("*.zip"))
            self.assertEqual(len(archive_paths), 1)
            with ZipFile(archive_paths[0]) as archive:
                messages_path = next(name for name in archive.namelist() if name.endswith("messages.json"))
                messages = [json.loads(line) for line in archive.read(messages_path).splitlines()]
                metrics = [message["payload"]["metric"] for message in messages
                           if message["type"] == "metric_msg"]
                self.assertEqual(metrics, [{"metricName": "train/loss", "metricValue": 0.25,
                                            "step": 7, "epoch": 2}])
                self.assertTrue(any(archive.read(name) == asset_content for name in archive.namelist()))
                self.assertIn("evaluation.batch_size", json.dumps(messages))
            connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
