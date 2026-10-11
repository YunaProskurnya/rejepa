import argparse
from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from src.config import parse_config_args


class TestExperimentConfig(unittest.TestCase):
    def make_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=Path)
        parser.add_argument("--data", type=Path)
        parser.add_argument("--smoke-test", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--batch-size", type=int, default=2)
        parser.add_argument("--max-steps", type=int)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--collator", choices=("custom", "reference"), default="custom")
        return parser

    def parse_config(self, config, *argv):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "experiment.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            return parse_config_args(self.make_parser(), ["--config", str(path), *argv])

    def assert_config_error(self, config, message, *argv):
        stderr = StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
            self.parse_config(config, *argv)
        self.assertEqual(caught.exception.code, 2)
        self.assertIn(message, stderr.getvalue())

    def test_defaults_work_without_a_file(self):
        args = parse_config_args(self.make_parser(), [])
        self.assertEqual(args.batch_size, 2)
        self.assertEqual(args.collator, "custom")
        self.assertIsNone(args.config)

    def test_json_types_and_paths(self):
        args = self.parse_config({"data": "data/train", "batch_size": 8, "lr": 1,
                                  "smoke_test": True, "device": "cuda:0"})
        self.assertEqual(args.data, Path("data/train"))
        self.assertFalse(args.data.is_absolute())
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.lr, 1.0)
        self.assertIsInstance(args.lr, float)
        self.assertTrue(args.smoke_test)
        self.assertEqual(args.device, "cuda:0")
        self.assertEqual(args.collator, "custom")

    def test_cli_overrides_json_including_false(self):
        args = self.parse_config({"batch_size": 8, "smoke_test": True,
                                  "collator": "reference"},
                                 "--batch-size", "4", "--no-smoke-test", "--collator", "custom")
        self.assertEqual(args.batch_size, 4)
        self.assertFalse(args.smoke_test)
        self.assertEqual(args.collator, "custom")

    def test_nullable_options(self):
        args = self.parse_config({"data": None, "max_steps": None})
        self.assertIsNone(args.data)
        self.assertIsNone(args.max_steps)
        self.assert_config_error({"batch_size": None}, "does not accept null")
        self.assert_config_error({"smoke_test": None}, "does not accept null")

    def test_invalid_choices_are_checked_before_cli_overrides(self):
        self.assert_config_error({"collator": "typo"}, "invalid config option 'collator'",
                                 "--collator", "custom")

    def test_unknown_keys_and_recursive_configs_are_rejected(self):
        for key in ("batch_szie", "batch-size", "config", "help"):
            with self.subTest(key=key):
                self.assert_config_error({key: "anything"}, f"unknown config option(s): {key}")

    def test_wrong_json_types_are_rejected(self):
        for key, value in (("batch_size", True), ("batch_size", 2.0), ("batch_size", "2"),
                           ("lr", False), ("lr", "0.1"), ("smoke_test", "false"),
                           ("smoke_test", 1), ("data", 123), ("device", {}),
                           ("batch_size", [])):
            with self.subTest(key=key, value=value):
                self.assert_config_error({key: value}, f"config option '{key}' must be")

    def test_nonfinite_numbers_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf"), 10 ** 400):
            with self.subTest(value=value):
                self.assert_config_error({"lr": value}, "must be a finite number")

    def test_top_level_must_be_an_object(self):
        for config in ([], "hello", 3, None):
            with self.subTest(config=config):
                self.assert_config_error(config, "config must contain a JSON object")

    def test_malformed_or_missing_file_has_clear_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "experiment.json"
            for contents in (None, '{"batch_size":'):
                if contents is not None:
                    path.write_text(contents, encoding="utf-8")
                stderr = StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
                    parse_config_args(self.make_parser(), ["--config", str(path)])
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("cannot read config", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
