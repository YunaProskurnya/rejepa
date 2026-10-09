"""Read flat JSON experiment options before applying command-line overrides."""

import argparse
import json
import math
from pathlib import Path


def _config_value(parser, action, value):
    """Validate JSON types explicitly; argparse only converts CLI strings."""
    name = action.dest
    if value is None:
        if action.default is None:
            return None
        parser.error(f"config option '{name}' does not accept null")

    value_type = action.type or str
    if isinstance(action, argparse.BooleanOptionalAction):
        valid = type(value) is bool
        expected = "a boolean"
    elif value_type is int:
        valid = type(value) is int
        expected = "an integer"
    elif value_type is float:
        valid = type(value) in (int, float)
        expected = "a finite number"
        if valid:
            try:
                valid = math.isfinite(value)
            except OverflowError:
                valid = False
    else:
        valid = isinstance(value, str)
        expected = "a string"
    if not valid:
        parser.error(f"config option '{name}' must be {expected}")

    if not isinstance(action, argparse.BooleanOptionalAction):
        value = value_type(value)
    if action.choices is not None and value not in action.choices:
        parser.error(
            f"invalid config option '{name}': {value!r}; "
            f"choose from {', '.join(map(str, action.choices))}"
        )
    return value


def parse_config_args(parser, argv=None):
    """Apply CLI > JSON > parser defaults; paths are relative to the current directory.

    The caller defines a ``--config`` Path option and its normal scalar options.
    JSON keys use argument destinations (``batch_size``, not ``--batch-size``).
    """
    bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=parser.allow_abbrev)
    bootstrap.add_argument("--config", type=Path)
    config_path = bootstrap.parse_known_args(argv)[0].config
    if config_path is not None:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            parser.error(f"cannot read config '{config_path}': {error}")
        if not isinstance(config, dict):
            parser.error("config must contain a JSON object")

        actions = {
            action.dest: action for action in parser._actions
            if action.option_strings and action.dest not in ("config", "help")
        }
        unknown = sorted(set(config) - set(actions))
        if unknown:
            parser.error(f"unknown config option(s): {', '.join(unknown)}")
        # Validate every supplied value, including values later overridden by CLI.
        values = {name: _config_value(parser, actions[name], value)
                  for name, value in config.items()}
        parser.set_defaults(**values)
    return parser.parse_args(argv)
