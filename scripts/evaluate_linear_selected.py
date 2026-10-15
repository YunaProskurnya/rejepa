"""Evaluate a checkpoint with one fixed recipe from a config or earlier final JSON.

Uses the same final LARS evaluation as evaluate_linear.py, without a head sweep.
Run with --help for checkpoint, dataset, output and Comet options.
"""

from evaluate_linear import main


if __name__ == "__main__":
    main(selected_only=True)
