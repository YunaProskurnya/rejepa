"""Compare already measured lightweight probe curves; no GPU required."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.linear_probe_plot import plot_linear_curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, nargs='+', required=True)
    parser.add_argument('--labels', nargs='+')
    parser.add_argument('--output', type=Path, required=True, help='plot filename stem, without extension')
    args = parser.parse_args()
    if args.labels is not None and len(args.labels) != len(args.csv):
        parser.error('provide one label per CSV')
    for path in plot_linear_curves(args.csv, args.output, args.labels):
        print(f'Saved {path.resolve()}')


if __name__ == '__main__':
    main()
