"""Presentation plots from lightweight probe CSVs, without loading models."""

import csv
from pathlib import Path


def plot_linear_curves(csv_paths, output, labels=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    csv_paths = [Path(path) for path in csv_paths]
    labels = list(labels) if labels is not None else [path.parent.name for path in csv_paths]
    if len(labels) != len(csv_paths):
        raise ValueError('one label is required for each CSV')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    try:
        for index, (path, label) in enumerate(zip(csv_paths, labels)):
            with path.open(newline='', encoding='utf-8') as stream:
                rows = sorted(csv.DictReader(stream), key=lambda row: int(row['epoch']))
            if not rows:
                raise ValueError(f'empty curve: {path}')
            epochs = [int(row['epoch']) for row in rows]
            color = f'C{index % 10}'
            for axis, metric in zip(axes, ('top1', 'loss')):
                for split, style in (('val', '-'), ('train', '--')):
                    axis.plot(epochs, [float(row[f'{split}_{metric}']) for row in rows],
                              linestyle=style, color=color, label=f'{label}: {split}', linewidth=1.8)
        axes[0].set(ylabel='Top-1 accuracy (%)', ylim=(0, 100))
        axes[1].set(ylabel='Cross-entropy loss')
        for axis in axes:
            axis.set_xlabel('Pretraining epoch (completed)')
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        fig.suptitle('Lightweight linear probe (fixed features, Adam)')
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        paths = [output.with_suffix('.png'), output.with_suffix('.svg')]
        for path in paths:
            fig.savefig(path, dpi=180)
        return paths
    finally:
        plt.close(fig)
