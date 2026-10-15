"""Evaluate saved epoch checkpoints with a fixed lightweight probe and resume CSV."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import parse_config_args
from src.linear_probe import evaluate_linear_probe, make_probe_loaders
from src.linear_probe_plot import plot_linear_curves
from src.models.encoder import VisionTransformer


FIELDS = ('epoch', 'step', 'checkpoint', 'checkpoint_size', 'checkpoint_mtime_ns',
          'encoder_source', 'train_top1', 'val_top1', 'train_loss', 'val_loss',
          'train_images', 'val_images', 'probe_seconds', 'point_seconds')


def discover_checkpoints(directory, every=1):
    found = []
    for path in Path(directory).glob('checkpoint_*.pt'):
        match = re.fullmatch(r'checkpoint_(epoch_)?(\d+)\.pt', path.name)
        if match:
            # main: checkpoint_0.pt is completed epoch 1;
            # experiment-monitoring: checkpoint_epoch_0010.pt is completed epoch 10.
            epoch = int(match[2]) + (0 if match[1] else 1)
            found.append((epoch, path.resolve()))
    found.sort(key=lambda item: (item[0], item[1].name))
    if not found:
        raise ValueError('no checkpoint_N.pt or checkpoint_epoch_N.pt files found')
    if len({epoch for epoch, _ in found}) != len(found):
        raise ValueError('multiple checkpoint files describe the same epoch; use one run directory')
    return [item for item in found if item in (found[0], found[-1]) or item[0] % every == 0]


def select_samples(dataset, per_class, seed):
    """One fixed, reproducible subset; never sample again for a new checkpoint."""
    if per_class == 0:
        return
    generator = torch.Generator().manual_seed(seed)
    chosen = []
    for label in range(len(dataset.classes)):
        indices = [index for index, (_, target) in enumerate(dataset.samples) if target == label]
        order = torch.randperm(len(indices), generator=generator).tolist()[:per_class]
        chosen.extend(indices[index] for index in order)
    dataset.samples = [dataset.samples[index] for index in sorted(chosen)]
    dataset.imgs = dataset.samples
    dataset.targets = [label for _, label in dataset.samples]


def run_signature(checkpoint):
    config = checkpoint['config']
    source = 'target_encoder' if checkpoint.get('target_encoder') is not None else 'encoder'
    return dict(encoder=config['encoder'], selected_classes=config.get('selected_classes'),
                mode=config.get('mode', 'ema'), regularizer=config.get('regularizer', 'none'),
                regularizer_weight=config.get('regularizer_weight'), encoder_source=source)


def read_rows(path):
    if not path.exists():
        return {}
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError('existing curve CSV has an incompatible header')
        rows = list(reader)
    if len({row['checkpoint'] for row in rows}) != len(rows):
        raise ValueError('duplicate checkpoints in existing CSV')
    if len({int(row['epoch']) for row in rows}) != len(rows):
        raise ValueError('duplicate epochs in existing CSV')
    for row in rows:
        for key in ('train_top1', 'val_top1', 'train_loss', 'val_loss', 'probe_seconds', 'point_seconds'):
            if not math.isfinite(float(row[key])):
                raise ValueError('non-finite metrics in existing CSV')
        checkpoint = Path(row['checkpoint'])
        if not checkpoint.exists():
            raise ValueError(f'previously evaluated checkpoint is missing: {checkpoint}')
        stat = checkpoint.stat()
        if (stat.st_size, stat.st_mtime_ns) != (int(row['checkpoint_size']), int(row['checkpoint_mtime_ns'])):
            raise ValueError(f'previously evaluated checkpoint changed: {checkpoint}; choose a new output')
    return {row['checkpoint']: row for row in rows}


def write_rows(path, rows):
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda row: int(row['epoch'])))
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--checkpoints', type=Path)
    parser.add_argument('--train', type=Path)
    parser.add_argument('--val', type=Path)
    parser.add_argument('--output', type=Path, help='directory for metrics, settings and plots')
    parser.add_argument('--every', type=int, default=1, help='first, last and each Nth completed epoch')
    parser.add_argument('--train-per-class', type=int, default=100, help='0: all train images')
    parser.add_argument('--val-per-class', type=int, default=0, help='0: all validation images')
    parser.add_argument('--probe-epochs', type=int, default=30)
    parser.add_argument('--probe-batch-size', type=int, default=256)
    parser.add_argument('--probe-lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=128, help='encoder feature extraction batch')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32')
    parser.add_argument('--seed', type=int, default=0)
    args = parse_config_args(parser, argv)
    for name in ('checkpoints', 'train', 'val'):
        if getattr(args, name) is None:
            parser.error(f'{name} is required in JSON or CLI')
    if min(args.every, args.probe_epochs, args.probe_batch_size, args.batch_size) < 1:
        parser.error('every, epochs and batch sizes must be positive')
    if min(args.train_per_class, args.val_per_class, args.num_workers) < 0:
        parser.error('sample limits and workers must be nonnegative')
    if not math.isfinite(args.probe_lr) or args.probe_lr <= 0:
        parser.error('probe-lr must be positive and finite')
    args.output = args.output or args.checkpoints / 'linear_curve'
    return parser, args


def main(argv=None):
    parser, args = parse_args(argv)
    # Fail before expensive encoder passes if plotting dependencies are missing.
    import matplotlib
    matplotlib.use('Agg')
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            parser.error('CUDA is unavailable')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)
        if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
            parser.error('this CUDA device does not support bf16')
    else:
        torch.set_num_threads(min(4, torch.get_num_threads()))
    checkpoints = discover_checkpoints(args.checkpoints, args.every)
    first = torch.load(checkpoints[0][1], map_location='cpu', weights_only=True)
    signature = run_signature(first)
    del first
    loaders = make_probe_loaders(
        args.train, args.val, class_names=signature['selected_classes'],
        image_size=signature['encoder'].get('image_size', 224), batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed, pin_memory=device.type == 'cuda')
    for loader, limit, seed in zip(loaders, (args.train_per_class, args.val_per_class), (args.seed, args.seed + 1)):
        select_samples(loader.dataset, limit, seed)
    selection = {}
    for split, loader in zip(('train', 'val'), loaders):
        selection[split] = []
        for filename, label in loader.dataset.samples:
            path = Path(filename).resolve()
            stat = path.stat()
            selection[split].append([str(path), label, stat.st_size, stat.st_mtime_ns])
    selection_hash = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
    settings = dict(
        version=1, checkpoints=str(args.checkpoints.resolve()), train=str(args.train.resolve()),
        val=str(args.val.resolve()), signature=signature, selection_sha256=selection_hash,
        class_to_idx=loaders[0].dataset.class_to_idx,
        train_images=len(loaders[0].dataset), val_images=len(loaders[1].dataset),
        train_per_class=args.train_per_class, val_per_class=args.val_per_class,
        protocol=dict(name='lightweight_linear_curve', head='fresh Linear with bias', optimizer='Adam',
                      epochs=args.probe_epochs, lr=args.probe_lr, batch_size=args.probe_batch_size,
                      weight_decay=0, pooling='mean of last-layer patch tokens', seed=args.seed,
                      precision=args.precision, transform=repr(loaders[0].dataset.transform)),
        extraction_batch_size=args.batch_size, torch_version=str(torch.__version__),
        torchvision_version=str(torchvision.__version__),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    settings_path, csv_path = args.output / 'settings.json', args.output / 'linear_curve.csv'
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding='utf-8')) != settings:
            raise ValueError('curve settings/data differ; use a new output directory')
    elif csv_path.exists():
        raise ValueError('existing CSV lacks settings.json; use a new output directory')
    else:
        settings_path.write_text(json.dumps(settings, indent=2), encoding='utf-8')
        (args.output / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    del selection
    rows = read_rows(csv_path)
    pending = sum(str(path) not in rows for _, path in checkpoints)
    print(f'Checkpoints={len(checkpoints)}, pending={pending}; '
          f"train={settings['train_images']}, val={settings['val_images']}; "
          'lightweight probe, not final LARS evaluation.', flush=True)
    encoder = VisionTransformer(**signature['encoder']).to(device)
    elapsed_points = []
    try:
        for inferred_epoch, path in checkpoints:
            if str(path) in rows:
                print(f'Skip already evaluated: {path.name}', flush=True)
                continue
            started = time.perf_counter()
            stat = path.stat()
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            if run_signature(checkpoint) != signature:
                raise ValueError(f'encoder/training mode/classes differ in {path}; use one run per curve')
            epoch = int(checkpoint.get('next_epoch', checkpoint.get('epoch', inferred_epoch)))
            if epoch < 1 or checkpoint.get('next_batch', 0) != 0:
                raise ValueError(f'checkpoint is not a completed training epoch: {path}')
            if any(int(row['epoch']) == epoch for row in rows.values()):
                raise ValueError(f'duplicate completed epoch {epoch}: {path}')
            step = checkpoint.get('step', '')
            encoder.load_state_dict(checkpoint[signature['encoder_source']])
            del checkpoint  # Release optimizer/RNG tensors before feature extraction.
            metrics = evaluate_linear_probe(
                encoder, *loaders, len(loaders[0].dataset.classes), device,
                epochs=args.probe_epochs, lr=args.probe_lr, batch_size=args.probe_batch_size,
                seed=args.seed, precision=args.precision)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise ValueError(f'checkpoint changed while evaluating: {path}')
            seconds = time.perf_counter() - started
            rows[str(path)] = dict(epoch=epoch, step=step, checkpoint=str(path),
                                  checkpoint_size=stat.st_size, checkpoint_mtime_ns=stat.st_mtime_ns,
                                  encoder_source=signature['encoder_source'], point_seconds=seconds, **metrics)
            write_rows(csv_path, rows)
            elapsed_points.append(seconds)
            pending -= 1
            eta = pending * sum(elapsed_points[-5:]) / len(elapsed_points[-5:])
            print(f"epoch={epoch} step={step} train_top1={metrics['train_top1']:.2f}% "
                  f"val_top1={metrics['val_top1']:.2f}% seconds={seconds:.1f} eta={eta / 60:.1f}min", flush=True)
    finally:
        if rows:
            plot_linear_curves([csv_path], args.output / 'linear_curve', [args.checkpoints.name])
    print(f'Saved curve to {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
