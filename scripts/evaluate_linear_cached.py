"""Fit the full LARS head grid from disk features; never load or run an encoder."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import parse_config_args
from src.linear_probe_cache import FeatureCache, prepare_feature_cache
from src.linear_probe_final import evaluate_final_linear_probe, make_final_probe_loaders
from src.linear_probe_report import summary_paths, write_probe_summary
from src.tracking import start_tracker


def parse_args(argv=None, *, prepare=False):
    parser = argparse.ArgumentParser(description=__doc__ if not prepare else 'Prepare epoch-wise FP32 feature cache')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--cache', type=Path)
    # Shared JSON: these paths are only used by prepare_linear_cache.py.
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--train', type=Path)
    parser.add_argument('--val', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128, help='encoder extraction batch (prepare only)')
    parser.add_argument('--probe-batch-size', type=int, default=16384)
    parser.add_argument('--num-workers', type=int, default=4, help='image loading workers (prepare only)')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16', help='encoder precision (prepare only)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--comet', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--comet-workspace', default='superustam')
    parser.add_argument('--comet-project', default='rejepa')
    args = parse_config_args(parser, argv)
    required = ('cache', 'checkpoint', 'train', 'val') if prepare else ('cache', 'output')
    for name in required:
        if getattr(args, name) is None:
            parser.error(f'{name} is required in JSON or CLI')
    if args.epochs < 1 or args.batch_size < 1 or args.probe_batch_size < 2 or args.num_workers < 0:
        parser.error('positive epochs/batch-size, head batch >= 2 and workers >= 0 required')
    if prepare and args.cache.exists():
        parser.error('cache directory already exists; choose a new cache path')
    if not prepare and any(path.exists() for path in [args.output, *summary_paths(args.output)]):
        parser.error('result file already exists; choose a new output path')
    return parser, args


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None, *, prepare=False):
    parser, args = parse_args(argv, prepare=prepare)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            parser.error('CUDA is unavailable')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)
        if prepare and args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
            parser.error('this GPU does not support bf16')
    else:
        torch.set_num_threads(min(4, torch.get_num_threads()))
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    if prepare:
        # Only preparation imports/constructs the model or opens the image dataset.
        from src.models.encoder import VisionTransformer
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        options = checkpoint['config']['encoder']
        source = 'target_encoder' if checkpoint.get('target_encoder') is not None else 'encoder'
        encoder = VisionTransformer(**options).to(device)
        encoder.load_state_dict(checkpoint[source])
        loaders = make_final_probe_loaders(
            args.train, args.val, class_names=checkpoint['config'].get('selected_classes'),
            image_size=options.get('image_size', 224), batch_size=args.batch_size,
            num_workers=args.num_workers, seed=args.seed, pin_memory=device.type == 'cuda')
        provenance = dict(config=config, checkpoint_sha256=_sha256(args.checkpoint),
                          encoder_source=source, encoder=options,
                          checkpoint_mode=checkpoint['config'].get('mode', 'ema'),
                          checkpoint_regularizer=checkpoint['config'].get('regularizer', 'none'),
                          checkpoint_regularizer_weight=checkpoint['config'].get('regularizer_weight'),
                          checkpoint_step=checkpoint.get('step'), checkpoint_epoch=checkpoint.get('epoch'),
                          class_to_idx=loaders[0].dataset.class_to_idx)
        # Keep paths and class labels in extraction order for dataset provenance.
        # No crop images are copied, and evaluation never opens these paths.
        provenance['dataset_samples'] = {
            split: [(str(Path(path).resolve()), int(label)) for path, label in loader.dataset.samples]
            for split, loader in (('train', loaders[0]), ('val', loaders[2]))}
        dimension = options['embed_dim'] * (4 if options['depth'] >= 4 else 1)
        estimate = (len(loaders[0].dataset) * (args.epochs + 1) + len(loaders[2].dataset)) * (4 * dimension + 8)
        print(f'Preparing {args.epochs} crop epochs plus train/val scoring; estimated {estimate / 1e9:.2f} GB.', flush=True)
        metadata = prepare_feature_cache(
            encoder, *loaders, len(loaders[0].dataset.classes), device, args.cache,
            epochs=args.epochs, seed=args.seed, precision=args.precision, provenance=provenance,
            progress=lambda row: print(f"Cached {row['split']}: {row['images']} images, {row['seconds']:.1f}s", flush=True))
        print(f"Cache ready: {args.cache.resolve()} ({metadata['preparation_seconds']:.1f}s)", flush=True)
        return

    cache = FeatureCache(args.cache)
    # The cache, not optional preparation paths in the JSON, identifies the encoder.
    # Keep per-image provenance on disk instead of duplicating it in every report/Comet.
    metadata = dict(cache.metadata)
    metadata['provenance'] = {key: value for key, value in metadata['provenance'].items()
                              if key != 'dataset_samples'}
    cache.metadata = metadata
    if args.epochs > metadata['epochs']:
        parser.error('epochs exceeds available cached crop epochs')
    config = {key: value for key, value in config.items()
              if key not in ('checkpoint', 'train', 'val', 'precision', 'batch_size', 'num_workers')}
    config['cache_metadata'] = metadata
    with ExitStack() as resources:
        tracker = start_tracker(args.output.parent, {'evaluation_final_cached': config},
                                enabled=args.comet, workspace=args.comet_workspace,
                                project=args.comet_project,
                                resume=(args.output.parent / 'comet.json').exists())
        resources.callback(tracker.end)

        def progress(row):
            print(f"Cached linear probe: epoch={row['epoch']}/{row['epochs']} "
                  f"head_steps={row['steps']} seconds={row['seconds']:.1f}", flush=True)
            tracker.log_metrics(dict(head_steps=row['steps'], seconds=row['seconds']),
                                prefix='linear_final_progress/', step=row['epoch'], epoch=row['epoch'])

        metrics = evaluate_final_linear_probe(
            None, None, None, None, metadata['num_classes'], device,
            epochs=args.epochs, batch_size=args.probe_batch_size, seed=args.seed,
            feature_cache=cache, progress=progress)
        config['protocol'] = metrics.pop('protocol')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(config=config, metrics=metrics), indent=2), encoding='utf-8')
        summaries = write_probe_summary(args.output, metrics, dict(
            cache=str(args.cache.resolve()), checkpoint_sha256=metadata['provenance'].get('checkpoint_sha256'),
            epochs=args.epochs, batch_size=args.probe_batch_size, seed=args.seed))
        step = metadata['provenance'].get('checkpoint_step') or 0
        epoch = metadata['provenance'].get('checkpoint_epoch') or 0
        tracker.log_metrics({key: value for key, value in metrics['best'].items() if isinstance(value, (int, float))},
                            prefix='linear_final/', step=step, epoch=epoch)
        tracker.log_metrics({'probe_seconds': metrics['probe_seconds']}, prefix='linear_final/', step=step, epoch=epoch)
        for index, trial in enumerate(metrics['trials']):
            tracker.log_metrics({key: value for key, value in trial.items() if isinstance(value, (int, float))},
                                prefix=f'linear_final/trial_{index}/', step=step, epoch=epoch)
        for file in (args.output, *summaries):
            tracker.log_asset(file)
        print(summaries[1].read_text(encoding='utf-8'), flush=True)
        print(f'Results: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
