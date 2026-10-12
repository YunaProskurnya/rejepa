"""Command to evaluate a saved checkpoint without restarting pretraining.

--mode final (default): 50-epoch LARS evaluation in src/linear_probe_final.py.
--mode monitor: the same cached-feature Adam probe used after training epochs,
implemented in src/linear_probe.py. This file only loads, dispatches and saves.
"""

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.linear_probe import evaluate_linear_probe, make_probe_loaders
from src.linear_probe_final import evaluate_final_linear_probe, make_final_probe_loaders
from src.models.encoder import VisionTransformer
from src.tracking import start_tracker
from src.config import parse_config_args
from src.linear_probe_report import summary_paths, write_probe_summary


def load_selected_recipe(path):
    """Read only the winning recipe, never copy previous scores to a new evaluation."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        protocol = report['config']['protocol']
        best = report['metrics']['best']
        if protocol['name'] != 'ijepa_final_linear':
            raise ValueError('expected an ijepa_final_linear report')
        representation, head = best['representation'], best['head']
        if representation not in ('last', 'last4') or head not in ('linear', 'batchnorm_linear'):
            raise ValueError('unsupported representation or head')
        lr, wd = best['reference_lr'], best['weight_decay']
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in (lr, wd)):
            raise ValueError('reference_lr and weight_decay must be finite numbers')
        if lr <= 0 or wd < 0:
            raise ValueError('reference_lr must be positive; weight_decay nonnegative')
        epochs, batch = protocol['epochs'], protocol['effective_batch_size']
        if type(epochs) is not int or epochs <= 0 or type(batch) is not int or batch < 2:
            raise ValueError('invalid epochs or effective_batch_size')
        return dict(representations=[representation], heads=[head],
                    learning_rates=[lr], weight_decays=[wd], epochs=epochs, batch_size=batch)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f'Cannot load final probe recipe from {path}: {error}') from error


def parse_args(argv=None, *, selected_only=False):
    parser = argparse.ArgumentParser(description="Evaluate a frozen encoder checkpoint on ImageFolder train/val")
    parser.add_argument("--mode", choices=("final",) if selected_only else ("monitor", "final"), default="final",
                        help="monitor: cached features/Adam; final (default): fresh crops/LARS")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--val", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--representation", choices=('last', 'last4'))
    parser.add_argument("--head", choices=('linear', 'batchnorm_linear'))
    parser.add_argument("--reference-lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--best-from", type=Path,
                        help="final only: fit one recipe from metrics.best of a previous final JSON")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=128, help="feature extraction batch size")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--probe-epochs", type=int, help="head epochs: monitor=30, final=50")
    parser.add_argument("--probe-batch-size", type=int,
                        help="head batch: monitor=256, final=16384 (images are encoded in smaller batches)")
    parser.add_argument("--probe-lr", type=float,
                        help="monitor only (default 0.001); final evaluates the documented LR grid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--comet", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--comet-workspace", default="superustam")
    parser.add_argument("--comet-project", default="rejepa")
    args = parse_config_args(parser, argv)
    for name in ('checkpoint', 'train', 'val', 'output'):
        if getattr(args, name) is None:
            parser.error(f'{name} is required in JSON or CLI')
    recipe_values = (args.representation, args.head, args.reference_lr, args.weight_decay)
    explicit_recipe = any(value is not None for value in recipe_values)
    if explicit_recipe:
        if not all(value is not None for value in recipe_values):
            parser.error('provide representation, head, reference_lr and weight_decay together')
        if args.best_from is not None or args.mode != 'final':
            parser.error('explicit recipe requires final mode and cannot be combined with best-from')
        if not 0 < args.reference_lr < float('inf') or not 0 <= args.weight_decay < float('inf'):
            parser.error('reference_lr must be positive and finite; weight_decay nonnegative and finite')
    if selected_only and not explicit_recipe and args.best_from is None:
        parser.error('provide a complete recipe in JSON/CLI or --best-from')
    args.selected_recipe = None
    if explicit_recipe:
        args.selected_recipe = dict(representations=[args.representation], heads=[args.head],
                                    learning_rates=[args.reference_lr], weight_decays=[args.weight_decay])
    if args.best_from is not None:
        if args.mode != 'final':
            parser.error('--best-from requires --mode final')
        try:
            args.selected_recipe = load_selected_recipe(args.best_from)
        except ValueError as error:
            parser.error(str(error))
        if args.probe_epochs is None:
            args.probe_epochs = args.selected_recipe['epochs']
        if args.probe_batch_size is None:
            args.probe_batch_size = args.selected_recipe['batch_size']
    if args.mode == "final" and args.probe_lr is not None:
        parser.error("--probe-lr applies only to --mode monitor; final uses its recorded LR grid")
    if args.probe_epochs is None:
        args.probe_epochs = 30 if args.mode == "monitor" else 50
    if args.probe_batch_size is None:
        args.probe_batch_size = 256 if args.mode == "monitor" else 16384
    if args.mode == "monitor" and args.probe_lr is None:
        args.probe_lr = 1e-3
    if args.output.exists():
        parser.error("output already exists; choose a new JSON path")
    if args.mode == 'final' and any(path.exists() for path in summary_paths(args.output)):
        parser.error('summary already exists; choose a new JSON path')
    if min(args.batch_size, args.probe_epochs, args.probe_batch_size) <= 0 or args.num_workers < 0:
        parser.error("batch sizes/epochs must be positive; num-workers must be nonnegative")
    if args.probe_lr is not None and not 0 < args.probe_lr < float("inf"):
        parser.error("probe-lr must be positive and finite")
    if args.mode == "final" and args.probe_batch_size < 2:
        parser.error("final head batch size must be at least 2 for BatchNorm")
    return parser, args


def run_evaluation(resources, argv=None, *, selected_only=False):
    parser, args = parse_args(argv, selected_only=selected_only)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            parser.error("this CUDA device does not support bfloat16")
    else:
        torch.set_num_threads(min(4, torch.get_num_threads()))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_options = checkpoint["config"]["encoder"]
    source = "target_encoder" if checkpoint.get("target_encoder") is not None else "encoder"
    encoder = VisionTransformer(**model_options).to(device)
    encoder.load_state_dict(checkpoint[source])
    make_loaders = make_probe_loaders if args.mode == "monitor" else make_final_probe_loaders
    loaders = make_loaders(
        args.train, args.val, class_names=checkpoint["config"].get("selected_classes"),
        image_size=model_options.get("image_size", 224), batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed, pin_memory=device.type == "cuda",
    )
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(encoder_source=source, encoder=model_options,
                  class_to_idx=loaders[0].dataset.class_to_idx,
                  checkpoint_step=checkpoint.get("step"), torch_version=str(torch.__version__))
    tracker = start_tracker(
        args.output.parent, {f"evaluation_{args.mode}": config}, enabled=args.comet,
        workspace=args.comet_workspace, project=args.comet_project,
        resume=(args.output.parent / "comet.json").exists(),
    )
    resources.callback(tracker.end)
    options = dict(epochs=args.probe_epochs, batch_size=args.probe_batch_size,
                   seed=args.seed, precision=args.precision)
    if args.selected_recipe is not None:
        options.update({key: value for key, value in args.selected_recipe.items()
                        if key not in ('epochs', 'batch_size')})
        print(f"Selected final probe settings: {options}", flush=True)
    if args.mode == "monitor":
        metrics = evaluate_linear_probe(
            encoder, *loaders, len(loaders[0].dataset.classes), device,
            lr=args.probe_lr, **options,
        )
        protocol = dict(
            mode="monitor", transform="bilinear resize + center crop + ImageNet normalization",
            image_size=model_options.get("image_size", 224),
            resize=int(model_options.get("image_size", 224) * 256 / 224),
            pooling="mean of final patch tokens; no extra normalization",
            cache="CPU float32; one encoder pass per split", head="fresh Linear with bias",
            optimizer="Adam", betas=[0.9, 0.999], eps=1e-8, weight_decay=0,
            accuracy_unit="percent",
        )
    else:
        def report_progress(row):
            print(f"Final linear probe: epoch={row['epoch']}/{row['epochs']} "
                  f"head_steps={row['steps']} seconds={row['seconds']:.1f}", flush=True)
            tracker.log_metrics(dict(head_steps=row['steps'], seconds=row['seconds']),
                                prefix="linear_final_progress/", step=row['epoch'], epoch=row['epoch'])

        metrics = evaluate_final_linear_probe(
            encoder, *loaders, len(loaders[0].dataset.classes), device,
            progress=report_progress, **options,
        )
        protocol = metrics.pop("protocol")
    config["protocol"] = protocol
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(config=config, metrics=metrics), indent=2), encoding="utf-8")
    step, epoch = checkpoint.get("step", 0), checkpoint.get("epoch", 0)
    if args.mode == "final":
        best_scalars = {key: value for key, value in metrics['best'].items()
                        if isinstance(value, (int, float))}
        tracker.log_metrics({**best_scalars, 'probe_seconds': metrics['probe_seconds']},
                            prefix="linear_final/", step=step, epoch=epoch)
        for index, trial in enumerate(metrics['trials']):
            trial_scalars = {key: value for key, value in trial.items() if isinstance(value, (int, float))}
            tracker.log_metrics(trial_scalars, prefix=f"linear_final/trial_{index}/", step=step, epoch=epoch)
    else:
        tracker.log_metrics(metrics, prefix="linear_monitor_checkpoint/", step=step, epoch=epoch)
    tracker.log_asset(args.output)
    if args.mode == 'final':
        for path in write_probe_summary(args.output, metrics, dict(
                checkpoint=str(args.checkpoint.resolve()), epochs=args.probe_epochs,
                batch_size=args.probe_batch_size, seed=args.seed)):
            tracker.log_asset(path)
    print(json.dumps(metrics.get("best", metrics), indent=2), flush=True)
    if args.mode == "final":
        if len(metrics['trials']) == 1:
            print("Saved one fixed head variant; no hyperparameter sweep.", flush=True)
        else:
            print(f"Saved all {len(metrics['trials'])} head variants; best is selected by val top-1.", flush=True)
    print(f"Saved linear probe evaluation to {args.output.resolve()}", flush=True)


def main(argv=None, *, selected_only=False):
    with ExitStack() as resources:
        run_evaluation(resources, argv, selected_only=selected_only)


if __name__ == "__main__":
    main()
