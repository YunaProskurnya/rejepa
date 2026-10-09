import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import PIL
import torch
import torchvision
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.encoder import VisionTransformer
from src.models.predictor import VisionTransformerPredictor
from src.models.target_encoder import create_target_encoder
from src.config import parse_config_args
from src.datasets.imagenet100 import MaskCollator, make_data_loader
from src.optimizer import create_optimizer
from src.reference.schedulers import CosineWDSchedule, WarmupCosineSchedule
from src.train_step import train_step


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Single-device I-JEPA training")
    parser.add_argument("--config", type=Path, help="JSON experiment settings; CLI options take precedence")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--smoke-test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dataset", choices=("imagefolder", "cifar10"), default="imagefolder")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "first-run")
    parser.add_argument("--model", choices=("tiny", "vit_base", "vit_large"), default="tiny")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--benchmark-warmup", type=int, default=5)
    parser.add_argument("--class-subset-count", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--start-lr", type=float, default=2e-4)
    parser.add_argument("--final-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--final-weight-decay", type=float, default=0.4)
    parser.add_argument("--ema-start", type=float, default=0.996)
    parser.add_argument("--ema-end", type=float, default=1.0)
    args = parse_config_args(parser, argv)
    if (args.data is not None) == args.smoke_test:
        parser.error("choose exactly one of --data or --smoke-test (in JSON or on the command line)")
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        parser.error("epochs/batch-size must be positive; num-workers must be nonnegative")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("max-steps must be positive")
    if args.accumulation_steps <= 0 or args.checkpoint_every <= 0 or args.benchmark_warmup < 0:
        parser.error("accumulation-steps/checkpoint-every must be positive; benchmark-warmup nonnegative")
    if args.max_seconds is not None and not 0 < args.max_seconds < float("inf"):
        parser.error("max-seconds must be finite and positive")
    if args.class_subset_count is not None and (
        args.class_subset_count <= 0 or args.smoke_test or args.dataset != "imagefolder"
    ):
        parser.error("class-subset-count requires ImageFolder and a positive count")
    if not 0 <= args.warmup_epochs < args.epochs:
        parser.error("warmup-epochs must be nonnegative and less than epochs")
    if not 0 <= args.ema_start <= args.ema_end <= 1:
        parser.error("EMA must satisfy 0 <= ema-start <= ema-end <= 1")
    if any(not 0 <= value < float("inf") for value in (
        args.lr, args.start_lr, args.final_lr, args.weight_decay, args.final_weight_decay,
    )):
        parser.error("learning rates and weight decay must be finite and nonnegative")

    return parser, args


def main():
    parser, args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable in this environment")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            parser.error("this CUDA device does not support bfloat16")
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))

    crop_options = dict(size=224, scale=(0.3, 1.0), ratio=(0.75, 4 / 3),
                        interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
    normalize_options = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    transform = transforms.Compose([
        transforms.RandomResizedCrop(**crop_options),
        transforms.ToTensor(),
        transforms.Normalize(**normalize_options),
    ])
    if args.smoke_test:
        dataset = datasets.FakeData(size=8, image_size=(3, 256, 256), transform=transform)
    elif args.dataset == "cifar10":
        dataset = datasets.CIFAR10(str(args.data), train=True, transform=transform, download=False)
    else:
        dataset = datasets.ImageFolder(str(args.data), transform=transform)

    selected_classes = None
    if isinstance(dataset, datasets.ImageFolder):
        selected_classes = dataset.classes
        if args.class_subset_count is not None:
            if args.class_subset_count > len(dataset.classes):
                parser.error("class-subset-count exceeds the number of ImageFolder classes")
            selected = torch.randperm(
                len(dataset.classes), generator=torch.Generator().manual_seed(args.seed),
            )[:args.class_subset_count].tolist()
            selected_classes = sorted(dataset.classes[index] for index in selected)
            mapping = {dataset.class_to_idx[name]: index for index, name in enumerate(selected_classes)}
            dataset.samples = [(path, mapping[label]) for path, label in dataset.samples if label in mapping]
            dataset.imgs = dataset.samples
            dataset.targets = [label for _, label in dataset.samples]
            dataset.classes = selected_classes
            dataset.class_to_idx = {name: index for index, name in enumerate(selected_classes)}

    mask_options = dict(
        input_size=224, patch_size=16, enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2), aspect_ratio=(0.75, 1.5),
        M=4, min_keep=10,
    )
    collator = MaskCollator(**mask_options)
    batches_per_epoch = len(dataset) // args.batch_size
    if batches_per_epoch == 0:
        parser.error("dataset must contain at least batch-size images")
    updates_per_epoch = math.ceil(batches_per_epoch / args.accumulation_steps)

    embed_dim, depth, num_heads, predictor_dim, predictor_depth = {
        "tiny": (48, 2, 3, 24, 2),
        "vit_base": (768, 12, 12, 384, 6),
        "vit_large": (1024, 24, 16, 384, 12),
    }[args.model]
    transformer_options = dict(mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0)
    model_options = dict(image_size=224, patch_size=16, in_channels=3, embed_dim=embed_dim,
                         depth=depth, num_heads=num_heads, **transformer_options)
    encoder = VisionTransformer(**model_options).to(device)
    predictor_options = dict(num_patches=encoder.num_patches, embed_dim=embed_dim,
                             predictor_embed_dim=predictor_dim,
                             depth=predictor_depth, num_heads=num_heads, **transformer_options)
    predictor = VisionTransformerPredictor(**predictor_options).to(device)
    target_encoder = create_target_encoder(encoder)
    optimizer = create_optimizer(encoder, predictor, lr=args.lr, weight_decay=args.weight_decay)
    encoder.train()
    predictor.train()
    target_encoder.train()

    total_steps = args.epochs * updates_per_epoch
    lr_schedule = WarmupCosineSchedule(
        optimizer, warmup_steps=args.warmup_epochs * updates_per_epoch,
        start_lr=args.start_lr, ref_lr=args.lr, final_lr=args.final_lr, T_max=total_steps,
    )
    wd_schedule = CosineWDSchedule(
        optimizer, ref_wd=args.weight_decay, final_wd=args.final_weight_decay, T_max=total_steps,
    )
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(encoder=model_options, predictor=predictor_options, masks=mask_options,
                  total_steps=total_steps, torch_version=str(torch.__version__),
                  resolved_device=str(device), cuda_version=torch.version.cuda,
                  gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  gpu_memory_bytes=torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
                  images=len(dataset), selected_classes=selected_classes,
                  batches_per_epoch=batches_per_epoch, updates_per_epoch=updates_per_epoch,
                  effective_batch_size=args.batch_size * args.accumulation_steps,
                  reference_commit="52c1ae95d05f743e000e8f10a1f3a79b10cff048")
    # Record fixed settings as well as editable flags: checkpoints are self-describing.
    config.update(
        preprocessing=dict(
            random_resized_crop={**crop_options, "interpolation": crop_options["interpolation"].value},
            to_tensor=True, normalize=normalize_options,
        ),
        optimizer=dict(name="AdamW", defaults=optimizer.defaults,
                       no_decay="parameter name contains 'bias' or parameter.ndim == 1"),
        objective=dict(loss="smooth_l1", beta=1.0, reduction="mean per block, then mean across blocks",
                       target_normalization="layer_norm over features", target_norm_eps=1e-5,
                       target_update="EMA after optimizer.step", gradient_clipping=None),
        schedules=dict(lr="warmup_cosine", weight_decay="cosine", ema="linear",
                       unit="optimizer update", warmup_steps=args.warmup_epochs * updates_per_epoch,
                       total_steps=total_steps),
        architecture=dict(position_embedding="fixed_2d_sincos", layer_norm_eps=1e-6,
                          initialization_std=0.02, residual_rescale="1 / sqrt(2 * layer_index)",
                          attention="scaled_dot_product_attention", stochastic_depth=0.0),
        data_loader=dict(drop_last=True, persistent_workers=False,
                         sampler="randperm with seed + epoch", pin_memory=device.type == "cuda"),
        smoke_data=dict(size=8, image_size=(3, 256, 256)) if args.smoke_test else None,
        versions=dict(python=sys.version.split()[0], torchvision=str(torchvision.__version__),
                      numpy=np.__version__, pillow=PIL.__version__),
    )
    for key in ("data", "output", "resume", "config"):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value.resolve())
    config = json.loads(json.dumps(config))

    step = 0
    next_epoch = 0
    next_batch = 0
    microbatches_seen = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        compatible = (
            "encoder", "predictor", "masks", "model", "dataset", "data", "smoke_test",
            "images", "selected_classes", "batch_size", "accumulation_steps", "epochs", "seed",
            "precision", "lr", "start_lr", "final_lr", "warmup_epochs", "weight_decay",
            "final_weight_decay", "ema_start", "ema_end", "total_steps",
            "preprocessing", "optimizer", "objective", "schedules", "architecture", "smoke_data",
        )
        differences = [key for key in compatible if checkpoint["config"].get(key) != config.get(key)]
        if differences:
            parser.error("resume configuration differs: " + ", ".join(differences))
        if "next_batch" not in checkpoint:
            parser.error("checkpoint lacks resume progress; use a checkpoint from this launcher version")
        encoder.load_state_dict(checkpoint["encoder"])
        predictor.load_state_dict(checkpoint["predictor"])
        target_encoder.load_state_dict(checkpoint["target_encoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        next_epoch = checkpoint["next_epoch"]
        next_batch = checkpoint["next_batch"]
        microbatches_seen = checkpoint["microbatches_seen"]
        lr_schedule._step = checkpoint["lr_schedule_step"]
        wd_schedule._step = checkpoint["wd_schedule_step"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        numpy_state = checkpoint["numpy_rng_state"]
        np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
        if device.type == "cuda" and checkpoint["cuda_rng_states"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        collator._itr_counter.value = microbatches_seen - 1
        print(f"Resuming after optimizer update {step}; worker prefetch/masks are not bitwise replayed.", flush=True)
        del checkpoint

    if args.output.exists() and not args.resume:
        parser.error("output already exists; choose a new directory or pass --resume")
    if args.output.exists() and args.resume and (args.output / "config.json").exists():
        previous_config = json.loads((args.output / "config.json").read_text(encoding="utf-8"))
        if any(previous_config.get(key) != config.get(key) for key in compatible):
            parser.error("existing output directory belongs to an incompatible run")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if args.smoke_test:
        print("Synthetic images: checks execution only, not representation quality.", flush=True)
    print(f"model={args.model} device={device} precision={args.precision} images={len(dataset)} "
          f"classes={len(selected_classes) if selected_classes is not None else 'n/a'} "
          f"updates_per_epoch={updates_per_epoch} effective_batch={config['effective_batch_size']}", flush=True)

    def save_checkpoint():
        numpy_state = np.random.get_state()
        temporary = args.output / "checkpoint.tmp"
        torch.save({
            "encoder": encoder.state_dict(), "predictor": predictor.state_dict(),
            "target_encoder": target_encoder.state_dict(), "optimizer": optimizer.state_dict(),
            "step": step, "epoch": next_epoch + int(next_batch > 0), "config": config,
            "next_epoch": next_epoch, "next_batch": next_batch, "microbatches_seen": microbatches_seen,
            "lr_schedule_step": lr_schedule._step, "wd_schedule_step": wd_schedule._step,
            "torch_rng_state": torch.get_rng_state(), "python_rng_state": random.getstate(),
            "numpy_rng_state": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }, temporary)
        temporary.replace(args.output / "checkpoint.pt")

    fieldnames = [
        "step", "epoch", "loss", "grad_norm", "lr", "weight_decay", "momentum", "context_tokens", "target_tokens",
        "images", "step_seconds", "images_per_second", "steady_images_per_second", "eta_seconds",
        "cuda_peak_memory_mb", "elapsed_seconds", "warmup",
    ]
    log_path = args.output / "metrics.csv"
    previous_rows = []
    if args.resume and log_path.exists():
        with log_path.open(newline="", encoding="utf-8") as previous:
            previous_rows = [row for row in csv.DictReader(previous) if int(row["step"]) <= step]
    session_start = time.perf_counter()
    measured_images = 0
    measured_seconds = 0.0
    session_updates = 0
    stop_reason = "epochs completed"
    stop = args.max_steps is not None and step >= args.max_steps
    if stop:
        stop_reason = "max-steps already reached"
    with log_path.open("w", newline="", encoding="utf-8") as log:
        writer = csv.DictWriter(log, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(previous_rows)
        for epoch in range(next_epoch, args.epochs):
            if stop:
                break
            permutation = torch.randperm(
                len(dataset), generator=torch.Generator().manual_seed(args.seed + epoch),
            ).tolist()
            first_batch = next_batch if epoch == next_epoch else 0
            loader = make_data_loader(
                dataset, batch_size=args.batch_size, sampler=permutation[first_batch * args.batch_size:],
                drop_last=True, num_workers=args.num_workers, collator=collator,
                generator=torch.Generator().manual_seed(args.seed + epoch), worker_init_fn=seed_worker,
                pin_mem=device.type == "cuda",
            )
            iterator = iter(loader)
            batch_index = first_batch
            while batch_index < batches_per_epoch:
                if args.max_seconds is not None and time.perf_counter() - session_start >= args.max_seconds:
                    stop = True
                    stop_reason = "max-seconds reached at an optimizer-update boundary"
                    break
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                group_size = min(args.accumulation_steps, batches_per_epoch - batch_index)
                lr = lr_schedule.step()
                weight_decay = wd_schedule.step()
                momentum = args.ema_start + (args.ema_end - args.ema_start) * step / total_steps
                loss_sum = 0.0
                context_sum = 0.0
                target_sum = 0.0
                for microstep in range(group_size):
                    batch, context_masks, target_masks = next(iterator)
                    images = batch[0].to(device, non_blocking=True)
                    context_indices = context_masks[0].to(device, non_blocking=True)
                    target_indices = [indices.to(device, non_blocking=True) for indices in target_masks]
                    metrics = train_step(
                        encoder, predictor, target_encoder, optimizer,
                        images, context_indices, target_indices, momentum,
                        precision=args.precision, accumulation_steps=group_size,
                        zero_grad=microstep == 0, update=microstep == group_size - 1,
                    )
                    loss_sum += metrics["loss"]
                    context_sum += context_indices.shape[1]
                    target_sum += target_indices[0].shape[1]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                seconds = time.perf_counter() - started
                step += 1
                session_updates += 1
                batch_index += group_size
                microbatches_seen += group_size
                next_epoch = epoch + int(batch_index == batches_per_epoch)
                next_batch = 0 if batch_index == batches_per_epoch else batch_index
                update_images = group_size * args.batch_size
                warmup = session_updates <= args.benchmark_warmup
                if not warmup:
                    measured_images += update_images
                    measured_seconds += seconds
                throughput = measured_images / measured_seconds if measured_seconds else None
                remaining_images = (
                    (args.epochs - next_epoch) * batches_per_epoch - next_batch
                ) * args.batch_size
                row = dict(
                    step=step, epoch=epoch + 1, loss=loss_sum / group_size, grad_norm=metrics["grad_norm"],
                    lr=lr, weight_decay=weight_decay, momentum=momentum,
                    context_tokens=context_sum / group_size, target_tokens=target_sum / group_size,
                    images=update_images, step_seconds=seconds, images_per_second=update_images / seconds,
                    steady_images_per_second=throughput, eta_seconds=remaining_images / throughput if throughput else None,
                    cuda_peak_memory_mb=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0,
                    elapsed_seconds=time.perf_counter() - session_start, warmup=warmup,
                )
                if group_size == 1:
                    row["context_tokens"] = int(row["context_tokens"])
                    row["target_tokens"] = int(row["target_tokens"])
                writer.writerow(row)
                log.flush()
                print(f"step={step} epoch={epoch + 1} loss={row['loss']:.6f} "
                      f"context={row['context_tokens']:g} target={row['target_tokens']:g} "
                      f"lr={lr:.6g} ema={momentum:.6f} seconds={seconds:.3f} "
                      f"images/s={row['images_per_second']:.1f} warmup={warmup} "
                      f"cuda_mb={row['cuda_peak_memory_mb']:.0f}", flush=True)
                if step % args.checkpoint_every == 0:
                    save_checkpoint()
                if args.max_steps is not None and step >= args.max_steps:
                    stop = True
                    stop_reason = "max-steps reached"
                    break
            save_checkpoint()
            del iterator
        if session_updates == 0:
            save_checkpoint()
    print(f"Stopped: {stop_reason}", flush=True)
    if measured_seconds:
        print(f"Steady throughput excluding {args.benchmark_warmup} warmup updates and checkpoint writes: "
              f"{measured_images / measured_seconds:.2f} images/s", flush=True)
    print(f"Saved configuration, metrics and checkpoint to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
