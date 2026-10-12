import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.encoder import VisionTransformer
from src.representation_diagnostics import evaluate_representation, make_representation_loader, save_representation


def main():
    parser = argparse.ArgumentParser(description="Analyze saved I-JEPA encoder checkpoints")
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="ImageFolder directory, preferably validation images")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--every", type=int, default=1, help="Analyze first, last, and every Nth completed epoch")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.samples < 2 or args.batch_size < 1 or args.every < 1:
        parser.error("samples must be at least 2; batch-size and every must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        parser.error("the selected GPU does not support bfloat16")
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))

    checkpoints = []
    for path in args.checkpoints.glob("checkpoint_*.pt"):
        suffix = path.stem.removeprefix("checkpoint_")
        if suffix.isdigit():
            checkpoints.append((int(suffix), path))
    checkpoints.sort()
    if not checkpoints:
        parser.error("no checkpoint_N.pt files found")
    checkpoints = [item for item in checkpoints if item == checkpoints[0] or item == checkpoints[-1]
                   or (item[0] + 1) % args.every == 0]

    first_config = torch.load(checkpoints[0][1], map_location="cpu", weights_only=True)["config"]
    model_options = first_config["encoder"]
    mode = first_config.get("mode", "ema")
    source = "encoder" if mode == "joint" else "target_encoder"
    paths, loader = make_representation_loader(args.data, model_options["image_size"], args.samples,
                                               args.seed, args.batch_size, device)
    if len(paths) < 2:
        parser.error("data must contain at least two images")

    output = args.output or args.checkpoints / "checkpoint_analysis"
    settings = dict(checkpoints=str(args.checkpoints.resolve()), paths=paths,
                    encoder_source=source, bf16=args.bf16, every=args.every)
    if mode == "joint":
        settings.update(regularizer=first_config["regularizer"],
                        regularizer_weight=first_config["regularizer_weight"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (output / "representations.csv").write_text("", encoding="utf-8")

    encoder = VisionTransformer(**model_options).to(device).eval()
    for epoch, path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        if (config["encoder"] != model_options or config.get("mode", "ema") != mode or
                mode == "joint" and (config.get("regularizer") != first_config.get("regularizer") or
                                     config.get("regularizer_weight") != first_config.get("regularizer_weight"))):
            raise ValueError(f"training configuration differs in {path}")
        encoder.load_state_dict(checkpoint[source])
        statistics = evaluate_representation(encoder, loader, device, args.bf16)
        save_representation(output, epoch, checkpoint.get("step"), statistics)
        print(f"epoch {epoch + 1}: rank={statistics['effective_rank']:.2f} "
              f"entropy={statistics['singular_entropy']:.3f} "
              f"variance={statistics['mean_variance']:.6f}", flush=True)
    print(f"Saved analysis to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
