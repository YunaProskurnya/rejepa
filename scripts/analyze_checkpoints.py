import argparse
import csv
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.encoder import VisionTransformer
from src.representation_diagnostics import evaluate_representation


def main():
    parser = argparse.ArgumentParser(description="Analyze saved I-JEPA target encoder checkpoints")
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

    first = torch.load(checkpoints[0][1], map_location="cpu", weights_only=True)
    model_options = first["config"]["encoder"]
    del first
    image_size = model_options.get("image_size", 224)
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = datasets.ImageFolder(args.data, transform=transform)
    if len(dataset) < 2:
        parser.error("data must contain at least two images")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[
        :args.samples
    ].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                        shuffle=False, pin_memory=device.type == "cuda")

    output = args.output or args.checkpoints / "representation_analysis"
    settings = dict(checkpoints=str(args.checkpoints.resolve()), data=str(args.data.resolve()),
                    paths=[str(Path(dataset.samples[index][0]).resolve()) for index in indices],
                    encoder=model_options, device=str(device), bf16=args.bf16,
                    batch_size=args.batch_size, every=args.every,
                    pooling="mean of full-image patch tokens after encoder final LayerNorm",
                    covariance="centered sample covariance of unnormalized pooled features",
                    singular_entropy="Shannon entropy of normalized singular values of centered features")
    settings_path = output / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != settings:
        parser.error("output contains analysis with different settings; choose another --output")
    output.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    csv_path = output / "representations.csv"
    fields = ["epoch", "step", "images", "feature_dimension", "mean_variance",
              "singular_entropy", "effective_rank", "artifact"]
    completed = set()
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as stream:
            completed = {int(row["epoch"]) for row in csv.DictReader(stream)
                         if (output / row["artifact"]).exists()}

    encoder = VisionTransformer(**model_options).to(device).eval()
    with csv_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if stream.tell() == 0:
            writer.writeheader()
        for epoch, path in checkpoints:
            if epoch in completed:
                continue
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint["config"]["encoder"] != model_options:
                raise ValueError(f"encoder architecture differs in {path}")
            encoder.load_state_dict(checkpoint["target_encoder"], strict=True)
            statistics = evaluate_representation(encoder, loader, device, args.bf16)
            artifact = f"spectrum_{epoch}.pt"
            torch.save(statistics, output / artifact)
            row = {key: statistics[key] for key in fields if key in statistics}
            row.update(epoch=epoch, step=checkpoint.get("step"), artifact=artifact)
            writer.writerow(row)
            stream.flush()
            print(f"epoch {epoch + 1}: rank={statistics['effective_rank']:.2f} "
                  f"entropy={statistics['singular_entropy']:.3f} "
                  f"variance={statistics['mean_variance']:.6f}", flush=True)
            del checkpoint
    print(f"Saved analysis to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
