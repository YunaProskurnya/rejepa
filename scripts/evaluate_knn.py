import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.encoder import VisionTransformer


def select_per_class(dataset, class_names, count, generator):
    if count <= 0:
        raise ValueError("images per class must be positive")
    by_class = {dataset.class_to_idx[name]: [] for name in class_names}
    for index, (_, label) in enumerate(dataset.samples):
        if label in by_class:
            by_class[label].append(index)
    selected = []
    for label, indices in by_class.items():
        if len(indices) < count:
            raise ValueError(f"class {dataset.classes[label]} has {len(indices)} images; need {count}")
        order = torch.randperm(len(indices), generator=generator)[:count].tolist()
        selected.extend(indices[position] for position in order)
    return selected


@torch.inference_mode()
def extract_features(encoder, loader, device, bf16):
    features, labels = [], []
    for images, targets in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            patch_features = encoder(images.to(device, non_blocking=True))
            pooled = patch_features.float().mean(dim=1)
        if not torch.isfinite(pooled).all():
            raise RuntimeError("Non-finite image features")
        features.append(pooled.cpu())
        labels.append(targets)
    return torch.cat(features), torch.cat(labels)


@torch.inference_mode()
def weighted_knn(reference, reference_labels, queries, query_labels, k=20,
                 temperature=0.07, chunk_size=128, device="cpu"):
    if k <= 0 or chunk_size <= 0 or not 0 < temperature < float("inf"):
        raise ValueError("k, chunk-size and temperature must be positive and finite")
    if reference.ndim != 2 or queries.ndim != 2 or reference.shape[1] != queries.shape[1]:
        raise ValueError("reference and queries must have shape [images, features]")
    if len(reference) == 0 or len(queries) == 0:
        raise ValueError("reference and query sets must be nonempty")
    if reference_labels.shape != (len(reference),) or query_labels.shape != (len(queries),):
        raise ValueError("one label is required per image")
    reference = F.normalize(reference.float(), dim=1).to(device)
    reference_labels = reference_labels.to(device)
    classes = reference_labels.unique(sorted=True)
    reference_labels = torch.searchsorted(classes, reference_labels)
    num_classes = len(classes)
    k = min(k, len(reference))
    correct1, correct5 = 0, 0
    for start in range(0, len(queries), chunk_size):
        query = F.normalize(queries[start:start + chunk_size].float(), dim=1).to(device)
        similarities, neighbors = (query @ reference.T).topk(k, dim=1)
        weights = ((similarities - similarities[:, :1]) / temperature).exp()
        scores = torch.zeros(len(query), num_classes, device=device)
        scores.scatter_add_(1, reference_labels[neighbors], weights)
        predictions = classes[scores.topk(min(5, num_classes), dim=1).indices]
        expected = query_labels[start:start + chunk_size].to(device).unsqueeze(1)
        correct1 += (predictions[:, :1] == expected).sum().item()
        correct5 += (predictions == expected).any(dim=1).sum().item()
    return dict(top1=100 * correct1 / len(queries), top5=100 * correct5 / len(queries),
                k_used=k, reference_images=len(reference), query_images=len(queries))


def feature_statistics(features):
    values = F.normalize(features.float(), dim=1).double()
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, len(values) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    tolerance = eigenvalues.max() * max(values.shape) * torch.finfo(values.dtype).eps
    singular_values = eigenvalues[eigenvalues > tolerance].sqrt()
    if len(singular_values):
        probabilities = singular_values / singular_values.sum()
        effective_rank = (-(probabilities * probabilities.log()).sum()).exp().item()
    else:
        effective_rank = 0.0
    return dict(
        pooled_feature_std_mean=features.float().std(dim=0, correction=0).mean().item(),
        normalized_feature_std_mean=values.std(dim=0, correction=0).mean().item(),
        centered_effective_rank=effective_rank,
        feature_dimension=values.shape[1], images=len(values),
    )


def main():
    parser = argparse.ArgumentParser(description="Frozen I-JEPA teacher weighted kNN on ImageFolder")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--compare-random", action="store_true")
    parser.add_argument("--reference-per-class", type=int, default=20)
    parser.add_argument("--queries-per-class", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if any(value <= 0 for value in (args.reference_per_class, args.queries_per_class,
                                    args.batch_size, args.chunk_size, args.k)):
        parser.error("sample counts, batch-size, chunk-size and k must be positive")
    if args.num_workers < 0 or not 0 < args.temperature < float("inf"):
        parser.error("num-workers must be nonnegative and temperature positive and finite")
    if args.output.exists():
        parser.error("output already exists; choose a new JSON path")
    if args.train.resolve() == args.val.resolve():
        parser.error("train and val must be distinct image directories")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        parser.error("the selected GPU does not support bfloat16")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_options = checkpoint["config"]["encoder"]
    initialization_seed = checkpoint["config"].get("seed", args.seed)
    image_size = model_options.get("image_size", 224)
    resize_size = int(image_size * 256 / 224)
    transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train = datasets.ImageFolder(args.train, transform=transform)
    val = datasets.ImageFolder(args.val, transform=transform)
    if train.class_to_idx != val.class_to_idx:
        parser.error("train/val ImageFolder class_to_idx must match")
    class_names = checkpoint["config"].get("selected_classes") or train.classes
    if len(set(class_names)) != len(class_names) or not set(class_names).issubset(train.classes):
        parser.error("checkpoint selected_classes must be unique ImageFolder class names")
    generator = torch.Generator().manual_seed(args.seed)
    reference_indices = select_per_class(train, class_names, args.reference_per_class, generator)
    query_indices = select_per_class(val, class_names, args.queries_per_class, generator)
    reference_paths = [Path(train.samples[i][0]).resolve() for i in reference_indices]
    query_paths = [Path(val.samples[i][0]).resolve() for i in query_indices]
    if set(reference_paths).intersection(query_paths):
        parser.error("reference and query sets contain the same resolved image paths")
    loader_options = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                          shuffle=False, pin_memory=device.type == "cuda")
    reference_loader = DataLoader(Subset(train, reference_indices), **loader_options)
    query_loader = DataLoader(Subset(val, query_indices), **loader_options)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(encoder=model_options, selected_classes=class_names,
                  initialization_seed=initialization_seed,
                  class_to_idx=train.class_to_idx, resize=resize_size, crop=image_size,
                  pooling="mean of patch tokens after encoder final LayerNorm, then L2 normalization",
                  effective_rank="entropy rank of centered normalized reference feature singular values",
                  checkpoint_step=checkpoint.get("step"), torch_version=str(torch.__version__))
    result = dict(config=config, reference_paths=[str(path) for path in reference_paths],
                  query_paths=[str(path) for path in query_paths], metrics={})
    for name in ("target_encoder", "random_encoder") if args.compare_random else ("target_encoder",):
        torch.manual_seed(initialization_seed)
        encoder = VisionTransformer(**model_options)
        if name == "target_encoder":
            encoder.load_state_dict(checkpoint["target_encoder"], strict=True)
        encoder.to(device).eval().requires_grad_(False)
        reference, reference_labels = extract_features(encoder, reference_loader, device, args.bf16)
        queries, query_labels = extract_features(encoder, query_loader, device, args.bf16)
        metrics = weighted_knn(reference, reference_labels, queries, query_labels,
                               k=args.k, temperature=args.temperature,
                               chunk_size=args.chunk_size, device=device)
        metrics["reference_feature_statistics"] = feature_statistics(reference)
        result["metrics"][name] = metrics
        print(f"{name}: top1={metrics['top1']:.3f}% top5={metrics['top5']:.3f}% "
              f"rank={metrics['reference_feature_statistics']['centered_effective_rank']:.2f}", flush=True)
        del encoder
    if args.compare_random:
        result["top1_gain_percentage_points"] = (
            result["metrics"]["target_encoder"]["top1"] - result["metrics"]["random_encoder"]["top1"]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved evaluation to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
