import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from .transforms import make_evaluation_transform


CSV_FIELDS = ("epoch", "step", "images", "feature_dimension", "mean_variance",
              "singular_entropy", "effective_rank", "artifact")


def make_representation_loader(data, image_size, samples, seed, batch_size, device):
    dataset = datasets.ImageFolder(data, transform=make_evaluation_transform(image_size))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:samples].tolist()
    paths = [str(Path(dataset.samples[index][0]).resolve()) for index in indices]
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(seed))
    return paths, loader


def spectrum_statistics(features):
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("features must have shape [N, D], N >= 2")

    values = features.detach().double().cpu()
    if not torch.isfinite(values).all():
        raise ValueError("features must be finite")

    n, d = values.shape

    covariance = torch.from_numpy(np.cov(values.numpy(), rowvar=False))
    eigenvalues = torch.linalg.eigvalsh(covariance).flip(0)
    singular_values = eigenvalues[:min(n - 1, d)].clamp_min(0).sqrt()

    total = singular_values.sum()
    if total > 0:
        p = singular_values / total
        p = p[p > 0]
        entropy = -(p * p.log()).sum().item()
    else:
        entropy = 0.0

    return {
        "singular_values": singular_values.float(),
        "singular_entropy": entropy,
        "effective_rank": math.exp(entropy) if total > 0 else 0.0,
        "mean_variance": covariance.diagonal().mean().item(),
        "images": n,
        "feature_dimension": d,
    }


def patchwise_spectrum_statistics(features):
    if features.ndim != 3 or features.shape[1] < 1:
        raise ValueError("features must have shape [N, P, D] with P >= 1")

    statistics = spectrum_statistics(features.mean(dim=1))
    statistics["patch_effective_ranks"] = torch.tensor([
        spectrum_statistics(features[:, position, :])["effective_rank"]
        for position in range(features.shape[1])
    ])
    return statistics


@torch.inference_mode()
def evaluate_representation(encoder, loader, device, bf16, *, patchwise=False):
    features = []
    for images, _ in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            encoded = encoder(images.to(device, non_blocking=True)).float()
            features.append((encoded if patchwise else encoded.mean(dim=1)).cpu())

    values = torch.cat(features)
    return patchwise_spectrum_statistics(values) if patchwise else spectrum_statistics(values)


def save_representation(output, epoch, step, statistics):
    artifact = f"spectrum_{epoch}.pt"
    torch.save(statistics, output / artifact)
    row = {key: statistics[key] for key in CSV_FIELDS if key in statistics}
    row.update(epoch=epoch, step=step, artifact=artifact)
    with (output / "representations.csv").open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
    if "patch_effective_ranks" in statistics:
        with (output / "patch_ranks.csv").open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if stream.tell() == 0:
                writer.writerow(("epoch", "step", "patch_index", "effective_rank"))
            writer.writerows(
                (epoch, step, position, rank.item())
                for position, rank in enumerate(statistics["patch_effective_ranks"])
            )
