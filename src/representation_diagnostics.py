import math

import numpy as np
import torch


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


@torch.inference_mode()
def evaluate_representation(encoder, loader, device, bf16):
    features = []
    for images, _ in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            features.append(encoder(images.to(device, non_blocking=True)).float().mean(dim=1).cpu())

    return spectrum_statistics(torch.cat(features))
