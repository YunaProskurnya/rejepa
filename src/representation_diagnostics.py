import math

import torch


def spectrum_statistics(features):
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("features must have shape [at least 2 images, dimensions]")

    values = features.detach().double().cpu()
    if not torch.isfinite(values).all():
        raise ValueError("features must be finite")

    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (len(values) - 1)
    singular_values = torch.linalg.svdvals(centered) / math.sqrt(len(values) - 1)
    singular_values = singular_values[:min(len(values) - 1, values.shape[1])]
    total = singular_values.sum()
    if total > 0:
        probabilities = singular_values / total
        nonzero = probabilities[probabilities > 0]
        entropy = -(nonzero * nonzero.log()).sum().item()
    else:
        entropy = 0.0

    return {
        "covariance": covariance.float(),
        "singular_values": singular_values.float(),
        "singular_entropy": entropy,
        "effective_rank": math.exp(entropy) if total > 0 else 0.0,
        "mean_variance": covariance.diagonal().mean().item(),
        "images": len(values),
        "feature_dimension": values.shape[1],
    }


@torch.inference_mode()
def evaluate_representation(encoder, loader, device, precision):
    was_training = encoder.training
    encoder.eval()
    features = []
    try:
        for images, _ in loader:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                pooled = encoder(images.to(device, non_blocking=True)).float().mean(dim=1)
            features.append(pooled.cpu())
    finally:
        encoder.train(was_training)

    return spectrum_statistics(torch.cat(features))
