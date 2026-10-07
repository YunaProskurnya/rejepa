"""Создание target encoder и EMA для текущего ViT без изменяемых буферов."""

import copy

import torch
import torch.nn.functional as F

from .token_selection import select_tokens


def create_target_encoder(encoder):
    target_encoder = copy.deepcopy(encoder)
    target_encoder.requires_grad_(False)
    target_encoder.zero_grad(set_to_none=True)
    return target_encoder


@torch.no_grad()
def prepare_targets(target_encoder, images, target_indices):
    """Полное изображение -> нормализованные цели [B,Nt,D]"""
    features = target_encoder(images)
    features = F.layer_norm(features, (features.shape[-1],), eps=1e-5)
    if target_indices.ndim != 2 or target_indices.shape[0] != images.shape[0]:
        raise ValueError("target_indices must have shape [B,Nt]")
    return select_tokens(features, target_indices)


@torch.no_grad()
def update_target_encoder(encoder, target_encoder, momentum):
    """После optimizer.step: target = momentum*target + (1-momentum)*encoder."""
    if not 0.0 <= momentum <= 1.0:
        raise ValueError("momentum must be between 0 and 1")
    if encoder is target_encoder:
        raise ValueError("target_encoder must be a separate copy of encoder")

    source_params = dict(encoder.named_parameters())
    target_params = dict(target_encoder.named_parameters())
    if source_params.keys() != target_params.keys():
        raise ValueError("encoder and target_encoder must have matching parameter names")
    for name, source in source_params.items():
        target = target_params[name]
        if source.shape != target.shape or source.device != target.device:
            raise ValueError(f"incompatible target parameter: {name}")

    for name, source in source_params.items():
        if source.requires_grad:
            target_params[name].mul_(momentum).add_(source, alpha=1.0 - momentum)
