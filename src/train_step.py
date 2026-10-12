import math

import torch

from .losses import prediction_loss
from .models.target_encoder import prepare_targets, update_target_encoder
from .regularizers.apply_regularization import apply_regularization


def train_step(encoder, predictor, target_encoder, optimizer,
               images, context_indices, target_indices, momentum, *,
               precision="fp32", accumulation_steps=1, update=True, zero_grad=True,
               mode="ema", regularizer="none", regularizer_weight=0.02,
               prediction_loss_type="smooth_l1", noise_std=0.0):
    """Backpropagate one microbatch; update optimizer and EMA on the last one.

    For K equally sized microbatches, pass accumulation_steps=K throughout,
    zero_grad=True only on the first and update=True only on the last.
    Returned loss is unscaled; grad_norm is None until the optimizer update.
    """
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    if not isinstance(accumulation_steps, int) or accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be a positive integer")
    if mode not in ("ema", "joint"):
        raise ValueError("mode must be ema or joint")
    if regularizer not in ("none", "sigreg", "visreg"):
        raise ValueError("regularizer must be none, sigreg or visreg")
    if not math.isfinite(regularizer_weight) or regularizer_weight < 0:
        raise ValueError("regularizer_weight must be finite and nonnegative")
    if prediction_loss_type not in ("smooth_l1", "mse"):
        raise ValueError("prediction_loss_type must be smooth_l1 or mse")
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    if mode == "ema" and (target_encoder is None or regularizer != "none"):
        raise ValueError("ema requires a target encoder and no regularizer")
    if mode == "joint" and target_encoder is not None:
        raise ValueError("joint uses the encoder without a separate target encoder")
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)

    with torch.autocast(images.device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
        targets, features = prepare_targets(
            encoder if mode == "joint" else target_encoder, images, target_indices,
            joint=mode == "joint", return_features=True,
        )
        context = encoder(images, context_indices)
        if noise_std:
            context = context + noise_std * torch.randn(
                context.shape, device=context.device, dtype=torch.float32,
            ).to(context.dtype)
        predictions = predictor(context, context_indices, target_indices)
        prediction = prediction_loss(predictions, targets, prediction_loss_type)
        regularization = apply_regularization(features, regularizer)
        loss = prediction + regularizer_weight * regularization
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite loss")

    (loss / accumulation_steps).backward()

    grad_norm = None
    if update:
        trainable_params = []
        for group in optimizer.param_groups:
            trainable_params.extend(group["params"])
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params, max_norm=float("inf"), error_if_nonfinite=True,
        ).item()

        optimizer.step()
        if mode == "ema":
            update_target_encoder(encoder, target_encoder, momentum)

    metrics = {"loss": loss.item(), "grad_norm": grad_norm}
    if mode == "joint":
        metrics.update(
            prediction_loss=prediction.item(),
            regularization_loss=regularization.detach().item() if regularizer != "none" else 0.0,
        )
    return metrics
