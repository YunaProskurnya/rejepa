import torch

from .losses import prediction_loss
from .models.target_encoder import prepare_targets, update_target_encoder


def train_step(encoder, predictor, target_encoder, optimizer,
               images, context_indices, target_indices, momentum, *,
               precision="fp32", accumulation_steps=1, update=True, zero_grad=True):
    """Backpropagate one microbatch; update optimizer and EMA on the last one.

    For K equally sized microbatches, pass accumulation_steps=K throughout,
    zero_grad=True only on the first and update=True only on the last.
    Returned loss is unscaled; grad_norm is None until the optimizer update.
    """
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    if not isinstance(accumulation_steps, int) or accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be a positive integer")
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)

    with torch.autocast(images.device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
        targets = prepare_targets(target_encoder, images, target_indices)
        context = encoder(images, context_indices)
        predictions = predictor(context, context_indices, target_indices)
        loss = prediction_loss(predictions, targets)
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
        update_target_encoder(encoder, target_encoder, momentum)

    return {"loss": loss.item(), "grad_norm": grad_norm}
