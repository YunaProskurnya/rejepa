import torch
import torch.nn.functional as F


def prediction_loss(predictions, targets, loss_type="smooth_l1"):
    if loss_type not in ("smooth_l1", "mse"):
        raise ValueError("loss_type must be smooth_l1 or mse")
    predictions = predictions if isinstance(predictions, list) else [predictions]
    targets = targets if isinstance(targets, list) else [targets]
    if not predictions or len(predictions) != len(targets):
        raise ValueError("predictions and targets must contain the same nonzero number of blocks")

    losses = []
    for prediction, target in zip(predictions, targets):
        if prediction.ndim != 3 or prediction.shape != target.shape:
            raise ValueError("each prediction and target must have matching shapes [B,Nt,D]")
        if prediction.numel() == 0:
            raise ValueError("prediction and target blocks must be nonempty")
        if loss_type == "mse":
            losses.append(F.mse_loss(prediction, target, reduction="mean"))
        else:
            losses.append(F.smooth_l1_loss(prediction, target, beta=1.0, reduction="mean"))
    return torch.stack(losses).mean()
