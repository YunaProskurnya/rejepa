import torch


def create_optimizer(encoder, predictor, lr, weight_decay, *, fused=None):
    decay_params = []
    no_decay_params = []

    for module in (encoder, predictor):
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue

            if "bias" in name or parameter.ndim == 1:
                no_decay_params.append(parameter)
            else:
                decay_params.append(parameter)

    parameter_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0, "WD_exclude": True},
    ]

    return torch.optim.AdamW(parameter_groups, lr=lr, fused=fused)
