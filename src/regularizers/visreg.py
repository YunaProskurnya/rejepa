import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VISReg(nn.Module):
    def __init__(self, num_slices=64, scale_weight=1.0, shape_weight=1.0,
                 center_weight=1.0, eps=1e-6):
        super().__init__()
        if num_slices < 1 or eps <= 0:
            raise ValueError("num_slices and eps must be positive")
        if min(scale_weight, shape_weight, center_weight) < 0:
            raise ValueError("loss weights must be nonnegative")

        self.num_slices = num_slices
        self.scale_weight = scale_weight
        self.shape_weight = shape_weight
        self.center_weight = center_weight
        self.eps = eps

    def forward(self, z):
        if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 1:
            raise ValueError("z must have shape [N, D] with N >= 2 and D >= 1")
        if not torch.is_floating_point(z):
            raise TypeError("z must be a floating-point tensor")

        x = z.float() if z.dtype in (torch.float16, torch.bfloat16) else z
        center = x.mean(dim=0)
        centered = x - center
        std = centered.std(dim=0, unbiased=False)

        scale_loss = (1 - std).square().mean()
        center_loss = center.square().mean()

        normalized = centered / (std.detach() + self.eps)
        directions = torch.randn(x.shape[1], self.num_slices, device=x.device, dtype=x.dtype)
        directions = F.normalize(directions, dim=0)
        projected = (normalized @ directions).sort(dim=0).values

        probabilities = torch.arange(1, x.shape[0] + 1, device=x.device, dtype=x.dtype)
        probabilities = probabilities / (x.shape[0] + 1)
        quantiles = math.sqrt(2) * torch.erfinv(2 * probabilities - 1)
        shape_loss = (projected - quantiles[:, None]).square().mean()

        return (self.scale_weight * scale_loss + self.shape_weight * shape_loss
                + self.center_weight * center_loss)
