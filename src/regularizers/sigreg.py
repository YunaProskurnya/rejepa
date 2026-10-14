import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    def __init__(self, num_slices=256, num_knots=17, t_max=3.0):
        super().__init__()
        if num_slices < 1 or num_knots < 2 or t_max <= 0:
            raise ValueError("num_slices must be positive, num_knots at least 2, and t_max positive")

        self.num_slices = num_slices
        t = torch.linspace(0, t_max, num_knots, dtype=torch.float32)
        weights = torch.full((num_knots,), 2 * t_max / (num_knots - 1), dtype=torch.float32)
        weights[0] /= 2
        weights[-1] /= 2
        phi = torch.exp(-t.square() / 2)

        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, z):
        if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 1:
            raise ValueError("z must have shape [N, D] with N >= 2 and D >= 1")
        if not torch.is_floating_point(z):
            raise TypeError("z must be a floating-point tensor")

        with torch.autocast(z.device.type, enabled=False):
            x = z.float() if z.dtype in (torch.float16, torch.bfloat16) else z
            directions = torch.randn(x.shape[1], self.num_slices, device=x.device, dtype=x.dtype)
            directions = F.normalize(directions, dim=0)

            t = self.t.to(x)
            phi = self.phi.to(x)
            weights = self.weights.to(x)
            angles = (x @ directions).unsqueeze(-1) * t
            real = angles.cos().mean(dim=0)
            imag = angles.sin().mean(dim=0)
            error = (real - phi).square() + imag.square()

            return ((error @ weights) * x.shape[0]).mean()
