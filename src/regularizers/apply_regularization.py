import torch

from .sigreg import SIGReg
from .visreg import VISReg


def apply_regularization(features, name):
    if name == "none":
        return 0
    if name == "sigreg":
        regularizer = SIGReg()
    elif name == "visreg":
        regularizer = VISReg()
    else:
        raise ValueError(f"Unknown regularizer: {name}")

    return torch.stack([
        regularizer(features[:, position, :])
        for position in range(features.shape[1])
    ]).mean()
