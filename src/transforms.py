"""
1. Load image
2. Decode it into RGB.
3. Sample **one random resized crop** and r
4. resize it to your model’s input resolution
5. Convert it to a tensor and normalize its channels.
6. Sample context and target masks on the resulting **patch grid**
"""

import logging
import os
import random

try:
    from . import masks
except ImportError:
    import masks

import torch
import torchvision.io as tv_io
import torchvision.transforms as transforms
from PIL import ImageFilter
from torchvision.utils import save_image

logger = logging.getLogger(__name__)


def make_transform(crop_size=224, crop_scale=(0.3, 1.0), seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    transform_list = [
        transforms.RandomResizedCrop(crop_size, scale=crop_scale),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    return transforms.Compose(transform_list)


def process_image(image_path, image_size=224, patch_size=16, seed=0, device=None):
    if image_size % patch_size != 0:
        raise ValueError("image_size must be divisible by patch_size")

    image = tv_io.read_image(image_path, mode=tv_io.ImageReadMode.RGB)
    image = transforms.ToPILImage()(image)
    image = make_transform(crop_size=image_size, seed=seed)(image)
    image = image.to(torch.device("cpu") if device is None else device)

    grid = image_size // patch_size
    torch.manual_seed(seed)
    target_size = masks.sample_block_size(grid, (0.15, 0.2), (0.75, 1.5))
    context_size = masks.sample_block_size(grid, (0.85, 1.0), (1.0, 1.0))
    target_patches_indices = masks.pick_target_patches(grid, target_size)
    context_patches_indices = masks.pick_context_patches(
        grid, target_patches_indices, context_size
    )

    return image, target_patches_indices, context_patches_indices


if "__main__" == __name__:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s - %(message)s [%(filename)s:%(lineno)d]",
        handlers=[logging.FileHandler("transforms.log", mode="w"), logging.StreamHandler()],
    )
    image, target_patches_indices, context_patches_indices = process_image(
        "example/lenna.png"
    )
    logger.info(
        "targets=%s context=%s",
        [tuple(indices.shape) for indices in target_patches_indices],
        tuple(context_patches_indices.shape),
    )
