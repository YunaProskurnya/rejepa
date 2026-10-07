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
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s - %(message)s [%(filename)s:%(lineno)d]",
    handlers=[logging.FileHandler("transforms.log", mode="w"), logging.StreamHandler()],
)


def make_transform(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    horizontal_flip=True,
    color_distortion=True,
    color_jitter=1.0,
    gaussian_blur=True,
):
    def get_color_distortion(s):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
        return color_distort

    transform_list = []
    transform_list += [transforms.RandomResizedCrop(crop_size, scale=crop_scale)]
    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if color_distortion:
        transform_list += [get_color_distortion(s=color_jitter)]
    if gaussian_blur:
        transform_list += [GaussianBlur(p=0.5)]
    transform_list += [transforms.ToTensor()]
    transform_list += [
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]

    transform = transforms.Compose(transform_list)
    return transform


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(()).item() * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


def process_image(image_path, patch_num=196):
    image = tv_io.read_image(image_path, mode=tv_io.ImageReadMode.RGB)
    image = transforms.ToPILImage()(image)
    image = make_transform()(image)

    # Creating patches with shape (C, n_patches_h, n_patches_w, patch_size, patch_size)
    C, H, W = image.shape
    patch_size = int((H * W / patch_num) ** 0.5)
    n_patches_h = H // patch_size
    n_patches_w = W // patch_size

    cropped_h = n_patches_h * patch_size
    cropped_w = n_patches_w * patch_size
    image = image[:, :cropped_h, :cropped_w]

    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous() # (n_patches_h, n_patches_w, C, patch_size, patch_size)
    patches = patches.view(-1, C, patch_size, patch_size)  # (num_patches, C, patch_size, patch_size)
    
    logger.info(f"{patches.shape=}")

    # base_name = os.path.splitext(os.path.basename(image_path))[0]
    # save_path = os.path.join(
    #     os.path.dirname(image_path), f"{base_name}_patches.png"
    # )
    # save_image(image, save_path)

    target_patches_indices = masks.pick_target_patches(patches)
    context_patches_indices = masks.pick_context_patches(patches, target_patches_indices)

    return image, target_patches_indices, context_patches_indices


if "__main__" == __name__:
    image, target_patches_indices, context_patches_indices = process_image("example/lenna.png")
    logger.info(
        "targets=%s context=%s",
        [tuple(indices.shape) for indices in target_patches_indices],
        tuple(context_patches_indices.shape),
    )
