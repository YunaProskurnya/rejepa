import torch
import torch.nn as nn
import torch.optim as optim
from datasets.imagenet100 import MaskCollator, make_imagenet100
from transforms import make_transform

def main():
    SEED = 42
    world_size = 1
    rank = 0

    mask_collator = MaskCollator(
        input_size=224,
        patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        M=4,
        min_keep=10,
    )

    image_transform = make_transform(seed=SEED)

    # dataloader
    _, unsupervised_loader, unsupervised_sampler = make_imagenet100(
        transform=image_transform,
        batch_size=128,
        collator=mask_collator,
        pin_mem=True,
        num_workers=10,
        world_size=world_size,
        rank=rank,
        root_path='data',
        training=True,
        drop_last=True, 
    )

if __name__ == "__main__":
    main()
