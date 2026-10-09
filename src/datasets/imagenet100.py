import os
from logging import getLogger
from multiprocessing import Value
import math

import torch
import torchvision

from src.masks import pick_context_patches, pick_target_patches, sample_block_size

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):
    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        M=4,
        min_keep=10,
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.patch_size = patch_size
        self.height, self.width = (
            input_size[0] // patch_size,
            input_size[1] // patch_size,
        )
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.M = M
        self.min_keep = min_keep  # minimum number of patches to keep
        self._itr_counter = Value("i", -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def __call__(self, batch):
        """
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        """
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)
        collated_masks_target, collated_masks_context = [], []
        min_target_size = self.height * self.width
        min_context_size = self.height * self.width
        assert self.height == self.width, "height and width must be equal"

        seed = self.step()
        generator = torch.Generator()
        generator.manual_seed(seed)
        target_size = sample_block_size(
            self.height, self.pred_mask_scale, self.aspect_ratio, generator
        )
        context_size = sample_block_size(
            self.height, self.enc_mask_scale, (1.0, 1.0), generator
        )

        # TODO: fix
        for _ in range(B):
            target_patches_indices = pick_target_patches(
                self.height,
                M=self.M,
                scale=self.pred_mask_scale,
                aspect_ratio=self.aspect_ratio,
                block_size=target_size,
            )
            min_target_size = min(
                min_target_size, min(len(mask) for mask in target_patches_indices)
            )
            collated_masks_target.append(target_patches_indices)

            context_patches_indices = pick_context_patches(
                self.height,
                target_patches_indices,
                scale=self.enc_mask_scale,
                aspect_ratio=(1.0, 1.0),
                min_keep=self.min_keep,
                block_size=context_size,
            )
            min_context_size = min(min_context_size, len(context_patches_indices))
            collated_masks_context.append([context_patches_indices])

        collated_masks_target = [
            [cm[:min_target_size] for cm in cm_list]
            for cm_list in collated_masks_target
        ]
        collated_masks_target = torch.utils.data.default_collate(collated_masks_target)

        collated_masks_context = [
            [cm[:min_context_size] for cm in cm_list]
            for cm_list in collated_masks_context
        ]
        collated_masks_context = torch.utils.data.default_collate(
            collated_masks_context
        )

        return collated_batch, collated_masks_context, collated_masks_target


def make_imagenet100(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    training=True,
    drop_last=True,
):
    suffix = "train/" if training else "val/"

    dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(root_path, suffix), transform=transform
    )
    logger.info("Initialized ImageNet on data-path %s", root_path)

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset, num_replicas=world_size, rank=rank
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
    )

    logger.info("ImageNet unsupervised data loader created")

    return dataset, data_loader, dist_sampler


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)
