"""Team collator and data loading, integrated from data_encoder commit 930120e."""

import os
from logging import getLogger
from multiprocessing import Value

import torch
import torchvision

from ..masks import pick_context_patches, pick_target_patches, sample_block_size

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
        if (len(input_size) != 2 or input_size[0] != input_size[1]
                or patch_size <= 0 or input_size[0] % patch_size
                or input_size[0] // patch_size < 2):
            raise ValueError("input_size must be square and divisible by patch_size, with at least 2 patches per side")
        if M <= 0 or min_keep < 0:
            raise ValueError("M must be positive and min_keep nonnegative")
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
        """Return ((images, labels), [context B x K], M target tensors B x T)."""
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)
        collated_masks_target, collated_masks_context = [], []
        min_target_size = self.height * self.width
        min_context_size = self.height * self.width

        seed = self.step()
        generator = torch.Generator()
        generator.manual_seed(seed)
        target_size = sample_block_size(
            self.height, self.pred_mask_scale, self.aspect_ratio, generator
        )
        context_size = sample_block_size(
            self.height, self.enc_mask_scale, (1.0, 1.0), generator
        )

        if min(target_size[0] * target_size[1], context_size[0] * context_size[1]) <= self.min_keep:
            raise ValueError("sampled target/context blocks must contain more than min_keep patches")

        for _ in range(B):
            target_patches_indices = pick_target_patches(
                self.height,
                M=self.M,
                scale=self.pred_mask_scale,
                aspect_ratio=self.aspect_ratio,
                block_size=target_size, seed=None,
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
                block_size=context_size, seed=None,
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


def make_data_loader(dataset, batch_size, collator=None, pin_mem=True,
                     num_workers=8, drop_last=True, *, sampler=None,
                     generator=None, worker_init_fn=None):
    """Build batches; the caller supplies order, worker seeds and resume position."""
    return torch.utils.data.DataLoader(
        dataset, collate_fn=collator, sampler=sampler, batch_size=batch_size,
        drop_last=drop_last, pin_memory=pin_mem, num_workers=num_workers,
        generator=generator, worker_init_fn=worker_init_fn, persistent_workers=False,
    )


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
    seed=0,
    worker_init_fn=None,
):
    """Read root_path/train or root_path/val; call sampler.set_epoch(epoch) when training."""
    suffix = "train/" if training else "val/"

    dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(root_path, suffix), transform=transform
    )
    logger.info("Initialized ImageNet on data-path %s", dataset.root)

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset, num_replicas=world_size, rank=rank, shuffle=training, seed=seed
    )

    data_loader = make_data_loader(
        dataset, batch_size, collator, pin_mem, num_workers, drop_last,
        sampler=dist_sampler, generator=torch.Generator().manual_seed(seed),
        worker_init_fn=worker_init_fn,
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
