import logging

import torch

logger = logging.getLogger(__name__)

# Coordinate space is patches of the image.
# A rectangle is [h, w, top, left]. scale is an area fraction of the
# patch grid. aspect_ratio is grid / grid.


def pick_target_patches(
    grid,
    M=4,
    aspect_ratio=(0.75, 1.5),
    scale=(0.15, 0.2),
    seed=0,
    device=None,
):
    """Sample M target blocks on the patch grid.

    grid: grid length of the square patch grid. Indexes are row * grid + col.

    One block size is shared. Targets may overlap.

    Output: list of M long tensors. Each tensor holds the patch indexes in that block.
    """
    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    logger.debug(
        "target masks: grid=%dx%d M=%d scale=%s aspect_ratio=%s seed=%s device=%s",
        grid,
        grid,
        M,
        scale,
        aspect_ratio,
        seed,
        device,
    )

    scale_min, scale_max = scale
    ratio_min, ratio_max = aspect_ratio
    u = torch.rand((), device=device, generator=generator).item()
    scale = scale_min + u * (scale_max - scale_min)
    ratio = ratio_min + u * (ratio_max - ratio_min)
    area = int(grid * grid * scale)
    block_h = min(round((area * ratio) ** 0.5), grid - 1)
    block_w = min(round((area / ratio) ** 0.5), grid - 1)

    logger.debug(
        "target size scale=%.4f ratio=%.4f -> h=%d w=%d area=%d",
        scale,
        ratio,
        block_h,
        block_w,
        area,
    )

    blocks = []
    for index in range(M):
        top = torch.randint(0, grid - block_h, (), device=device, generator=generator).item()
        left = torch.randint(0, grid - block_w, (), device=device, generator=generator).item()
        blocks.append([block_h, block_w, top, left])
        logger.debug("target_%d accepted top=%d left=%d", index + 1, top, left)

    target_patches_indices = []
    for block_h, block_w, top, left in blocks:
        rows = torch.arange(top, top + block_h, device=device)
        cols = torch.arange(left, left + block_w, device=device)
        grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing="ij")
        target_patches_indices.append((grid_rows * grid + grid_cols).reshape(-1))
    logger.debug("targets=%s", [tuple(indices.shape) for indices in target_patches_indices])
    return target_patches_indices


def pick_context_patches(
    grid,
    target_patches_indices,
    aspect_ratio=(1.0, 1.0),
    scale=(0.85, 1.0),
    seed=0,
    min_keep=10,
    allow_overlap=False,
    device=None,
):
    """Sample the context crop and drop target patches from it.

    grid: grid length of the square patch grid. Indexes are row * grid + col.

    target_patches_indices:
        Patch indexes from pick_target_patches.
        One long tensor per target.

    The crop is kept when more than min_keep patches remain. allow_overlap
    leaves every target patch in the context mask.

    Returns:
        context_patches_indices: long tensor of kept patch indexes, shape [K].
    """
    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    logger.debug(
        "context mask: grid=%dx%d scale=%s aspect_ratio=%s seed=%s targets=%d min_keep=%d allow_overlap=%s device=%s",
        grid,
        grid,
        scale,
        aspect_ratio,
        seed,
        len(target_patches_indices),
        min_keep,
        allow_overlap,
        device,
    )

    scale_min, scale_max = scale
    ratio_min, ratio_max = aspect_ratio
    u = torch.rand((), device=device, generator=generator).item()
    scale = scale_min + u * (scale_max - scale_min)
    ratio = ratio_min + u * (ratio_max - ratio_min)
    area = int(grid * grid * scale)
    block_h = min(round((area * ratio) ** 0.5), grid - 1)
    block_w = min(round((area / ratio) ** 0.5), grid - 1)

    # 1. Sample context crop

    logger.debug(
        "context size scale=%.4f ratio=%.4f -> h=%d w=%d area=%d",
        scale,
        ratio,
        block_h,
        block_w,
        area,
    )

    if allow_overlap:
        exclusions = []
    else:
        exclusions = [indices.to(device) for indices in target_patches_indices]
    dropped = 0
    attempts = 0

    while True:
        top = torch.randint(0, grid - block_h, (), device=device, generator=generator).item()
        left = torch.randint(0, grid - block_w, (), device=device, generator=generator).item()

        # ---------------------------------------------------------
        # 2. Context crop
        # ---------------------------------------------------------

        rows = torch.arange(top, top + block_h, device=device)
        cols = torch.arange(left, left + block_w, device=device)
        grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing="ij")
        context_patches_indices = (grid_rows * grid + grid_cols).reshape(-1)

        # ---------------------------------------------------------
        # 3. Remove Target masks from Context mask
        # ---------------------------------------------------------

        active = exclusions[: max(len(exclusions) - dropped, 0)]
        if active:
            before = context_patches_indices.numel()
            banned = torch.cat(active)
            context_patches_indices = context_patches_indices[
                ~torch.isin(context_patches_indices, banned)
            ]
            logger.debug(
                "context minus %d targets removed %d patches",
                len(active),
                before - context_patches_indices.numel(),
            )

        # 4. Extract patches belonging to context mask

        if context_patches_indices.numel() > min_keep:
            logger.debug(
                "context kept %d/%d patches at top=%d left=%d shape=%s dropped_targets=%d",
                context_patches_indices.numel(),
                grid * grid,
                top,
                left,
                tuple(context_patches_indices.shape),
                dropped,
            )
            return context_patches_indices

        attempts += 1
        if attempts == 20:
            dropped += 1
            attempts = 0
            logger.debug("context mask rejected, dropped target constraints=%d", dropped)
