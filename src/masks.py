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
    overlap=True,
    seed=0,
):
    """Sample M target blocks on the patch grid.

    grid: grid length of the square patch grid. Indexes are row * grid + col.

    Targets may overlap. overlap=True matches I-JEPA.

    Output: list of M long tensors. Each tensor holds the patch indexes in that block.
    """
    device = torch.device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    blocks = []

    logger.info(
        "target masks: grid=%dx%d M=%d scale=%s aspect_ratio=%s overlap=%s seed=%s",
        grid,
        grid,
        M,
        scale,
        aspect_ratio,
        overlap,
        seed,
    )

    for index in range(M):
        name = f"target_{index + 1}"
        while True:
            scale_crop = (
                torch.empty((), device=device)
                .uniform_(*scale, generator=generator)
                .item()
            )
            aspect_crop = (
                torch.empty((), device=device)
                .uniform_(*aspect_ratio, generator=generator)
                .item()
            )

            h = min(grid, max(1, round(grid * scale_crop)))
            w = min(grid, max(1, round(h * aspect_crop)))

            t = torch.randint(
                0, grid - h + 1, (), device=device, generator=generator
            ).item()
            l = torch.randint(
                0, grid - w + 1, (), device=device, generator=generator
            ).item()

            overlaps = False
            if not overlap:
                for prev_h, prev_w, prev_t, prev_l in blocks:
                    overlaps_w = max(prev_l, l) < min(prev_l + prev_w, l + w)
                    overlaps_h = max(prev_t, t) < min(prev_t + prev_h, t + h)
                    if overlaps_w and overlaps_h:
                        overlaps = True
                        break

            if not overlaps:
                blocks.append([h, w, t, l])
                logger.info("%s accepted", name)
                break

    target_patches_indices = []
    for block_h, block_w, top, left in blocks:
        rows = torch.arange(top, top + block_h, device=device)
        cols = torch.arange(left, left + block_w, device=device)
        grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing="ij")
        target_patches_indices.append((grid_rows * grid + grid_cols).reshape(-1))
    logger.info(f"{scale_crop=:4} {aspect_crop=:4} {overlap=} {h * w=} {target_patches_indices=}")
    return target_patches_indices


def pick_context_patches(
    grid,
    target_patches_indices,
    aspect_ratio=(1.0, 1.0),
    scale=(0.85, 1.0),
    seed=0,
):
    """Sample the context crop and drop every target patch from it.

    grid: grid length of the square patch grid. Indexes are row * grid + col.

    target_patches_indices:
        Patch indexes from pick_target_patches.
        One long tensor per target.

    Returns:
        context_patches_indices: long tensor of kept patch indexes, shape [K].
    """
    device = torch.device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    logger.info(
        "context mask: grid=%dx%d scale=%s aspect_ratio=%s seed=%s targets=%d",
        grid,
        grid,
        scale,
        aspect_ratio,
        seed,
        len(target_patches_indices),
    )

    # 1. Sample context crop

    scale_crop = (
        torch.empty((), device=device).uniform_(*scale, generator=generator).item()
    )
    aspect_ratio_crop = (
        torch.empty((), device=device)
        .uniform_(*aspect_ratio, generator=generator)
        .item()
    )

    crop_h = min(grid, max(1, round(grid * scale_crop)))
    crop_w = min(grid, max(1, round(grid * scale_crop * aspect_ratio_crop)))

    top_pos_crop = torch.randint(
        0,
        grid - crop_h + 1,
        (),
        device=device,
        generator=generator,
    ).item()

    left_pos_crop = torch.randint(
        0,
        grid - crop_w + 1,
        (),
        device=device,
        generator=generator,
    ).item()

    logger.info(
        "A sample scale=%.4f aspect_ratio=%.4f -> [h=%d, w=%d, top=%d, left=%d] area=%d",
        scale_crop,
        aspect_ratio_crop,
        crop_h,
        crop_w,
        top_pos_crop,
        left_pos_crop,
        crop_h * crop_w,
    )

    # ---------------------------------------------------------
    # 2. Context crop
    # ---------------------------------------------------------

    rows = torch.arange(top_pos_crop, top_pos_crop + crop_h, device=device)
    cols = torch.arange(left_pos_crop, left_pos_crop + crop_w, device=device)
    grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing="ij")
    context_patches_indices = (grid_rows * grid + grid_cols).reshape(-1)

    # ---------------------------------------------------------
    # 3. Remove Target masks from Context mask
    # ---------------------------------------------------------

    for index, target_indices in enumerate(target_patches_indices):
        before = context_patches_indices.numel()
        context_patches_indices = context_patches_indices[
            ~torch.isin(context_patches_indices, target_indices.to(device))
        ]
        logger.info(
            f"Context mask minus target_{index + 1} removed {before - context_patches_indices.numel()} patches",
        )

    # 4. Extract patches belonging to context mask

    logger.info(
        "A kept %d/%d patches, context_patches_indices shape=%s",
        context_patches_indices.numel(),
        grid * grid,
        tuple(context_patches_indices.shape),
    )

    logger.info(f"{context_patches_indices=}")
    return context_patches_indices
