import logging

import torch

logger = logging.getLogger(__name__)

# Coordinate space is patches of the image.
# A rectangle is [h, w, top, left]. scale is an area fraction of the
# patch grid. aspect_ratio is height / width.


def target_crop_patches_emb(
    patches,
    M=4,
    aspect_ratio=(0.75, 1.5),
    scale=(0.15, 0.2),
    overlap=True,
    seed=0,
):
    """Sample M target rectangles on the patch grid.

    patches: [C, H, W, D]. Only H and W are read.

    Targets may overlap. overlap=True matches I-JEPA.

    Output: [M, 4] long tensor. Each row is [h, w, top, left].
    """
    _, height, width, _ = (
        patches.shape
    )  # TODO: adapt to encoder, needed height, width
    device = patches.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    blocks = []

    logger.info(
        "target masks: grid=%dx%d M=%d scale=%s aspect_ratio=%s overlap=%s seed=%s",
        height,
        width,
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

            h = min(height, max(1, round(height * scale_crop)))
            w = min(width, max(1, round(h * aspect_crop)))

            t = torch.randint(
                0, height - h + 1, (), device=device, generator=generator
            ).item()
            l = torch.randint(
                0, width - w + 1, (), device=device, generator=generator
            ).item()

            overlaps = False
            if not overlap:
                for prev_h, prev_w, prev_t, prev_l in blocks:
                    overlaps_w = max(prev_l, l) < min(prev_l + prev_w, l + w)
                    overlaps_h = max(prev_t, t) < min(prev_t + prev_h, t + h)
                    if overlaps_w and overlaps_h:
                        overlaps = True
                        break

            logger.info(
                "%s sample scale=%.4f aspect_ratio=%.4f -> [h=%d, w=%d, top=%d, left=%d] area=%d overlap=%s",
                name,
                scale_crop,
                aspect_crop,
                h,
                w,
                t,
                l,
                h * w,
                overlaps,
            )

            if not overlaps:
                blocks.append([h, w, t, l])
                logger.info("%s accepted", name)
                break

    if not blocks:
        return torch.zeros((0, 4), dtype=torch.long, device=device)
    return torch.tensor(blocks, dtype=torch.long, device=device)


def context_crop_patches(
    patches,
    blocks,
    aspect_ratio=(1.0, 1.0),
    scale=(0.85, 1.0),
    seed=0,
):
    """Sample the context crop and drop every target rectangle from it.

    patches: [C, H, W, D].

    blocks:
        Target rectangles from target_crop_patches_emb.
        Each block is [h, w, top, left], shape [M, 4].

    Returns:
        context_patches: [C, N, D]
        keep_mask:       [H, W]
        context_crop:    [h, w, top, left]
    """
    _, height, width, _ = patches.shape # TODO: adapt to encoder, needed height, width
    device = patches.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    blocks = blocks.tolist()

    logger.info(
        "context mask A: grid=%dx%d scale=%s aspect_ratio=%s seed=%s targets=%d",
        height,
        width,
        scale,
        aspect_ratio,
        seed,
        len(blocks),
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

    crop_h = min(height, max(1, round(height * scale_crop)))
    crop_w = min(width, max(1, round(width * scale_crop * aspect_ratio_crop)))

    top_pos_crop = torch.randint(
        0,
        height - crop_h + 1,
        (),
        device=device,
        generator=generator,
    ).item()

    left_pos_crop = torch.randint(
        0,
        width - crop_w + 1,
        (),
        device=device,
        generator=generator,
    ).item()

    context_crop = [
        crop_h,
        crop_w,
        top_pos_crop,
        left_pos_crop,
    ]

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

    keep_mask = torch.zeros(
        (height, width),
        dtype=torch.bool,
        device=device,
    )

    keep_mask[
        top_pos_crop : top_pos_crop + crop_h,
        left_pos_crop : left_pos_crop + crop_w,
    ] = True

    # ---------------------------------------------------------
    # 3. Remove Target masks from Context mask
    # ---------------------------------------------------------

    for index, (target_h, target_w, target_top, target_left) in enumerate(blocks):
        target_bottom = min(target_top + target_h, height)
        target_right = min(target_left + target_w, width)
        name = f"target_{index + 1}"

        if target_top < target_bottom and target_left < target_right:
            removed = int(
                keep_mask[target_top:target_bottom, target_left:target_right]
                .sum()
                .item()
            )
            keep_mask[target_top:target_bottom, target_left:target_right] = False
            logger.info(
                "A minus %s [h=%d, w=%d, top=%d, left=%d] removed %d patches",
                name,
                target_h,
                target_w,
                target_top,
                target_left,
                removed,
            )
        else:
            logger.info("A minus %s skipped, rectangle is empty", name)

    # 4. Extract patches belonging to context mask

    context_patches = patches[:, keep_mask, :]

    logger.info(
        "A kept %d/%d patches, context_patches shape=%s",
        int(keep_mask.sum().item()),
        height * width,
        tuple(context_patches.shape),
    )

    logger.info(f"{context_patches=} {keep_mask=} {context_crop=}")
    return context_patches, keep_mask, context_crop
