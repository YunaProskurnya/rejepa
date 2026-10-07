"""Draw one context/target crop on example/lenna.png.

Run from the repo root:

    python example/visualize_crops.py
"""

import logging
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.masks import context_crop_patches, target_crop_patches_emb

IMAGE_PATH = REPO_ROOT / "example" / "lenna.png"
OUTPUT_PATH = REPO_ROOT / "example" / "lenna_crops.png"
PATCH_SIZE = 50
SEED = 42

TARGET_COLORS = (
    (214, 69, 80),
    (232, 149, 47),
    (66, 116, 214),
    (142, 84, 196),
)
CONTEXT_COLOR = (36, 168, 104)
OUTSIDE_DIM = 0.28


def _font(size):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _tint(image, box, color, alpha):
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return
    region = image.crop((x0, y0, x1, y1))
    solid = Image.new("RGB", region.size, color)
    image.paste(Image.blend(region, solid, alpha), (x0, y0))


def _patch_box(top, left, height, width, patch_size):
    return (
        left * patch_size,
        top * patch_size,
        (left + width) * patch_size - 1,
        (top + height) * patch_size - 1,
    )


def _draw_grid(draw, image_size, patch_size, fill):
    for pos in range(0, image_size, patch_size):
        draw.line([(pos, 0), (pos, image_size - 1)], fill=fill)
        draw.line([(0, pos), (image_size - 1, pos)], fill=fill)


def _panel(title, image, font):
    header = 42
    canvas = Image.new("RGB", (image.width, header + image.height), (248, 246, 242))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill=(32, 32, 32), font=font)
    canvas.paste(image, (0, header))
    return canvas


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)

    image = Image.open(IMAGE_PATH).convert("RGB")
    if image.width != image.height or image.width % PATCH_SIZE != 0:
        raise SystemExit(
            f"{IMAGE_PATH.name} is {image.size}; expected a square multiple of {PATCH_SIZE}px"
        )

    grid = image.width // PATCH_SIZE
    patches = torch.zeros(1, grid, grid, 1)
    blocks = target_crop_patches_emb(patches, seed=SEED)
    print(f"{blocks=}")
    context_patches, keep_mask, context_crop = context_crop_patches(
        patches, blocks, seed=SEED + 1
    )

    kept = int(keep_mask.sum().item())
    if context_patches.shape != (1, kept, 1):
        raise SystemExit(
            f"context patches {tuple(context_patches.shape)} do not match keep_mask sum {kept}"
        )

    crop_h, crop_w, crop_top, crop_left = context_crop
    block_rows = blocks.tolist()

    grid_view = image.copy()
    grid_draw = ImageDraw.Draw(grid_view)
    _draw_grid(grid_draw, image.width, PATCH_SIZE, (255, 255, 255))

    overlay = image.copy()
    _tint(
        overlay,
        _patch_box(crop_top, crop_left, crop_h, crop_w, PATCH_SIZE),
        CONTEXT_COLOR,
        0.22,
    )
    for color, (block_h, block_w, top, left) in zip(TARGET_COLORS, block_rows):
        _tint(overlay, _patch_box(top, left, block_h, block_w, PATCH_SIZE), color, 0.45)

    overlay_draw = ImageDraw.Draw(overlay)
    _draw_grid(overlay_draw, image.width, PATCH_SIZE, (255, 255, 255))
    overlay_draw.rectangle(
        _patch_box(crop_top, crop_left, crop_h, crop_w, PATCH_SIZE),
        outline=CONTEXT_COLOR,
        width=4,
    )
    label_font = _font(18)
    for index, (color, (block_h, block_w, top, left)) in enumerate(
        zip(TARGET_COLORS, block_rows)
    ):
        box = _patch_box(top, left, block_h, block_w, PATCH_SIZE)
        overlay_draw.rectangle(box, outline=color, width=4)
        overlay_draw.text((box[0] + 6, box[1] + 4), f"B_{index + 1}", fill=color, font=label_font)

    overlay_draw.text(
        (crop_left * PATCH_SIZE + 6, crop_top * PATCH_SIZE + 4),
        "A",
        fill=CONTEXT_COLOR,
        font=label_font,
    )

    kept_view = Image.blend(image, Image.new("RGB", image.size, (18, 18, 18)), 1.0 - OUTSIDE_DIM)
    for row in range(grid):
        for col in range(grid):
            if not bool(keep_mask[row, col]):
                continue
            box = (
                col * PATCH_SIZE,
                row * PATCH_SIZE,
                (col + 1) * PATCH_SIZE,
                (row + 1) * PATCH_SIZE,
            )
            kept_view.paste(image.crop(box), box[:2])
    kept_draw = ImageDraw.Draw(kept_view)
    kept_draw.rectangle(
        _patch_box(crop_top, crop_left, crop_h, crop_w, PATCH_SIZE),
        outline=CONTEXT_COLOR,
        width=4,
    )

    title_font = _font(20)
    note_font = _font(16)
    panels = [
        _panel("1  Patch grid", grid_view, title_font),
        _panel("2  Context mask A, target masks B", overlay, title_font),
        _panel("3  Encoder input  A minus B", kept_view, title_font),
    ]

    gap = 18
    margin = 18
    note_height = 28 + 22 * max(1, len(block_rows))
    width = margin * 2 + gap * (len(panels) - 1) + sum(panel.width for panel in panels)
    height = margin + panels[0].height + note_height + margin
    canvas = Image.new("RGB", (width, height), (248, 246, 242))
    draw = ImageDraw.Draw(canvas)

    x = margin
    for panel in panels:
        canvas.paste(panel, (x, margin))
        x += panel.width + gap

    notes = [
        f"grid {grid}x{grid}   patch {PATCH_SIZE}px   seed {SEED}   "
        f"A context mask [h={crop_h}, w={crop_w}, top={crop_top}, left={crop_left}]   "
        f"kept {kept}/{grid * grid}",
    ]
    for index, (block_h, block_w, top, left) in enumerate(block_rows):
        notes.append(
            f"B_{index + 1} target mask  [h={block_h}, w={block_w}, top={top}, left={left}]  "
            f"area {block_h * block_w}"
        )
    y = margin + panels[0].height + 10
    for line in notes:
        draw.text((margin, y), line, fill=(48, 48, 48), font=note_font)
        y += 22

    canvas.save(OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
