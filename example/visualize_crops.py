"""Draw target and context mask steps on example/lenna.png.

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

from src.masks import pick_context_patches, pick_target_patches

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
PAPER = (248, 246, 242)
INK = (32, 32, 32)
NOTE = (48, 48, 48)


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
    canvas = Image.new("RGB", (image.width, header + image.height), PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill=INK, font=font)
    canvas.paste(image, (0, header))
    return canvas


def _bounds(indices, grid):
    rows = torch.div(indices, grid, rounding_mode="floor")
    cols = indices % grid
    top = int(rows.min())
    left = int(cols.min())
    return int(rows.max()) - top + 1, int(cols.max()) - left + 1, top, left


def _rectangle_indices(top, left, height, width, side):
    rows = torch.arange(top, top + height)
    cols = torch.arange(left, left + width)
    grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing="ij")
    return (grid_rows * side + grid_cols).reshape(-1)


def _sample_context_rectangle(side, seed, aspect_ratio=(1.0, 1.0), scale=(0.85, 1.0)):
    device = torch.device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    scale_crop = torch.empty((), device=device).uniform_(*scale, generator=generator).item()
    aspect_ratio_crop = (
        torch.empty((), device=device).uniform_(*aspect_ratio, generator=generator).item()
    )
    crop_h = min(side, max(1, round(side * scale_crop)))
    crop_w = min(side, max(1, round(side * scale_crop * aspect_ratio_crop)))
    top = torch.randint(0, side - crop_h + 1, (), device=device, generator=generator).item()
    left = torch.randint(0, side - crop_w + 1, (), device=device, generator=generator).item()
    return crop_h, crop_w, top, left


def _overlap_box(block, crop):
    block_h, block_w, block_top, block_left = block
    crop_h, crop_w, crop_top, crop_left = crop
    top = max(block_top, crop_top)
    left = max(block_left, crop_left)
    bottom = min(block_top + block_h, crop_top + crop_h)
    right = min(block_left + block_w, crop_left + crop_w)
    if bottom <= top or right <= left:
        return None
    return bottom - top, right - left, top, left


def _restore(view, source, cells, patch_size):
    grid = cells.shape[0]
    for row in range(grid):
        for col in range(grid):
            if not bool(cells[row, col]):
                continue
            box = (
                col * patch_size,
                row * patch_size,
                (col + 1) * patch_size,
                (row + 1) * patch_size,
            )
            view.paste(source.crop(box), box[:2])
    return view


def _target_view(image, blocks, newest):
    view = image.copy()
    for color, (block_h, block_w, top, left) in zip(TARGET_COLORS, blocks):
        _tint(view, _patch_box(top, left, block_h, block_w, PATCH_SIZE), color, 0.45)

    draw = ImageDraw.Draw(view)
    _draw_grid(draw, image.width, PATCH_SIZE, (255, 255, 255))
    label_font = _font(18)
    for index, (color, (block_h, block_w, top, left)) in enumerate(zip(TARGET_COLORS, blocks)):
        box = _patch_box(top, left, block_h, block_w, PATCH_SIZE)
        draw.rectangle(box, outline=color, width=6 if newest is not None and index == newest else 4)
        draw.text((box[0] + 6, box[1] + 4), f"T_{index + 1}", fill=color, font=label_font)
    return view


def _context_view(image, crop, blocks, removed_count, wash, mark_latest=True):
    crop_h, crop_w, crop_top, crop_left = crop
    dark = Image.new("RGB", image.size, (18, 18, 18))
    view = Image.blend(image, dark, 1.0 - OUTSIDE_DIM)

    cells = torch.zeros(image.height // PATCH_SIZE, image.width // PATCH_SIZE, dtype=torch.bool)
    cells[crop_top : crop_top + crop_h, crop_left : crop_left + crop_w] = True
    for block in blocks[:removed_count]:
        overlap = _overlap_box(block, crop)
        if overlap is None:
            continue
        hole_h, hole_w, hole_top, hole_left = overlap
        cells[hole_top : hole_top + hole_h, hole_left : hole_left + hole_w] = False

    _restore(view, image, cells, PATCH_SIZE)
    if wash:
        _tint(view, _patch_box(crop_top, crop_left, crop_h, crop_w, PATCH_SIZE), CONTEXT_COLOR, 0.22)

    if mark_latest and removed_count:
        active = blocks[removed_count - 1]
        overlap = _overlap_box(active, crop)
        if overlap is not None:
            hole_h, hole_w, hole_top, hole_left = overlap
            hole_box = (
                hole_left * PATCH_SIZE,
                hole_top * PATCH_SIZE,
                (hole_left + hole_w) * PATCH_SIZE,
                (hole_top + hole_h) * PATCH_SIZE,
            )
            view.paste(image.crop(hole_box), hole_box[:2])
            _tint(view, _patch_box(hole_top, hole_left, hole_h, hole_w, PATCH_SIZE), TARGET_COLORS[removed_count - 1], 0.45)

    draw = ImageDraw.Draw(view)
    _draw_grid(draw, image.width, PATCH_SIZE, (255, 255, 255))
    draw.rectangle(
        _patch_box(crop_top, crop_left, crop_h, crop_w, PATCH_SIZE),
        outline=CONTEXT_COLOR,
        width=4,
    )
    label_font = _font(18)
    draw.text(
        (crop_left * PATCH_SIZE + 6, crop_top * PATCH_SIZE + 4),
        "A",
        fill=CONTEXT_COLOR,
        font=label_font,
    )
    for index, (color, block) in enumerate(zip(TARGET_COLORS, blocks[:removed_count])):
        overlap = _overlap_box(block, crop)
        if overlap is None:
            continue
        hole_h, hole_w, hole_top, hole_left = overlap
        box = _patch_box(hole_top, hole_left, hole_h, hole_w, PATCH_SIZE)
        draw.rectangle(box, outline=color, width=6 if mark_latest and index == removed_count - 1 else 4)
        draw.text((box[0] + 6, box[1] + 4), f"T_{index + 1}", fill=color, font=label_font)
    return view


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
    patches = torch.zeros(grid * grid, 3, 1, 1)
    target_indices = pick_target_patches(patches, seed=SEED)
    context_indices = pick_context_patches(patches, target_indices, seed=SEED + 1)

    blocks = [_bounds(indices, grid) for indices in target_indices]
    crop_h, crop_w, crop_top, crop_left = _sample_context_rectangle(grid, SEED + 1)
    crop = (crop_h, crop_w, crop_top, crop_left)
    initial_indices = _rectangle_indices(crop_top, crop_left, crop_h, crop_w, grid)
    replayed = initial_indices
    for indices in target_indices:
        replayed = replayed[~torch.isin(replayed, indices)]
    if not torch.equal(replayed, context_indices):
        raise SystemExit("context replay does not match pick_context_patches")

    grid_view = image.copy()
    grid_draw = ImageDraw.Draw(grid_view)
    _draw_grid(grid_draw, image.width, PATCH_SIZE, (255, 255, 255))

    number_view = image.copy()
    number_draw = ImageDraw.Draw(number_view)
    _draw_grid(number_draw, image.width, PATCH_SIZE, (255, 255, 255))
    number_font = _font(16)
    for row in range(grid):
        for col in range(grid):
            text = str(row * grid + col)
            box = number_draw.textbbox((0, 0), text, font=number_font)
            text_w = box[2] - box[0]
            text_h = box[3] - box[1]
            x = col * PATCH_SIZE + (PATCH_SIZE - text_w) / 2
            y = row * PATCH_SIZE + (PATCH_SIZE - text_h) / 2 - 1
            number_draw.text(
                (x, y),
                text,
                fill=(255, 255, 255),
                font=number_font,
                stroke_width=2,
                stroke_fill=(20, 20, 20),
            )

    title_font = _font(20)
    kept = int(context_indices.numel())
    panels = [
        _panel("1  Patch grid", grid_view, title_font),
        _panel("2  Patch numbers", number_view, title_font),
        _panel("3  Target masks", _target_view(image, blocks, None), title_font),
        _panel(
            f"4  Initial context mask {crop_h * crop_w} patches",
            _context_view(image, crop, blocks, 0, wash=True),
            title_font,
        ),
        _panel(
            f"5  Context mask without targets kept {kept}",
            _context_view(image, crop, blocks, len(blocks), wash=False, mark_latest=False),
            title_font,
        ),
    ]

    gap = 18
    margin = 18
    columns = 3
    rows = [panels[index : index + columns] for index in range(0, len(panels), columns)]
    row_width = margin * 2 + gap * (columns - 1) + sum(panel.width for panel in rows[0])
    note_font = _font(16)
    notes = [
        f"grid {grid}x{grid}   patch {PATCH_SIZE}px   seed {SEED}   kept {kept}/{grid * grid}",
    ]
    for index, indices in enumerate(target_indices):
        listed = ", ".join(str(int(value)) for value in indices.tolist())
        notes.append(f"Target mask {index + 1}: {listed}")
    note_height = 16 + 22 * len(notes)
    height = margin + rows[0][0].height * 2 + gap + note_height + margin
    canvas = Image.new("RGB", (row_width, height), PAPER)
    draw = ImageDraw.Draw(canvas)

    y = margin
    for row in rows:
        x = margin
        for panel in row:
            canvas.paste(panel, (x, y))
            x += panel.width + gap
        y += row[0].height + gap

    y += 4
    for line in notes:
        draw.text((margin, y), line, fill=NOTE, font=note_font)
        y += 22

    canvas.save(OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
