"""Draw the image pipeline and the context and target masks.

Run from the repo root:

    python example/visualize.py
"""

import sys
from pathlib import Path

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.io import ImageReadMode, read_image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.masks import pick_context_patches, pick_target_patches
from src.transforms import make_transform, process_image

IMAGE_PATH = REPO_ROOT / "example" / "lenna.png"
OUTPUT_PATH = REPO_ROOT / "example" / "lenna_pipeline.png"
PANEL = 448
PATCH_NUM = 196
SEED = 0

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MEAN_TENSOR = torch.tensor(MEAN)[:, None, None]
STD_TENSOR = torch.tensor(STD)[:, None, None]

TARGET_COLORS = (
    (214, 69, 80),
    (232, 149, 47),
    (66, 116, 214),
    (142, 84, 196),
)
CONTEXT_COLOR = (36, 168, 104)
CROP_COLOR = (214, 69, 80)
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


def _panel(title, image, font):
    header = 42
    canvas = Image.new("RGB", (image.width, header + image.height), PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill=INK, font=font)
    canvas.paste(image, (0, header))
    return canvas


def _tint(image, box, color, alpha):
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return
    region = image.crop((x0, y0, x1, y1))
    solid = Image.new("RGB", region.size, color)
    image.paste(Image.blend(region, solid, alpha), (x0, y0))


def _draw_grid(draw, image_size, patch_size, fill):
    for pos in range(0, image_size, patch_size):
        draw.line([(pos, 0), (pos, image_size - 1)], fill=fill)
        draw.line([(0, pos), (image_size - 1, pos)], fill=fill)


def _patch_box(top, left, height, width, patch_size):
    return (
        left * patch_size,
        top * patch_size,
        (left + width) * patch_size - 1,
        (top + height) * patch_size - 1,
    )


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


def _sample_context_rectangle(
    side,
    seed,
    target_indices,
    aspect_ratio=(1.0, 1.0),
    scale=(0.85, 1.0),
    min_keep=10,
    allow_overlap=False,
):
    device = torch.device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    scale_min, scale_max = scale
    ratio_min, ratio_max = aspect_ratio
    u = torch.rand((), device=device, generator=generator).item()
    scale = scale_min + u * (scale_max - scale_min)
    ratio = ratio_min + u * (ratio_max - ratio_min)
    area = int(side * side * scale)
    block_h = min(round((area * ratio) ** 0.5), side - 1)
    block_w = min(round((area / ratio) ** 0.5), side - 1)

    if allow_overlap:
        exclusions = []
    else:
        exclusions = list(target_indices)
    dropped = 0
    attempts = 0
    while True:
        top = torch.randint(0, side - block_h, (), device=device, generator=generator).item()
        left = torch.randint(0, side - block_w, (), device=device, generator=generator).item()
        kept = _rectangle_indices(top, left, block_h, block_w, side)
        active = exclusions[: max(len(exclusions) - dropped, 0)]
        if active:
            banned = torch.cat(active)
            kept = kept[~torch.isin(kept, banned)]
        if kept.numel() > min_keep:
            return block_h, block_w, top, left, kept, len(active)
        attempts += 1
        if attempts == 20:
            dropped += 1
            attempts = 0


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


def _load_source():
    tensor = read_image(str(IMAGE_PATH), mode=ImageReadMode.RGB)
    return T.ToPILImage()(tensor)


def _run(source, seed):
    torch.manual_seed(seed)
    current = source
    crop_box = None
    stages = {}

    for op in make_transform().transforms:
        if isinstance(op, T.RandomResizedCrop):
            top, left, height, width = op.get_params(current, op.scale, op.ratio)
            crop_box = (top, left, height, width)
            current = F.resized_crop(
                current,
                top,
                left,
                height,
                width,
                op.size,
                op.interpolation,
                antialias=op.antialias,
            )
        else:
            current = op(current)
        stages[type(op).__name__] = current

    return crop_box, stages


def _denorm(tensor):
    return tensor * STD_TENSOR + MEAN_TENSOR


def _show_pil(image):
    return image.resize((PANEL, PANEL), Image.Resampling.NEAREST)


def _show_denormalized(tensor):
    image = T.ToPILImage()(_denorm(tensor).clamp(0, 1))
    return image.resize((PANEL, PANEL), Image.Resampling.NEAREST)


def _normalize_view(tensor):
    shown = (tensor * 0.25 + 0.5).clamp(0, 1)
    return T.ToPILImage()(shown).resize((PANEL, PANEL), Image.Resampling.NEAREST)


def _number_view(image, patch_size, grid):
    view = image.copy()
    draw = ImageDraw.Draw(view)
    _draw_grid(draw, view.width, patch_size, (255, 255, 255))
    number_font = _font(11)
    for row in range(grid):
        for col in range(grid):
            text = str(row * grid + col)
            box = draw.textbbox((0, 0), text, font=number_font)
            text_w = box[2] - box[0]
            text_h = box[3] - box[1]
            x = col * patch_size + (patch_size - text_w) / 2
            y = row * patch_size + (patch_size - text_h) / 2 - 1
            draw.text(
                (x, y),
                text,
                fill=(255, 255, 255),
                font=number_font,
                stroke_width=2,
                stroke_fill=(20, 20, 20),
            )
    return view


def _grid_view(image, patch_size):
    view = image.copy()
    draw = ImageDraw.Draw(view)
    _draw_grid(draw, view.width, patch_size, (255, 255, 255))
    return view


def _mask_view(image, blocks, crop, patch_size, excluded=None):
    if excluded is None:
        excluded = blocks
    crop_h, crop_w, crop_top, crop_left = crop
    dark = Image.new("RGB", image.size, (18, 18, 18))
    view = Image.blend(image, dark, 1.0 - OUTSIDE_DIM)

    grid = image.height // patch_size
    cells = torch.zeros(grid, grid, dtype=torch.bool)
    cells[crop_top : crop_top + crop_h, crop_left : crop_left + crop_w] = True
    for block in excluded:
        overlap = _overlap_box(block, crop)
        if overlap is None:
            continue
        hole_h, hole_w, hole_top, hole_left = overlap
        cells[hole_top : hole_top + hole_h, hole_left : hole_left + hole_w] = False

    _restore(view, image, cells, patch_size)
    _tint(view, _patch_box(crop_top, crop_left, crop_h, crop_w, patch_size), CONTEXT_COLOR, 0.18)
    for color, (block_h, block_w, top, left) in zip(TARGET_COLORS, blocks):
        _tint(view, _patch_box(top, left, block_h, block_w, patch_size), color, 0.45)

    draw = ImageDraw.Draw(view)
    _draw_grid(draw, image.width, patch_size, (255, 255, 255))
    draw.rectangle(
        _patch_box(crop_top, crop_left, crop_h, crop_w, patch_size),
        outline=CONTEXT_COLOR,
        width=4,
    )
    label_font = _font(18)
    draw.text(
        (crop_left * patch_size + 6, crop_top * patch_size + 4),
        "Context",
        fill=CONTEXT_COLOR,
        font=label_font,
    )
    for index, (color, (block_h, block_w, top, left)) in enumerate(zip(TARGET_COLORS, blocks)):
        box = _patch_box(top, left, block_h, block_w, patch_size)
        draw.rectangle(box, outline=color, width=4)
        draw.text((box[0] + 4, box[1] + 2), f"T_{index + 1}", fill=color, font=label_font)
    return view


def _context_holes_view(image, crop, blocks, patch_size):
    crop_h, crop_w, crop_top, crop_left = crop
    dark = Image.new("RGB", image.size, (18, 18, 18))
    view = Image.blend(image, dark, 1.0 - OUTSIDE_DIM)

    grid = image.height // patch_size
    cells = torch.zeros(grid, grid, dtype=torch.bool)
    cells[crop_top : crop_top + crop_h, crop_left : crop_left + crop_w] = True
    for block in blocks:
        overlap = _overlap_box(block, crop)
        if overlap is None:
            continue
        hole_h, hole_w, hole_top, hole_left = overlap
        cells[hole_top : hole_top + hole_h, hole_left : hole_left + hole_w] = False

    _restore(view, image, cells, patch_size)

    draw = ImageDraw.Draw(view)
    _draw_grid(draw, image.width, patch_size, (255, 255, 255))
    draw.rectangle(
        _patch_box(crop_top, crop_left, crop_h, crop_w, patch_size),
        outline=CONTEXT_COLOR,
        width=4,
    )
    label_font = _font(18)
    draw.text(
        (crop_left * patch_size + 6, crop_top * patch_size + 4),
        "Context",
        fill=CONTEXT_COLOR,
        font=label_font,
    )
    for index, (color, block) in enumerate(zip(TARGET_COLORS, blocks)):
        overlap = _overlap_box(block, crop)
        if overlap is None:
            continue
        hole_h, hole_w, hole_top, hole_left = overlap
        box = _patch_box(hole_top, hole_left, hole_h, hole_w, patch_size)
        draw.rectangle(box, outline=color, width=4)
        draw.text((box[0] + 6, box[1] + 4), f"T_{index + 1}", fill=color, font=label_font)
    return view


def _wrap(draw, text, font, width):
    words = text.split()
    lines = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if draw.textlength(trial, font=font) <= width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def main():
    source = _load_source()
    crop_box, stages = _run(source, SEED)
    top, left, height, width = crop_box
    area = height * width / (source.height * source.width)

    torch.manual_seed(SEED)
    returned, target_indices, context_indices = process_image(str(IMAGE_PATH))
    normalized = stages["Normalize"]
    if not torch.equal(returned, normalized):
        raise SystemExit("process_image output does not match the staged transform")

    denorm_error = (_denorm(normalized) - stages["ToTensor"]).abs().max().item()
    if denorm_error > 1e-5:
        raise SystemExit(f"denormalize error {denorm_error}")

    grid = normalized.shape[-1] // 16
    patch_size = normalized.shape[-1] // grid
    if (grid, patch_size, grid * grid) != (14, 16, PATCH_NUM):
        raise SystemExit(f"unexpected patch grid {grid}x{grid} patch {patch_size}")

    fresh_targets = pick_target_patches(grid, seed=SEED)
    fresh_context = pick_context_patches(grid, fresh_targets, seed=SEED)
    for got, expected in zip(target_indices, fresh_targets):
        if not torch.equal(got, expected):
            raise SystemExit("process_image targets do not match pick_target_patches")
    if not torch.equal(context_indices, fresh_context):
        raise SystemExit("process_image context does not match pick_context_patches")

    blocks = [_bounds(indices, grid) for indices in target_indices]
    crop_h, crop_w, crop_top, crop_left, replayed, n_excluded = _sample_context_rectangle(
        grid, SEED, target_indices
    )
    crop = (crop_h, crop_w, crop_top, crop_left)
    excluded_blocks = blocks[:n_excluded]
    for indices in target_indices[:n_excluded]:
        if torch.isin(context_indices, indices).any():
            raise SystemExit("context still contains an excluded target patch")
    if not torch.equal(replayed, context_indices):
        raise SystemExit("context replay does not match pick_context_patches")

    scale = PANEL / source.width
    source_view = source.resize((PANEL, PANEL), Image.Resampling.BICUBIC)
    crop_view = source_view.copy()
    crop_draw = ImageDraw.Draw(crop_view)
    crop_draw.rectangle(
        [
            round(left * scale),
            round(top * scale),
            round((left + width) * scale) - 1,
            round((top + height) * scale) - 1,
        ],
        outline=CROP_COLOR,
        width=4,
    )

    display_patch = PANEL // grid
    model = _normalize_view(normalized)
    kept = int(context_indices.numel())
    title_font = _font(20)
    panels = [
        _panel("1  Source RGB", source_view, title_font),
        _panel("2  Random resized crop", crop_view, title_font),
        _panel("3  Resized to 224", _show_pil(stages["RandomResizedCrop"]), title_font),
        _panel("4  Normalize", _normalize_view(normalized), title_font),
        _panel("5  Patch numbers", _number_view(_show_denormalized(normalized), display_patch, grid), title_font),
        _panel("6  Patch grid, normalized", _grid_view(model, display_patch), title_font),
        _panel(
            f"7  Masks on grid  kept {kept}",
            _mask_view(model, blocks, crop, display_patch, excluded_blocks),
            title_font,
        ),
        _panel(
            f"8  Context without targets  kept {kept}",
            _context_holes_view(model, crop, excluded_blocks, display_patch),
            title_font,
        ),
    ]

    gap = 18
    margin = 18
    columns = 4
    means = normalized.mean(dim=(1, 2))
    notes = [
        (
            f"seed {SEED}   crop {width}x{height} at left {left} top {top}"
            f"   area {area:.0%}   output 224x224"
        ),
        (
            f"tensor {tuple(normalized.shape)}"
            f"   normalize mean {MEAN[0]} {MEAN[1]} {MEAN[2]}"
            f"   std {STD[0]} {STD[1]} {STD[2]}"
        ),
        (
            f"normalized channel mean {means[0]:.3f} {means[1]:.3f} {means[2]:.3f}"
            f"   denorm error {denorm_error:.1e}"
        ),
        (
            f"grid {grid}x{grid}   patch {patch_size}px   masks seed {SEED}"
            f"   context {crop_h}x{crop_w} at left {crop_left} top {crop_top}"
            f"   kept {kept}/{grid * grid}"
        ),
    ]
    for index, indices in enumerate(target_indices):
        listed = ", ".join(str(int(value)) for value in indices.tolist())
        block_h, block_w, block_top, block_left = blocks[index]
        notes.append(
            f"Target mask {index + 1}: {block_h}x{block_w} at left {block_left} top {block_top}   {listed}"
        )

    rows = [panels[index : index + columns] for index in range(0, len(panels), columns)]
    row_width = margin * 2 + gap * (columns - 1) + sum(panel.width for panel in rows[0])
    note_font = _font(16)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    wrapped = []
    for line in notes:
        wrapped.extend(_wrap(measure, line, note_font, row_width - margin * 2))
    note_height = 4 + 18 * len(wrapped)
    height_px = margin + sum(row[0].height for row in rows) + gap * len(rows) + note_height + margin
    canvas = Image.new("RGB", (row_width, height_px), PAPER)
    draw = ImageDraw.Draw(canvas)

    y = margin
    for row in rows:
        x = margin
        for panel in row:
            canvas.paste(panel, (x, y))
            x += panel.width + gap
        y += row[0].height + gap

    for line in wrapped:
        draw.text((margin, y), line, fill=NOTE, font=note_font)
        y += 18

    canvas.save(OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
