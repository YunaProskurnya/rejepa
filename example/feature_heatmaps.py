import argparse
import json
from pathlib import Path
import sys

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.encoder import VisionTransformer
from src.transforms import IMAGENET_NORMALIZE, make_evaluation_transform


def pca_rgb(features, grid):
    if features.ndim != 2 or features.shape[0] != grid * grid:
        raise ValueError("Expected one feature vector per patch")
    if min(features.shape) < 3 or not torch.isfinite(features).all():
        raise ValueError("PCA needs finite features with at least three patches and channels")
    centered = features.double() - features.double().mean(dim=0)
    _, singular_values, directions = torch.linalg.svd(centered, full_matrices=False)
    basis = directions[:3].T
    signs = basis[basis.abs().argmax(dim=0), torch.arange(3)].sign()
    scores = centered @ (basis * signs)
    low, high = scores.amin(dim=0), scores.amax(dim=0)
    rgb = ((scores - low) / (high - low).clamp_min(1e-12)).clamp(0, 1)
    assert torch.isfinite(rgb).all() and rgb.shape == (grid * grid, 3)
    variance = singular_values[:3].square().sum() / singular_values.square().sum().clamp_min(1e-12)
    return rgb.reshape(grid, grid, 3).permute(2, 0, 1).float(), variance.item()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="RGB PCA maps of final encoder patch features")
    parser.add_argument("--checkpoints", nargs="+", type=Path, default=[
        Path("/home/container/shared/rejepa/runs/imagenet100-vit-base-400ep-visreg/checkpoint.pt"),
        Path("/home/container/shared/rejepa/runs/imagenet100-vit-base-600ep/checkpoint_349.pt"),
    ])
    parser.add_argument("--images", nargs="+", type=Path,
                        default=[ROOT / "example/lenna.png", ROOT / "example/yann.png"])
    parser.add_argument("--output", type=Path, default=ROOT / "example/feature_heatmaps")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    sources = [Image.open(path).convert("RGB") for path in args.images]
    args.output.mkdir(parents=True)
    panel = 336
    row_height = panel + 56
    overview = Image.new("RGB", ((1 + 2 * len(args.checkpoints)) * panel,
                                 len(sources) * row_height + 64), "white")
    draw = ImageDraw.Draw(overview)
    draw.text((12, 8), "Final patch features: PC1 / PC2 / PC3 -> R / G / B", fill="black")
    draw.text((12, 28), "Independent PCA per map; colors are not aligned across models. Bilinear display; no background mask.", fill="black")
    metadata = dict(device=args.device, torch_version=str(torch.__version__),
                    preprocessing="evaluation resize (256/224), center crop, ImageNet normalization",
                    method="Centered PCA per image/model; component sign fixed by largest loading; per-channel min-max RGB",
                    maps=[])
    saved = {}
    column = 1
    for checkpoint_index, path in enumerate(args.checkpoints):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        options = checkpoint["config"]["encoder"]
        transform = make_evaluation_transform(options["image_size"])
        images = torch.stack([transform(source) for source in sources])
        grid = options["image_size"] // options["patch_size"]
        label = path.parent.name.removeprefix("imagenet100-vit-base-")
        mean = torch.tensor(IMAGENET_NORMALIZE["mean"]).view(3, 1, 1)
        std = torch.tensor(IMAGENET_NORMALIZE["std"]).view(3, 1, 1)
        if checkpoint_index == 0:
            for index, tensor in enumerate(images):
                crop = to_pil_image((tensor * std + mean).clamp(0, 1))
                crop.save(args.output / f"{args.images[index].stem}_input.png")
                overview.paste(crop.resize((panel, panel)), (0, index * row_height + 120))
                draw.text((12, index * row_height + 80), args.images[index].name + " | input crop", fill="black")
        else:
            assert torch.equal(images, saved["inputs"]), "Checkpoints require different input crops"
        saved["inputs"] = images
        encoder = VisionTransformer(**options).to(args.device).eval().requires_grad_(False)
        model_names = ("encoder", "target_encoder") if "target_encoder" in checkpoint else ("encoder",)
        for model_name in model_names:
            encoder.load_state_dict(checkpoint[model_name], strict=True)
            features = encoder(images.to(args.device)).cpu().float()
            assert features.shape == (len(sources), grid * grid, options["embed_dim"])
            saved[f"{checkpoint_index}_{model_name}"] = features
            for index, tokens in enumerate(features):
                rgb, variance = pca_rgb(tokens, grid)
                stem = f"{args.images[index].stem}_{label}_{path.stem}_{model_name}"
                raw = to_pil_image(rgb)
                raw.save(args.output / f"{stem}_patches.png")
                display = raw.resize((panel, panel), Image.Resampling.BILINEAR)
                display.save(args.output / f"{stem}.png")
                x = column * panel
                y = index * row_height + 80
                draw.text((x + 8, y), f"{label} | {model_name}", fill="black")
                draw.text((x + 8, y + 18), f"epoch={checkpoint.get('epoch')} | {grid}x{grid} patches | PCA={variance:.1%}", fill="black")
                overview.paste(display, (x, y + 40))
                metadata["maps"].append(dict(image=str(args.images[index]), checkpoint=str(path),
                                            model=model_name, epoch=checkpoint.get("epoch"),
                                            step=checkpoint.get("step"), encoder=options,
                                            pca_explained_variance=variance, output=f"{stem}.png"))
                print(f"{stem}: features={tuple(tokens.shape)}, PCA variance={variance:.3%}", flush=True)
            column += 1
        del encoder, checkpoint
    overview.crop((0, 0, column * panel, overview.height)).save(args.output / "comparison.png")
    torch.save(saved, args.output / "features.pt")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved results to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
