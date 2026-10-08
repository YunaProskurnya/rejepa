import argparse
import logging
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.losses import prediction_loss
from src.models.encoder import VisionTransformer
from src.models.predictor import VisionTransformerPredictor
from src.models.target_encoder import (
    create_target_encoder,
    prepare_targets,
    update_target_encoder,
)


def main():
    parser = argparse.ArgumentParser(description="Single-image I-JEPA training smoke test")
    parser.add_argument("--image", type=Path, default=ROOT / "example" / "lenna.png")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")

    logging.basicConfig(level=logging.WARNING)
    from src.transforms import process_image

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))

    image, target_indices, context_indices = process_image(str(args.image), seed=args.seed)
    images = image.unsqueeze(0).to(device)
    context_indices = context_indices.unsqueeze(0).to(device)
    target_indices = [indices.unsqueeze(0).to(device) for indices in target_indices]

    encoder = VisionTransformer(embed_dim=48, depth=2, num_heads=3).to(device)
    predictor = VisionTransformerPredictor(
        num_patches=encoder.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=24,
        depth=2,
        num_heads=3,
    ).to(device)
    target_encoder = create_target_encoder(encoder)

    decay, no_decay = [], []
    for module in (encoder, predictor):
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                group = no_decay if "bias" in name or parameter.ndim == 1 else decay
                group.append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.04},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=1e-3,
    )

    print(f"device={device} images={tuple(images.shape)}")
    print(f"context={tuple(context_indices.shape)} "
          f"targets={[tuple(indices.shape) for indices in target_indices]}")
    print("Fixed image and masks; tiny model; diagnostic run only.")

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)

        targets = prepare_targets(target_encoder, images, target_indices)
        context = encoder(images, context_indices)
        predictions = predictor(context, context_indices, target_indices)
        loss = prediction_loss(predictions, targets)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite loss at step {step}")

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            decay + no_decay, max_norm=float("inf"), error_if_nonfinite=True,
        )
        optimizer.step()
        update_target_encoder(encoder, target_encoder, momentum=0.996)

        print(f"step={step:03d} loss={loss.item():.6f} grad_norm={grad_norm.item():.6f}")


if __name__ == "__main__":
    main()
