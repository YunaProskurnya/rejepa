"""Lightweight linear-probe monitoring, shared by training and checkpoint evaluation.

Do not run this module as a script. Use scripts/train.py for scheduled evaluation
or scripts/evaluate_linear.py --mode monitor for an already saved checkpoint.
The slower final evaluation lives in src/linear_probe_final.py.
"""

import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def make_probe_loaders(train_path, val_path, *, class_names=None, image_size=224,
                       batch_size=128, num_workers=0, seed=0, pin_memory=False):
    """Use all selected images, deterministic crops and ordinary (images, labels) batches."""
    if Path(train_path).resolve() == Path(val_path).resolve():
        raise ValueError("linear probe requires distinct train and val directories")
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224),
                          interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(image_size), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train = datasets.ImageFolder(train_path, transform=transform)
    val = datasets.ImageFolder(val_path, transform=transform)
    if train.class_to_idx != val.class_to_idx:
        raise ValueError("linear probe train/val class_to_idx must match")
    classes = list(class_names) if class_names is not None else train.classes
    if not classes or len(set(classes)) != len(classes) or not set(classes).issubset(train.classes):
        raise ValueError("probe classes must be unique class names present in train and val")
    for dataset in (train, val):
        mapping = {dataset.class_to_idx[name]: index for index, name in enumerate(classes)}
        dataset.samples = [(path, mapping[label]) for path, label in dataset.samples if label in mapping]
        dataset.imgs = dataset.samples
        dataset.targets = [label for _, label in dataset.samples]
        dataset.classes = classes
        dataset.class_to_idx = {name: index for index, name in enumerate(classes)}
    options = dict(batch_size=batch_size, num_workers=num_workers, shuffle=False,
                   drop_last=False, pin_memory=pin_memory)
    return tuple(DataLoader(dataset, generator=torch.Generator().manual_seed(seed), **options)
                 for dataset in (train, val))


@torch.no_grad()
def extract_features(encoder, loader, device, precision="fp32"):
    """[B, C, H, W] -> patch features [B, N, D] -> image features [B, D], cached on CPU."""
    features, labels = [], []
    for images, targets in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            tokens = encoder(images.to(device, non_blocking=True))
            pooled = tokens.float().mean(dim=1)
        if not torch.isfinite(pooled).all():
            raise RuntimeError("non-finite linear probe features")
        features.append(pooled.cpu())
        labels.append(targets.cpu())
    if not features:
        raise ValueError("linear probe loaders must be nonempty")
    return torch.cat(features), torch.cat(labels)


def fit_linear_head(features, labels, num_classes, device, *, epochs, lr, batch_size, seed):
    """Fit W and b in logits = features @ W.T + b; the encoder is not involved."""
    head = nn.Linear(features.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(features), generator=generator)
        for indices in order.split(batch_size):
            x = features[indices].to(device)
            y = labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite linear probe loss")
            loss.backward()
            optimizer.step()
    return head


@torch.no_grad()
def score_linear_head(head, features, labels, device, batch_size):
    """Evaluate the final head, including on train; do not average training-time accuracies."""
    head.eval()
    loss_sum, correct = 0.0, 0
    for start in range(0, len(features), batch_size):
        x = features[start:start + batch_size].to(device)
        y = labels[start:start + batch_size].to(device)
        logits = head(x)
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite linear probe predictions")
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=1) == y).sum().item()
    return dict(loss=loss_sum / len(features), top1=100 * correct / len(features))


def evaluate_linear_probe(encoder, train_loader, val_loader, num_classes, device,
                          epochs=30, lr=1e-3, *, batch_size=256, seed=0, precision="fp32"):
    """Return separate train/val metrics without updating the encoder or training RNG.

    For EMA I-JEPA pass target_encoder; for joint training pass the shared encoder.
    Both loaders must use deterministic transforms and no masking. A fresh head
    with the same seed is fitted each call. Val labels are used only for scoring.
    """
    if epochs <= 0 or batch_size <= 0 or num_classes <= 0 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("epochs, batch_size, num_classes and lr must be positive")
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    device = torch.device(device)
    modes = [(module, module.training) for module in encoder.modules()]
    generators = {loader.generator for loader in (train_loader, val_loader)
                  if loader.generator is not None}
    generator_states = [(generator, generator.get_state()) for generator in generators]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            encoder.eval()
            train_features, train_labels = extract_features(encoder, train_loader, device, precision)
            val_features, val_labels = extract_features(encoder, val_loader, device, precision)
            for labels in (train_labels, val_labels):
                if labels.min() < 0 or labels.max() >= num_classes:
                    raise ValueError("probe labels must lie in [0, num_classes)")
            head = fit_linear_head(train_features, train_labels, num_classes, device,
                                   epochs=epochs, lr=lr, batch_size=batch_size, seed=seed)
            train_metrics = score_linear_head(head, train_features, train_labels, device, batch_size)
            val_metrics = score_linear_head(head, val_features, val_labels, device, batch_size)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    finally:
        for generator, state in generator_states:
            generator.set_state(state)
        for module, training in modes:
            module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    return dict(train_loss=train_metrics["loss"], train_top1=train_metrics["top1"],
                val_loss=val_metrics["loss"], val_top1=val_metrics["top1"],
                train_images=len(train_labels), val_images=len(val_labels),
                probe_seconds=time.perf_counter() - started)
