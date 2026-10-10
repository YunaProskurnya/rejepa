"""Independent final I-JEPA linear evaluation; the epoch monitor lives separately.

Paper-level settings come from Appendix A.2. Feature/head/transformation details
follow public VISSL code; LARS follows the cited MAE implementation. The original
I-JEPA evaluation code and VISSL's custom linear_eval_heads groups are unreleased.
"""

import copy
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from .linear_probe import make_probe_loaders
from .reference.lars import LARS


def make_final_probe_loaders(train_path, val_path, *, class_names=None, image_size=224,
                             batch_size=128, num_workers=0, seed=0, pin_memory=False):
    """Return augmented training, deterministic training, and validation loaders."""
    train_eval, val = make_probe_loaders(
        train_path, val_path, class_names=class_names, image_size=image_size,
        batch_size=batch_size, num_workers=num_workers, seed=seed, pin_memory=pin_memory,
    )
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    deterministic = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224),
                          interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(image_size), transforms.ToTensor(), normalize,
    ])
    train_eval.dataset.transform = deterministic
    val.dataset.transform = deterministic
    augmented = copy.copy(train_eval.dataset)
    augmented.transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0),
                                     ratio=(0.75, 4 / 3),
                                     interpolation=transforms.InterpolationMode.BICUBIC,
                                     antialias=True),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,
    ])
    train = DataLoader(augmented, batch_size=batch_size, shuffle=True, drop_last=False,
                       num_workers=num_workers, pin_memory=pin_memory,
                       generator=torch.Generator().manual_seed(seed))
    return train, train_eval, val


def _loader_generators(loaders):
    generators = {}
    for loader in loaders:
        owners = (loader, getattr(loader, 'sampler', None),
                  getattr(getattr(loader, 'batch_sampler', None), 'sampler', None))
        for owner in owners:
            generator = getattr(owner, 'generator', None)
            if isinstance(generator, torch.Generator):
                generators[id(generator)] = generator
    return list(generators.values())


@torch.no_grad()
def _pooled_features(encoder, images, device, precision, use_last4):
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == 'bf16'):
        if use_last4:
            layers = encoder.get_intermediate_layers(images.to(device, non_blocking=True), n=4)
            pooled = [layer.float().mean(dim=1) for layer in layers]
            features = {'last': pooled[-1], 'last4': torch.cat(pooled, dim=-1)}
        else:
            tokens = encoder(images.to(device, non_blocking=True))
            features = {'last': tokens.float().mean(dim=1)}
    if any(not torch.isfinite(value).all() for value in features.values()):
        raise RuntimeError('non-finite final linear probe features')
    return {name: value.detach().cpu() for name, value in features.items()}


def _labels(targets, num_classes):
    targets = targets.detach().to(device='cpu', dtype=torch.long)
    if targets.ndim != 1 or not len(targets):
        raise ValueError('final probe labels must be a nonempty vector')
    if targets.min() < 0 or targets.max() >= num_classes:
        raise ValueError('final probe labels must lie in [0, num_classes)')
    return targets


def _feature_batches(encoder, loader, device, precision, use_last4, batch_size, num_classes,
                     stats):
    """Buffer only detached features; discard the final incomplete optimizer batch."""
    chunks, target_chunks, count = {}, [], 0
    for images, targets in loader:
        features = _pooled_features(encoder, images, device, precision, use_last4)
        targets = _labels(targets, num_classes)
        stats['images'] += len(targets)
        offset = 0
        while offset < len(targets):
            take = min(batch_size - count, len(targets) - offset)
            for name, values in features.items():
                chunks.setdefault(name, []).append(values[offset:offset + take])
            target_chunks.append(targets[offset:offset + take])
            count += take
            offset += take
            if count == batch_size:
                yield ({name: torch.cat(values) for name, values in chunks.items()},
                       torch.cat(target_chunks))
                chunks, target_chunks, count = {}, [], 0


def _make_trials(features, num_classes, device, batch_size, learning_rates, weight_decays):
    trials = []
    for representation, values in features.items():
        dimension = values.shape[1]
        for head_name in ('linear', 'batchnorm_linear'):
            for reference_lr in learning_rates:
                for weight_decay in weight_decays:
                    linear = nn.Linear(dimension, num_classes)
                    head = (nn.Sequential(nn.BatchNorm1d(dimension, affine=True, eps=1e-5,
                                                        momentum=0.1), linear)
                            if head_name == 'batchnorm_linear' else linear).to(device)
                    lr = reference_lr * batch_size / 256
                    trials.append(dict(
                        representation=representation, head=head_name,
                        feature_dim=dimension, reference_lr=reference_lr, lr=lr,
                        weight_decay=weight_decay, _head=head,
                        _optimizer=LARS(head.parameters(), lr=lr, weight_decay=weight_decay),
                    ))
    return trials


@torch.no_grad()
def _score_trials(encoder, trials, loader, num_classes, device, precision, use_last4):
    totals = [dict(loss=0.0, correct=0) for _ in trials]
    count = 0
    for trial in trials:
        trial['_head'].eval()
    for images, targets in loader:
        features = _pooled_features(encoder, images, device, precision, use_last4)
        targets = _labels(targets, num_classes).to(device)
        values = {name: value.to(device) for name, value in features.items()}
        count += len(targets)
        for trial, total in zip(trials, totals):
            logits = trial['_head'](values[trial['representation']])
            if not torch.isfinite(logits).all():
                raise RuntimeError('non-finite final linear probe predictions')
            total['loss'] += F.cross_entropy(logits, targets, reduction='sum').item()
            total['correct'] += (logits.argmax(dim=1) == targets).sum().item()
    if count == 0:
        raise ValueError('final probe scoring loaders must be nonempty')
    return [dict(loss=total['loss'] / count, top1=100 * total['correct'] / count,
                 images=count) for total in totals]


def evaluate_final_linear_probe(encoder, train_loader, train_eval_loader, val_loader,
                                num_classes, device, *, epochs=50, batch_size=16384,
                                seed=0, precision='fp32', learning_rates=(0.01, 0.05, 0.001),
                                weight_decays=(0.0005, 0.0), progress=None):
    """Fit fresh independent probes with shared encoder passes and report final scores.

    The training loader must draw fresh random crops. The two scoring loaders must
    use deterministic crops. Only validation scores select the returned best head;
    validation labels never participate in optimization. Encoder parameters and
    existing gradients are never written. Caller RNG and loader generators survive
    both successful evaluation and exceptions. Optional progress receives one dict
    per completed training epoch: epoch (one-based), epochs, steps, and seconds
    (elapsed for that epoch).
    """
    learning_rates, weight_decays = tuple(map(float, learning_rates)), tuple(map(float, weight_decays))
    if epochs <= 0 or batch_size < 2 or num_classes <= 0:
        raise ValueError('epochs and num_classes must be positive; batch_size must be at least 2')
    if not learning_rates or any(not math.isfinite(x) or x <= 0 for x in learning_rates):
        raise ValueError('learning_rates must contain positive finite reference rates')
    if not weight_decays or any(not math.isfinite(x) or x < 0 for x in weight_decays):
        raise ValueError('weight_decays must contain nonnegative finite values')
    if precision not in ('fp32', 'bf16'):
        raise ValueError('precision must be fp32 or bf16')
    if len(train_loader.dataset) < batch_size:
        raise ValueError('final probe training dataset is smaller than the effective head batch; '
                         'choose an explicit smaller batch_size for a reduced-data experiment')
    device = torch.device(device)
    use_last4 = (hasattr(encoder, 'get_intermediate_layers') and
                 len(getattr(encoder, 'blocks', ())) >= 4)
    modes = [(module, module.training) for module in encoder.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    generators = _loader_generators((train_loader, train_eval_loader, val_loader))
    generator_states = [(generator, generator.get_state()) for generator in generators]
    trials, epoch_batches, dropped_images = [], [], []
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            for generator in generators:
                generator.manual_seed(seed)
            encoder.eval()
            for epoch in range(epochs):
                epoch_started = time.perf_counter()
                for trial in trials:
                    for group in trial['_optimizer'].param_groups:
                        group['lr'] = trial['lr'] * 0.1 ** (epoch // 15)
                batches = 0
                stats = {'images': 0}
                for features, targets in _feature_batches(
                    encoder, train_loader, device, precision, use_last4, batch_size, num_classes,
                    stats,
                ):
                    if not trials:
                        trials = _make_trials(features, num_classes, device, batch_size,
                                              learning_rates, weight_decays)
                    values = {name: value.to(device) for name, value in features.items()}
                    targets = targets.to(device)
                    for trial in trials:
                        optimizer, head = trial['_optimizer'], trial['_head']
                        optimizer.zero_grad(set_to_none=True)
                        loss = F.cross_entropy(head(values[trial['representation']]), targets)
                        if not torch.isfinite(loss):
                            raise RuntimeError('non-finite final linear probe loss')
                        loss.backward()
                        optimizer.step()
                    batches += 1
                if not batches:
                    raise ValueError('final probe loader did not produce one full effective head batch')
                epoch_batches.append(batches)
                dropped_images.append(stats['images'] - batches * batch_size)
                if progress is not None:
                    progress(dict(epoch=epoch + 1, epochs=epochs, steps=batches,
                                  seconds=time.perf_counter() - epoch_started))
            train_scores = _score_trials(encoder, trials, train_eval_loader, num_classes,
                                         device, precision, use_last4)
            val_scores = _score_trials(encoder, trials, val_loader, num_classes,
                                       device, precision, use_last4)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
    finally:
        for module, training in modes:
            module.training = training
        for generator, state in generator_states:
            generator.set_state(state)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    results = []
    for trial, train, val in zip(trials, train_scores, val_scores):
        result = {key: value for key, value in trial.items() if not key.startswith('_')}
        result.update(train_loss=train['loss'], train_top1=train['top1'],
                      val_loss=val['loss'], val_top1=val['top1'],
                      train_images=train['images'], val_images=val['images'])
        results.append(result)
    best = max(results, key=lambda trial: (trial['val_top1'], -trial['val_loss']))
    encoder_norm = getattr(encoder, 'norm', None)
    protocol = dict(
        name='ijepa_final_linear', epochs=epochs, optimizer='MAE LARS',
        effective_batch_size=batch_size, reference_batch_size=256,
        learning_rates=list(learning_rates), weight_decays=list(weight_decays),
        representations=['last', 'last4'] if use_last4 else ['last'],
        heads=['linear', 'batchnorm_linear'], lr_decay_epochs=[15, 30, 45],
        lr_decay_factor=0.1, warmup_epochs=0, momentum=0.9, trust_coefficient=0.001,
        batchnorm_affine=True, batchnorm_eps=1e-5, batchnorm_momentum=0.1,
        linear_initialization='torch.nn.Linear default', precision=precision, seed=seed,
        transform_repr={
            'train': repr(getattr(train_loader.dataset, 'transform', None)),
            'train_eval': repr(getattr(train_eval_loader.dataset, 'transform', None)),
            'val': repr(getattr(val_loader.dataset, 'transform', None)),
        },
        feature_pooling='arithmetic mean over spatial patch tokens',
        last4_combination=('concatenate pooled features from the last four blocks, '
                           'oldest to newest, along the feature dimension'
                           if use_last4 else None),
        intermediate_normalization=('shared final encoder LayerNorm applied independently '
                                    'to each selected block before pooling'
                                    if use_last4 else 'final encoder output as returned'),
        encoder_norm_type=type(encoder_norm).__name__ if encoder_norm is not None else None,
        encoder_norm_eps=float(encoder_norm.eps) if hasattr(encoder_norm, 'eps') else None,
        accuracy_unit='percent', scoring_epoch=epochs,
        best_selection='highest final validation top1, then lowest final validation '
                       'cross-entropy; remaining ties keep trial grid order',
        optimizer_steps_per_epoch=epoch_batches,
        optimized_images_per_epoch=[steps * batch_size for steps in epoch_batches],
        dropped_images_per_epoch=dropped_images,
        drop_last_effective_batch=True, fresh_training_augmentations=True,
        paper='https://arxiv.org/pdf/2301.08243',
        vissl_revision='09270ed25a6c2cf71263d955b64cbe076d34ac45',
        mae_revision='efb2a8062c206524e35e47d04501ed4f544c0ae8',
        vissl_config='https://github.com/facebookresearch/vissl/blob/'
                     '09270ed25a6c2cf71263d955b64cbe076d34ac45/configs/config/benchmark/'
                     'linear_image_classification/imagenet1k/models/vit_h16_no_cls.yaml',
        mae_optimizer='https://github.com/facebookresearch/mae/blob/'
                      'efb2a8062c206524e35e47d04501ed4f544c0ae8/util/lars.py',
        provenance='Paper A.2 epochs/batch/sweep/step schedule; VISSL pooling, heads and '
                   'bicubic transforms; MAE LARS and batch-256 reference LR scaling.',
        caveat='The I-JEPA evaluation code and VISSL custom linear_eval_heads parameter '
               'groups are not published. Exact original bias-decay rules are unverified; '
               'this implementation uses MAE LARS exclusions for all parameters with ndim <= 1. '
               'The paper grid takes precedence over the different grid in the public VISSL preset.',
    )
    return dict(best=dict(best), trials=results, protocol=protocol,
                probe_seconds=time.perf_counter() - started)
