"""Disk-backed FP32 features: one ordered crop/flip realization per training epoch.

No images or encoder are needed by FeatureCache. NumPy mmap reads bounded batches,
not the entire cache. The manifest appears only after every split is complete.
"""

import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .linear_probe_final import _labels, _loader_generators, _pooled_features


def prepare_feature_cache(encoder, train_loader, train_eval_loader, val_loader,
                          num_classes, device, path, *, epochs=50, seed=0,
                          precision='fp32', provenance=None, progress=None):
    """Extract all images, including optimizer-batch tails, without fitting heads.

    Output must not exist. An interrupted cache is deliberately not usable; use a
    new directory or remove the incomplete one explicitly before restarting.
    """
    if epochs < 1 or num_classes < 1 or precision not in ('fp32', 'bf16'):
        raise ValueError('invalid cache epochs, class count or precision')
    path, device = Path(path), torch.device(device)
    if any(len(loader.dataset) == 0 for loader in (train_loader, train_eval_loader, val_loader)):
        raise ValueError('cache requires nonempty train and validation datasets')
    path.mkdir(parents=True, exist_ok=False)
    use_last4 = (hasattr(encoder, 'get_intermediate_layers') and
                 len(getattr(encoder, 'blocks', ())) >= 4)
    stored = 'last4' if use_last4 else 'last'
    norm = getattr(encoder, 'norm', None)
    metadata = dict(
        version=1, epochs=epochs, seed=seed, precision=precision, storage_dtype='float32',
        representations=['last', 'last4'] if use_last4 else ['last'],
        stored_representation=stored, num_classes=num_classes,
        train_images=len(train_loader.dataset), val_images=len(val_loader.dataset),
        extraction_batch_size=train_loader.batch_size,
        encoder_norm_type=type(norm).__name__ if norm is not None else None,
        encoder_norm_eps=float(norm.eps) if hasattr(norm, 'eps') else None,
        transforms={name: repr(getattr(loader.dataset, 'transform', None)) for name, loader in
                    zip(('train', 'train_eval', 'val'), (train_loader, train_eval_loader, val_loader))},
        provenance=provenance or {}, splits={}, torch_version=str(torch.__version__),
    )
    modes = [(module, module.training) for module in encoder.modules()]
    generators = _loader_generators((train_loader, train_eval_loader, val_loader))
    states = [(generator, generator.get_state()) for generator in generators]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    started = time.perf_counter()
    try:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            for generator in generators:
                generator.manual_seed(seed)
            encoder.eval()
            splits = [(f'train_{epoch + 1:04d}', train_loader) for epoch in range(epochs)]
            splits += [('train_eval', train_eval_loader), ('val', val_loader)]
            for name, loader in splits:
                split_started = time.perf_counter()
                feature_map = label_map = None
                offset = 0
                try:
                    for images, labels in loader:
                        features = _pooled_features(encoder, images, device, precision, use_last4)
                        labels = _labels(labels, num_classes)
                        values = features[stored].numpy()
                        if feature_map is None:
                            count, dimension = len(loader.dataset), values.shape[1]
                            metadata.setdefault('feature_dim', dimension)
                            feature_map = np.lib.format.open_memmap(
                                path / f'{name}.npy', mode='w+', dtype=np.float32,
                                shape=(count, dimension))
                            label_map = np.lib.format.open_memmap(
                                path / f'{name}_labels.npy', mode='w+', dtype=np.int64,
                                shape=(count,))
                        stop = offset + len(labels)
                        if stop > len(feature_map):
                            raise ValueError('loader yielded more images than its dataset')
                        feature_map[offset:stop] = values
                        label_map[offset:stop] = labels.numpy()
                        offset = stop
                    if feature_map is None or offset != len(loader.dataset):
                        raise ValueError('cache loader must visit every image exactly once')
                    feature_map.flush()
                    label_map.flush()
                finally:
                    # Close mappings explicitly (also releases Windows file handles).
                    for mapping in (feature_map, label_map):
                        if mapping is not None:
                            mapping._mmap.close()
                metadata['splits'][name] = dict(
                    images=offset, features_bytes=(path / f'{name}.npy').stat().st_size,
                    labels_bytes=(path / f'{name}_labels.npy').stat().st_size)
                if progress is not None:
                    progress(dict(split=name, images=offset,
                                  seconds=time.perf_counter() - split_started))
    finally:
        for module, training in modes:
            module.training = training
        for generator, state in states:
            generator.set_state(state)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    metadata['preparation_seconds'] = time.perf_counter() - started
    temporary = path / 'manifest.json.tmp'
    temporary.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    temporary.replace(path / 'manifest.json')
    return metadata


class FeatureCache:
    """Validate a completed cache and read batches in exactly the stored order."""

    def __init__(self, path):
        self.path = Path(path)
        if not (self.path / 'manifest.json').is_file():
            raise ValueError('missing manifest.json: feature cache is absent or incomplete')
        self.metadata = json.loads((self.path / 'manifest.json').read_text(encoding='utf-8'))
        meta = self.metadata
        if meta['version'] != 1 or meta['storage_dtype'] != 'float32':
            raise ValueError('unsupported feature cache format')
        expected = {f'train_{epoch + 1:04d}' for epoch in range(meta['epochs'])} | {'train_eval', 'val'}
        if set(meta['splits']) != expected:
            raise ValueError('feature cache has missing or unexpected splits')
        for name, split in meta['splits'].items():
            expected_images = meta['val_images'] if name == 'val' else meta['train_images']
            if split['images'] != expected_images:
                raise ValueError(f'invalid image count in cache: {name}')
            for suffix, size in (('', 'features_bytes'), ('_labels', 'labels_bytes')):
                file = self.path / f'{name}{suffix}.npy'
                if not file.is_file() or file.stat().st_size != split[size]:
                    raise ValueError(f'missing or truncated cache file: {file}')

    def batches(self, split, batch_size, *, drop_last=False):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        count = self.metadata['splits'][split]['images']
        features = np.load(self.path / f'{split}.npy', mmap_mode='r', allow_pickle=False)
        labels = None
        try:
            labels = np.load(self.path / f'{split}_labels.npy', mmap_mode='r', allow_pickle=False)
            dimension = self.metadata['feature_dim']
            if features.shape != (count, dimension) or features.dtype != np.float32:
                raise ValueError(f'invalid cached feature shape/dtype: {split}')
            if labels.shape != (count,) or labels.dtype != np.int64:
                raise ValueError(f'invalid cached label shape/dtype: {split}')
            end = count - count % batch_size if drop_last else count
            for start in range(0, end, batch_size):
                stop = min(start + batch_size, end)
                values = torch.from_numpy(np.array(features[start:stop], copy=True))
                targets = _labels(torch.from_numpy(np.array(labels[start:stop], copy=True)),
                                  self.metadata['num_classes'])
                if not torch.isfinite(values).all():
                    raise ValueError(f'non-finite cached features: {split}')
                if self.metadata['stored_representation'] == 'last4':
                    # Match the contiguous pooled tensor used in online evaluation;
                    # a strided slice can select different floating-point kernels.
                    batch = {'last': values[:, -(dimension // 4):].contiguous(), 'last4': values}
                else:
                    batch = {'last': values}
                yield batch, targets
        finally:
            features._mmap.close()
            if labels is not None:
                labels._mmap.close()
