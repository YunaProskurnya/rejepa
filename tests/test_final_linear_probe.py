import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms

from src.linear_probe_final import (
    _pooled_features, evaluate_final_linear_probe, make_final_probe_loaders,
)
from src.reference.lars import LARS
from src.models.encoder import VisionTransformer


class ToyEncoder(nn.Module):
    def __init__(self, depth=1, fail_at=None):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)
        self.proj.weight.data.copy_(torch.eye(2))
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(depth)])
        self.calls = 0
        self.seen = []
        self.fail_at = fail_at

    def get_intermediate_layers(self, images, n=4):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('injected encoder failure')
        self.seen.append(images.detach().clone())
        value = self.proj(images).unsqueeze(1)
        return [value + index for index in range(len(self.blocks) - n, len(self.blocks))]

    def forward(self, images):
        return self.get_intermediate_layers(images, n=1)[0]


class AugmentedDataset(Dataset):
    def __init__(self, length=9):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        jitter = torch.rand(2) + random.random() + np.random.random()
        return torch.tensor([float(index % 2), float(1 - index % 2)]) + jitter, index % 2


def loaders(length=9, inverse_val=False, augmented=False):
    labels = torch.arange(length) % 2
    features = F_one_hot(labels)
    train_dataset = AugmentedDataset(length) if augmented else TensorDataset(features, labels)
    return (
        DataLoader(train_dataset, batch_size=3, shuffle=True,
                   generator=torch.Generator().manual_seed(123)),
        DataLoader(TensorDataset(features, labels), batch_size=3,
                   generator=torch.Generator().manual_seed(124)),
        DataLoader(TensorDataset(features, 1 - labels if inverse_val else labels), batch_size=3,
                   generator=torch.Generator().manual_seed(125)),
    )


def F_one_hot(labels):
    return torch.nn.functional.one_hot(labels, num_classes=2).float()


def evaluate(encoder, data, **kwargs):
    settings = dict(epochs=2, batch_size=4, learning_rates=(0.05,), weight_decays=(0.0005,), seed=7)
    settings.update(kwargs)
    return evaluate_final_linear_probe(encoder, *data, 2, 'cpu', **settings)


class TestFinalLinearProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def assert_rng_equal(self, before):
        torch_state, python_state, numpy_state = before
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        self.assertEqual(np.random.get_state()[0], numpy_state[0])
        self.assertEqual(np.random.get_state()[2:], numpy_state[2:])

    def test_fresh_epochs_share_forward_and_restore_rng_encoder_and_generators(self):
        encoder = ToyEncoder(depth=4).train()
        encoder.proj.eval()
        encoder.proj.weight.grad = torch.ones_like(encoder.proj.weight)
        weight, grad = encoder.proj.weight.detach().clone(), encoder.proj.weight.grad.clone()
        data = loaders(augmented=True)
        states = [loader.generator.get_state().clone() for loader in data]
        rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
        result = evaluate(encoder, data)
        self.assertEqual(encoder.calls, 2 * len(data[0]) + len(data[1]) + len(data[2]))
        self.assertFalse(torch.equal(encoder.seen[0], encoder.seen[len(data[0])]))
        self.assertEqual(len(result['trials']), 4)
        self.assertEqual({trial['feature_dim'] for trial in result['trials']}, {2, 8})
        self.assertTrue(encoder.training)
        self.assertFalse(encoder.proj.training)
        self.assertTrue(encoder.proj.weight.requires_grad)
        torch.testing.assert_close(encoder.proj.weight, weight, rtol=0, atol=0)
        torch.testing.assert_close(encoder.proj.weight.grad, grad, rtol=0, atol=0)
        self.assert_rng_equal(rng)
        for loader, state in zip(data, states):
            self.assertTrue(torch.equal(loader.generator.get_state(), state))
        again = evaluate(encoder, data)
        self.assertEqual(result['trials'], again['trials'])
        self.assertEqual(result['best'], again['best'])
        json.dumps(result, allow_nan=False)

    def test_effective_batch_discards_tail_and_scores_all_images(self):
        result = evaluate(ToyEncoder(), loaders(length=9))
        self.assertEqual(result['protocol']['representations'], ['last'])
        self.assertEqual(result['protocol']['optimizer_steps_per_epoch'], [2, 2])
        self.assertEqual(result['protocol']['optimized_images_per_epoch'], [8, 8])
        self.assertEqual(result['protocol']['dropped_images_per_epoch'], [1, 1])
        self.assertTrue(all(trial['train_images'] == 9 for trial in result['trials']))
        self.assertTrue(all(trial['val_images'] == 9 for trial in result['trials']))
        self.assertAlmostEqual(result['trials'][0]['lr'], 0.05 * 4 / 256)

    def test_last4_concatenation_keeps_each_layer_and_runs_encoder_once(self):
        encoder = ToyEncoder(depth=4)
        images = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        features = _pooled_features(encoder, images, torch.device('cpu'), 'fp32', True)
        torch.testing.assert_close(features['last'], images + 3)
        torch.testing.assert_close(features['last4'],
                                   torch.tensor([[1., 2., 2., 3., 3., 4., 4., 5.],
                                                 [3., 4., 4., 5., 5., 6., 6., 7.]]))
        self.assertEqual(encoder.calls, 1)
        self.assertFalse(features['last4'].requires_grad)

    def test_real_encoder_intermediates_match_block_hooks_and_normal_forward(self):
        encoder = VisionTransformer(image_size=8, patch_size=4, embed_dim=8,
                                    depth=4, num_heads=2).eval()
        images = torch.randn(2, 3, 8, 8)
        block_outputs = []
        hooks = [block.register_forward_hook(
            lambda _module, _args, output: block_outputs.append(output.detach().clone())
        ) for block in encoder.blocks]
        try:
            intermediate = encoder.get_intermediate_layers(images, n=4)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(len(intermediate), 4)
        self.assertEqual(len(block_outputs), 4)
        for actual, raw in zip(intermediate, block_outputs):
            torch.testing.assert_close(actual, encoder.norm(raw), rtol=0, atol=0)
            self.assertEqual(tuple(actual.shape), (2, 4, 8))
        torch.testing.assert_close(encoder(images), intermediate[-1], rtol=0, atol=0)
        pooled = _pooled_features(encoder, images, torch.device('cpu'), 'fp32', True)
        torch.testing.assert_close(pooled['last4'],
                                   torch.cat([layer.mean(dim=1) for layer in intermediate], dim=-1))

    def test_progress_and_protocol_describe_actual_evaluation(self):
        rows = []
        encoder = ToyEncoder(depth=4)
        encoder.norm = nn.LayerNorm(2, eps=1e-6)
        result = evaluate(encoder, loaders(), progress=rows.append)
        self.assertEqual([row['epoch'] for row in rows], [1, 2])
        self.assertTrue(all(row['epochs'] == 2 and row['steps'] == 2 for row in rows))
        self.assertTrue(all(row['seconds'] >= 0 for row in rows))
        self.assertEqual(result['protocol']['accuracy_unit'], 'percent')
        self.assertEqual(result['protocol']['encoder_norm_eps'], 1e-6)
        self.assertEqual(result['protocol']['transform_repr'],
                         {'train': 'None', 'train_eval': 'None', 'val': 'None'})
        self.assertIn('lowest final validation', result['protocol']['best_selection'])

    def test_progress_exception_restores_state(self):
        encoder = ToyEncoder().train()
        data = loaders(augmented=True)
        states = [loader.generator.get_state().clone() for loader in data]
        rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()

        def fail(_row):
            raise RuntimeError('progress callback failure')

        with self.assertRaisesRegex(RuntimeError, 'progress callback'):
            evaluate(encoder, data, progress=fail)
        self.assert_rng_equal(rng)
        self.assertTrue(encoder.training)
        for loader, state in zip(data, states):
            self.assertTrue(torch.equal(loader.generator.get_state(), state))

    def test_reference_lr_scaling_and_all_three_step_boundaries(self):
        optimizers = []

        class RecordingLARS(LARS):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.history = []
                optimizers.append(self)

            def step(self):
                self.history.append(self.param_groups[0]['lr'])
                return super().step()

        with patch('src.linear_probe_final.LARS', RecordingLARS):
            evaluate(ToyEncoder(), loaders(length=4), epochs=46, learning_rates=(25.6,))
        expected = [0.4] * 15 + [0.04] * 15 + [0.004] * 15 + [0.0004]
        self.assertEqual(len(optimizers), 2)
        for optimizer in optimizers:
            np.testing.assert_allclose(optimizer.history, expected, rtol=1e-12, atol=0)

    def test_validation_labels_never_change_training(self):
        normal = evaluate(ToyEncoder(), loaders())
        inverted = evaluate(ToyEncoder(), loaders(inverse_val=True))
        for first, second in zip(normal['trials'], inverted['trials']):
            self.assertEqual(first['train_loss'], second['train_loss'])
            self.assertEqual(first['train_top1'], second['train_top1'])
            self.assertAlmostEqual(first['val_top1'] + second['val_top1'], 100)
        expected = max(normal['trials'], key=lambda trial: (trial['val_top1'], -trial['val_loss']))
        self.assertEqual(normal['best'], expected)

    def test_exception_restores_caller_state(self):
        encoder = ToyEncoder(fail_at=2).train()
        encoder.proj.eval()
        encoder.proj.weight.grad = torch.ones_like(encoder.proj.weight)
        weight = encoder.proj.weight.detach().clone()
        data = loaders(augmented=True)
        states = [loader.generator.get_state().clone() for loader in data]
        rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            evaluate(encoder, data)
        self.assert_rng_equal(rng)
        self.assertTrue(encoder.training)
        self.assertFalse(encoder.proj.training)
        torch.testing.assert_close(encoder.proj.weight, weight, rtol=0, atol=0)
        torch.testing.assert_close(encoder.proj.weight.grad, torch.ones_like(weight))
        for loader, state in zip(data, states):
            self.assertTrue(torch.equal(loader.generator.get_state(), state))

    def test_rejects_data_smaller_than_effective_batch(self):
        with self.assertRaisesRegex(ValueError, 'smaller than the effective head batch'):
            evaluate(ToyEncoder(), loaders(length=3))

    def test_loader_transforms_and_class_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ('train', 'val'):
                for name in ('a', 'b'):
                    path = root / split / name
                    path.mkdir(parents=True)
                    Image.new('RGB', (32, 24), color=(50, 80, 100)).save(path / 'x.png')
            train, train_eval, val = make_final_probe_loaders(
                root / 'train', root / 'val', class_names=['b', 'a'], image_size=16,
                batch_size=2,
            )
            crop = train.dataset.transform.transforms[0]
            self.assertIsInstance(crop, transforms.RandomResizedCrop)
            self.assertEqual(crop.scale, (0.08, 1.0))
            self.assertEqual(crop.interpolation, transforms.InterpolationMode.BICUBIC)
            self.assertIsInstance(train.dataset.transform.transforms[1], transforms.RandomHorizontalFlip)
            for loader in (train_eval, val):
                self.assertIsInstance(loader.dataset.transform.transforms[0], transforms.Resize)
                self.assertEqual(loader.dataset.transform.transforms[0].interpolation,
                                 transforms.InterpolationMode.BICUBIC)
                self.assertEqual(loader.dataset.class_to_idx, {'b': 0, 'a': 1})
                first = list(loader)[0][0]
                second = list(loader)[0][0]
                torch.testing.assert_close(first, second, rtol=0, atol=0)
            self.assertIsNot(train.dataset, train_eval.dataset)


if __name__ == '__main__':
    unittest.main()
