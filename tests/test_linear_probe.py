from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.linear_probe import evaluate_linear_probe, make_probe_loaders




class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)
        self.proj.weight.data.copy_(torch.eye(2))
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return self.proj(images).unsqueeze(1)


class TestLinearProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_only_head_learns_and_val_labels_never_train_it(self):
        labels = torch.arange(16) % 2
        features = torch.nn.functional.one_hot(labels, num_classes=2).float() * 4
        generator = torch.Generator().manual_seed(42)
        generator_state = generator.get_state().clone()
        train = DataLoader(TensorDataset(features, labels), batch_size=4, generator=generator)
        # Deliberately opposite val labels: leakage would allow the head to fit them.
        val = DataLoader(TensorDataset(features, 1 - labels), batch_size=4)
        encoder = ToyEncoder().train()
        encoder.proj.eval()  # Even a mixed tree of module modes must survive evaluation.
        encoder.proj.weight.grad = torch.ones_like(encoder.proj.weight)
        weights = encoder.proj.weight.detach().clone()
        torch_state = torch.get_rng_state().clone()
        python_state, numpy_state = random.getstate(), np.random.get_state()
        metrics = evaluate_linear_probe(encoder, train, val, 2, 'cpu', epochs=15,
                                        lr=0.1, batch_size=8, seed=0)
        self.assertEqual(metrics['train_top1'], 100)
        self.assertEqual(metrics['val_top1'], 0)
        self.assertLess(metrics['train_loss'], metrics['val_loss'])
        self.assertEqual(encoder.calls, len(train) + len(val))
        self.assertTrue(torch.equal(weights, encoder.proj.weight))
        self.assertTrue(torch.equal(encoder.proj.weight.grad, torch.ones_like(weights)))
        self.assertTrue(encoder.proj.weight.requires_grad)
        self.assertTrue(encoder.training)
        self.assertFalse(encoder.proj.training)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertTrue(torch.equal(generator_state, generator.get_state()))
        self.assertEqual(python_state, random.getstate())
        np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
        self.assertEqual(numpy_state[2:], np.random.get_state()[2:])
        repeat = evaluate_linear_probe(encoder, train, val, 2, 'cpu', epochs=15,
                                       lr=0.1, batch_size=8, seed=0)
        for key in ('train_top1', 'val_top1', 'train_loss', 'val_loss'):
            self.assertEqual(metrics[key], repeat[key])
        mixed_precision = evaluate_linear_probe(encoder, train, val, 2, 'cpu', epochs=15,
                                                lr=0.1, batch_size=8, seed=0, precision='bf16')
        self.assertEqual(mixed_precision['train_top1'], 100)
        self.assertEqual(mixed_precision['val_top1'], 0)

    def test_failure_restores_state_and_rejects_empty_data(self):
        encoder = ToyEncoder().train()
        empty = DataLoader(TensorDataset(torch.zeros(0, 2), torch.zeros(0, dtype=torch.long)))
        before = torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            evaluate_linear_probe(encoder, empty, empty, 2, 'cpu')
        self.assertTrue(encoder.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def make_images(self, root):
        generator = np.random.default_rng(12)
        for split in ('train', 'val'):
            for name in ('ant', 'bird', 'cat'):
                directory = root / split / name
                directory.mkdir(parents=True)
                for i in range(2):
                    pixels = generator.integers(0, 256, (240, 260, 3), dtype=np.uint8)
                    Image.fromarray(pixels).save(directory / f'{i}.png')

    def test_loaders_use_stable_views_and_consistent_class_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_images(root)
            train, val = make_probe_loaders(root / 'train', root / 'val',
                                            class_names=['cat', 'ant'], batch_size=3)
            self.assertEqual(train.dataset.class_to_idx, {'cat': 0, 'ant': 1})
            self.assertEqual(train.dataset.class_to_idx, val.dataset.class_to_idx)
            self.assertEqual(len(train.dataset), 4)
            first = list(train)
            second = list(train)
            self.assertEqual(sum(len(labels) for _, labels in first), 4)  # Keep the last small batch.
            for (x, y), (repeated, targets) in zip(first, second):
                self.assertTrue(torch.equal(x, repeated))
                self.assertTrue(torch.equal(y, targets))
            with self.assertRaisesRegex(ValueError, 'distinct'):
                make_probe_loaders(root / 'train', root / 'train')
            (root / 'val' / 'cat').rename(root / 'val' / 'dog')
            with self.assertRaisesRegex(ValueError, 'class_to_idx'):
                make_probe_loaders(root / 'train', root / 'val')


    def test_cli_modes_have_distinct_defaults(self):
        from scripts.evaluate_linear import parse_args
        with tempfile.TemporaryDirectory() as temporary:
            base = ['--checkpoint', 'checkpoint.pt', '--train', 'train', '--val', 'val',
                    '--output', str(Path(temporary) / 'result.json')]
            _, final = parse_args(base)
            self.assertEqual((final.mode, final.probe_epochs, final.probe_batch_size),
                             ('final', 50, 16384))
            self.assertIsNone(final.probe_lr)
            _, monitor = parse_args([*base, '--mode', 'monitor'])
            self.assertEqual((monitor.probe_epochs, monitor.probe_batch_size, monitor.probe_lr),
                             (30, 256, 0.001))




if __name__ == '__main__':
    unittest.main()
