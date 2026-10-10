import csv
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.linear_probe import evaluate_linear_probe, make_probe_loaders


ROOT = Path(__file__).resolve().parents[1]


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

    def launch(self, *args):
        result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

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

    def test_rollback_with_probe_disabled_trims_future_probe_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_images(root)
            output = root / 'monitored'
            base = ['scripts/train.py', '--data', root / 'train', '--epochs', '2',
                    '--batch-size', '2', '--class-subset-count', '2', '--seed', '17',
                    '--output', output]
            probe = ['--linear-probe', '--val-data', root / 'val', '--probe-epochs', '1',
                     '--probe-batch-size', '2']
            self.launch(*base, *probe, '--max-steps', '2')
            old_checkpoint = root / 'epoch1.pt'
            old_checkpoint.write_bytes((output / 'checkpoint.pt').read_bytes())
            probe_log = output / 'linear_probe.csv'
            retained_log = probe_log.read_text()
            self.launch(*base, *probe, '--resume', output / 'checkpoint.pt')
            with probe_log.open(newline='') as stream:
                completed_rows = list(csv.DictReader(stream))
            self.assertEqual([int(row['epoch']) for row in completed_rows], [1, 2])

            rolled_back = self.launch(*base, '--resume', old_checkpoint,
                                      '--no-linear-probe', '--max-steps', '2')
            self.assertEqual(probe_log.read_text(), retained_log)
            self.assertNotIn('Linear probe monitor epoch=', rolled_back.stdout)
            checkpoint = torch.load(output / 'checkpoint.pt', weights_only=True)
            self.assertEqual((checkpoint['step'], checkpoint['next_epoch']), (2, 1))
            self.assertFalse(checkpoint['config']['linear_probe'])
            self.assertEqual(checkpoint['config']['linear_probe_protocol']['epochs'], 1)

    def test_epoch_logging_resume_and_standalone_do_not_change_pretraining(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_images(root)
            base = ['scripts/train.py', '--data', root / 'train', '--epochs', '2',
                    '--batch-size', '2', '--class-subset-count', '2', '--seed', '17']
            plain = root / 'plain'
            monitored = root / 'monitored'
            self.launch(*base, '--output', plain)
            probe = ['--linear-probe', '--val-data', root / 'val', '--probe-epochs', '2',
                     '--probe-batch-size', '3', '--output', monitored]
            # Stop part-way through epoch 1: there must be no misleading epoch evaluation.
            self.launch(*base, *probe, '--max-steps', '1')
            self.assertFalse((monitored / 'linear_probe.csv').exists())
            self.launch(*base, *probe, '--resume', monitored / 'checkpoint.pt')
            expected = torch.load(plain / 'checkpoint.pt', weights_only=True)
            actual = torch.load(monitored / 'checkpoint.pt', weights_only=True)
            for component in ('encoder', 'predictor', 'target_encoder'):
                for name in expected[component]:
                    torch.testing.assert_close(expected[component][name], actual[component][name], rtol=0, atol=0)
            with (monitored / 'linear_probe.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row['epoch']) for row in rows], [1, 2])
            self.assertEqual([int(row['step']) for row in rows], [2, 4])
            self.assertTrue(all(int(row['train_images']) == int(row['val_images']) == 4 for row in rows))
            saved_config = json.loads((monitored / 'config.json').read_text())
            self.assertEqual(actual['config'], saved_config)
            self.assertEqual(saved_config['linear_probe_protocol']['epochs'], 2)

            # Simulate missing epoch-2 metrics after an interruption during evaluation.
            log_path = monitored / 'linear_probe.csv'
            log_path.write_text('\n'.join(log_path.read_text().splitlines()[:2]) + '\n')
            self.launch(*base, *probe, '--resume', monitored / 'checkpoint.pt')
            with log_path.open(newline='') as stream:
                recovered = list(csv.DictReader(stream))
            self.assertEqual([int(row['epoch']) for row in recovered], [1, 2])
            for key in ('train_loss', 'val_loss', 'train_top1', 'val_top1'):
                self.assertEqual(rows[-1][key], recovered[-1][key])
            output = root / 'linear.json'
            self.launch('scripts/evaluate_linear.py', '--mode', 'monitor',
                        '--checkpoint', monitored / 'checkpoint.pt',
                        '--train', root / 'train', '--val', root / 'val', '--output', output,
                        '--probe-epochs', '2', '--probe-batch-size', '3', '--batch-size', '2', '--seed', '17')
            standalone = json.loads(output.read_text())
            self.assertEqual(standalone['config']['encoder_source'], 'target_encoder')
            for key in ('train_loss', 'val_loss', 'train_top1', 'val_top1'):
                self.assertAlmostEqual(standalone['metrics'][key], float(rows[-1][key]), places=6)

            # Both entry modes work on the same checkpoint. Tiny has only two
            # blocks, so final evaluates last-block features, not a fake last-four.
            final_output = root / 'linear_final.json'
            self.launch('scripts/evaluate_linear.py', '--mode', 'final',
                        '--checkpoint', monitored / 'checkpoint.pt',
                        '--train', root / 'train', '--val', root / 'val', '--output', final_output,
                        '--probe-epochs', '1', '--probe-batch-size', '2', '--batch-size', '2', '--seed', '17')
            final_result = json.loads(final_output.read_text())
            self.assertEqual(final_result['config']['mode'], 'final')
            self.assertEqual(len(final_result['metrics']['trials']), 12)
            self.assertEqual(final_result['metrics']['best']['val_images'], 4)
            self.assertEqual(final_result['metrics']['best']['val_top1'],
                             max(trial['val_top1'] for trial in final_result['metrics']['trials']))

            # Existing training checkpoints can enable the probe without changing pretraining.
            self.launch(*base, '--output', plain, '--resume', plain / 'checkpoint.pt',
                        '--linear-probe', '--val-data', root / 'val', '--probe-epochs', '2',
                        '--probe-batch-size', '3')
            with (plain / 'linear_probe.csv').open(newline='') as stream:
                enabled_later = list(csv.DictReader(stream))
            self.assertEqual([int(row['epoch']) for row in enabled_later], [2])
            for key in ('train_loss', 'val_loss', 'train_top1', 'val_top1'):
                self.assertEqual(rows[-1][key], enabled_later[-1][key])

            # Turning monitoring off is safe, but must not erase the old recipe
            # and permit incompatible measurements to enter the same CSV later.
            log_before = (plain / 'linear_probe.csv').read_text()
            self.launch(*base, '--output', plain, '--resume', plain / 'checkpoint.pt',
                        '--no-linear-probe')
            disabled = torch.load(plain / 'checkpoint.pt', weights_only=True)
            self.assertFalse(disabled['config']['linear_probe'])
            self.assertEqual(disabled['config']['linear_probe_protocol']['epochs'], 2)
            self.assertEqual((plain / 'linear_probe.csv').read_text(), log_before)
            rejected = subprocess.run(
                [sys.executable, *map(str, base), '--output', str(plain),
                 '--resume', str(plain / 'checkpoint.pt'), '--linear-probe',
                 '--val-data', str(root / 'val'), '--probe-epochs', '3'],
                cwd=ROOT, capture_output=True, text=True, timeout=120,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('linear_probe_protocol', rejected.stderr)


if __name__ == '__main__':
    unittest.main()
