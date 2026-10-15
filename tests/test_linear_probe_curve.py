import csv
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from scripts.evaluate_linear_curve import discover_checkpoints, main, select_samples
from src.linear_probe import make_probe_loaders
from src.models.encoder import VisionTransformer


ROOT = Path(__file__).resolve().parents[1]


class TestLinearProbeCurve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def make_data(self, root):
        for split, count in (('train', 4), ('val', 3)):
            for label in ('a', 'b'):
                directory = root / split / label
                directory.mkdir(parents=True)
                for index in range(count):
                    pixels = np.random.default_rng(index).integers(0, 256, (12, 16, 3), dtype=np.uint8)
                    Image.fromarray(pixels).save(directory / f'{index}.png')

    def make_checkpoints(self, root, regularizer='sigreg', epochs=(1, 2, 3)):
        options = dict(image_size=8, patch_size=4, embed_dim=8, depth=1, num_heads=2)
        model = VisionTransformer(**options)
        directory = root / regularizer
        directory.mkdir()
        mode = 'ema' if regularizer == 'none' else 'joint'
        for epoch in epochs:
            checkpoint = dict(config=dict(encoder=options, mode=mode, regularizer=regularizer),
                              encoder=model.state_dict(), epoch=epoch, next_epoch=epoch,
                              next_batch=0, step=epoch * 7)
            if mode == 'ema':
                checkpoint['target_encoder'] = model.state_dict()
                # A wrong choice of the online encoder must fail feature extraction.
                checkpoint['encoder'] = {key: value.clone().fill_(float('nan')) for key, value in model.state_dict().items()}
            torch.save(checkpoint, directory / f'checkpoint_{epoch - 1}.pt')
        return directory

    def args(self, root, checkpoints, output):
        return ['--checkpoints', str(checkpoints), '--train', str(root / 'train'),
                '--val', str(root / 'val'), '--output', str(output), '--device', 'cpu',
                '--num-workers', '0', '--batch-size', '2', '--train-per-class', '2',
                '--probe-epochs', '1', '--probe-batch-size', '2']

    def test_discovery_orders_and_selects_both_filename_conventions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('checkpoint_9.pt', 'checkpoint_0.pt', 'checkpoint_1.pt', 'checkpoint.pt'):
                (root / name).touch()
            self.assertEqual([epoch for epoch, _ in discover_checkpoints(root)], [1, 2, 10])
            self.assertEqual([epoch for epoch, _ in discover_checkpoints(root, every=5)], [1, 10])
            (root / 'checkpoint_epoch_0020.pt').touch()
            self.assertEqual([epoch for epoch, _ in discover_checkpoints(root, every=5)], [1, 10, 20])
            (root / 'checkpoint_epoch_0001.pt').touch()
            with self.assertRaisesRegex(ValueError, 'same epoch'):
                discover_checkpoints(root)

    def test_subset_is_fixed_balanced_and_shared_between_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_data(root)
            first, val = make_probe_loaders(root / 'train', root / 'val', image_size=8)
            second, _ = make_probe_loaders(root / 'train', root / 'val', image_size=8)
            select_samples(first.dataset, 2, seed=17)
            select_samples(second.dataset, 2, seed=17)
            self.assertEqual(first.dataset.samples, second.dataset.samples)
            self.assertEqual(first.dataset.targets.count(0), 2)
            self.assertEqual(first.dataset.targets.count(1), 2)
            select_samples(val.dataset, 0, seed=17)
            self.assertEqual(len(val.dataset), 6)

    def test_resume_adds_missing_epochs_and_rejects_mixed_protocol_or_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_data(root)
            checkpoints = self.make_checkpoints(root)
            output = root / 'curve'
            args = self.args(root, checkpoints, output)
            with redirect_stdout(StringIO()):
                main([*args, '--every', '10'])
            with (output / 'linear_curve.csv').open(newline='') as file:
                old_rows = list(csv.DictReader(file))
            self.assertEqual([int(row['epoch']) for row in old_rows], [1, 3])
            with patch('scripts.evaluate_linear_curve.evaluate_linear_probe', wraps=__import__(
                    'src.linear_probe', fromlist=['evaluate_linear_probe']).evaluate_linear_probe) as evaluate:
                with redirect_stdout(StringIO()):
                    main(args)
                self.assertEqual(evaluate.call_count, 1)
            with (output / 'linear_curve.csv').open(newline='') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([int(row['epoch']) for row in rows], [1, 2, 3])
            self.assertEqual(rows[0], old_rows[0])
            self.assertEqual(rows[2], old_rows[1])
            self.assertTrue(all(int(row['train_images']) == 4 and int(row['val_images']) == 6 for row in rows))
            with patch('scripts.evaluate_linear_curve.evaluate_linear_probe', side_effect=AssertionError('rerun')):
                with redirect_stdout(StringIO()):
                    main(args)
            before = (output / 'linear_curve.csv').read_bytes()
            with self.assertRaisesRegex(ValueError, 'settings/data differ'):
                main([*args, '--probe-lr', '0.02'])
            self.assertEqual((output / 'linear_curve.csv').read_bytes(), before)
            checkpoint = checkpoints / 'checkpoint_0.pt'
            with checkpoint.open('ab') as file:
                file.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                main(args)
            self.assertEqual((output / 'linear_curve.csv').read_bytes(), before)

    def test_interruption_preserves_completed_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_data(root)
            checkpoints = self.make_checkpoints(root)
            output = root / 'curve'
            from src.linear_probe import evaluate_linear_probe
            calls = 0

            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt()
                return evaluate_linear_probe(*args, **kwargs)

            with patch('scripts.evaluate_linear_curve.evaluate_linear_probe', side_effect=interrupted):
                with redirect_stdout(StringIO()), self.assertRaises(KeyboardInterrupt):
                    main(self.args(root, checkpoints, output))
            with (output / 'linear_curve.csv').open(newline='') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([int(row['epoch']) for row in rows], [1])
            self.assertTrue((output / 'linear_curve.svg').is_file())

    def test_real_cli_handles_ema_sigreg_visreg_and_comparison_plot_without_comet(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_data(root)
            env = {key: value for key, value in os.environ.items() if not key.upper().startswith('COMET_')}
            env['PYTHONIOENCODING'] = 'utf-8'
            csvs = []
            for regularizer in ('none', 'sigreg', 'visreg'):
                checkpoints = self.make_checkpoints(root, regularizer, epochs=(1, 2))
                output = root / f'{regularizer}_curve'
                config = root / f'{regularizer}.json'
                config.write_text(json.dumps(dict(
                    checkpoints=str(checkpoints), train=str(root / 'train'), val=str(root / 'val'),
                    output=str(output), device='cpu', num_workers=0, batch_size=2,
                    train_per_class=2, probe_epochs=1, probe_batch_size=2)))
                result = subprocess.run([sys.executable, 'scripts/evaluate_linear_curve.py', '--config', str(config)],
                                        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                csv_path = output / 'linear_curve.csv'
                csvs.append(str(csv_path))
                with csv_path.open(newline='') as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual([int(row['step']) for row in rows], [7, 14])
                self.assertEqual({row['encoder_source'] for row in rows},
                                 {'target_encoder' if regularizer == 'none' else 'encoder'})
                for suffix in ('.png', '.svg'):
                    self.assertGreater((output / f'linear_curve{suffix}').stat().st_size, 100)
                self.assertFalse((output / 'comet.json').exists())
            result = subprocess.run([sys.executable, 'scripts/plot_linear_curves.py', '--csv', *csvs,
                                     '--labels', 'EMA', 'SIGReg', 'VISReg', '--output', str(root / 'comparison')],
                                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((root / 'comparison.svg').is_file())


if __name__ == '__main__':
    unittest.main()
