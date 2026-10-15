"""The actual main-branch launcher must produce evaluable EMA and joint checkpoints."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]


class TestLinearEvaluationHandoff(unittest.TestCase):
    def test_main_training_checkpoints_support_cached_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ('train', 'val'):
                for label in ('a', 'b'):
                    folder = root / split / label
                    folder.mkdir(parents=True)
                    for index in range(2):
                        pixels = np.random.default_rng(index).integers(0, 256, (32, 32, 3), dtype=np.uint8)
                        Image.fromarray(pixels).save(folder / f'{index}.png')
            env = {key: value for key, value in os.environ.items() if not key.upper().startswith('COMET_')}
            env['PYTHONIOENCODING'] = 'utf-8'

            def run(*arguments):
                result = subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT,
                                        env=env, text=True, capture_output=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            for mode, regularizer in (('ema', 'none'), ('joint', 'sigreg'), ('joint', 'visreg')):
                with self.subTest(mode=mode, regularizer=regularizer):
                    output = root / regularizer
                    # Existing helper mocks only the training Comet SDK. It runs the
                    # unchanged scripts/train.py, not a reconstructed checkpoint fixture.
                    run('tests/run_training.py', '--smoke-test', '--mode', mode,
                        '--regularizer', regularizer, '--max-steps', '1', '--output', output)
                    checkpoint = output / 'checkpoint.pt'
                    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                    self.assertEqual(saved['config']['mode'], mode)
                    self.assertEqual('target_encoder' in saved, mode == 'ema')
                    config_path = root / f'{regularizer}.json'
                    config_path.write_text(json.dumps(dict(
                        checkpoint=str(checkpoint), train=str(root / 'train'), val=str(root / 'val'),
                        cache=str(output / 'features'), output=str(output / 'linear_cached.json'),
                        epochs=1, batch_size=2, probe_batch_size=2, device='cpu',
                        precision='fp32', num_workers=0, seed=0, comet=False,
                    )))
                    run('scripts/prepare_linear_cache.py', '--config', config_path)
                    run('scripts/evaluate_linear_cached.py', '--config', config_path)
                    manifest = json.loads((output / 'features' / 'manifest.json').read_text())
                    expected_source = 'target_encoder' if mode == 'ema' else 'encoder'
                    self.assertEqual(manifest['provenance']['encoder_source'], expected_source)
                    self.assertEqual(manifest['provenance']['checkpoint_mode'], mode)
                    self.assertEqual(manifest['provenance']['checkpoint_regularizer'], regularizer)
                    report = json.loads((output / 'linear_cached.json').read_text())
                    self.assertEqual(len(report['metrics']['trials']), 12)
                    self.assertEqual(report['metrics']['best']['val_images'], 4)
                    self.assertTrue((output / 'linear_cached.best.md').is_file())
                    self.assertFalse((output / 'comet.json').exists())


if __name__ == '__main__':
    unittest.main()
