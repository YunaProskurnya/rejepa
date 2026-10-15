"""Verify online/cache equivalence, bounded replay, and both real CLI commands."""

import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from src.linear_probe_cache import FeatureCache, prepare_feature_cache
from src.linear_probe_final import evaluate_final_linear_probe
from src.models.encoder import VisionTransformer
from tests.test_final_linear_probe import ToyEncoder, loaders


ROOT = Path(__file__).resolve().parents[1]


class TestFeatureCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_online_and_cached_full_grid_match_with_random_augmentations(self):
        for depth, trials in ((1, 12), (4, 24)):
            with self.subTest(depth=depth), tempfile.TemporaryDirectory() as temporary:
                encoder = ToyEncoder(depth=depth).train()
                data = loaders(augmented=True)
                torch_rng, py_rng, np_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
                states = [loader.generator.get_state().clone() for loader in data]
                weights = {name: tensor.clone() for name, tensor in encoder.state_dict().items()}
                path = Path(temporary) / 'cache'
                prepare_feature_cache(encoder, *data, 2, 'cpu', path, epochs=3, seed=7)
                self.assertTrue(encoder.training)
                self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
                self.assertEqual(random.getstate(), py_rng)
                np.testing.assert_array_equal(np.random.get_state()[1], np_rng[1])
                for loader, state in zip(data, states):
                    self.assertTrue(torch.equal(loader.generator.get_state(), state))
                for name, value in encoder.state_dict().items():
                    torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
                cache = FeatureCache(path)
                first = next(cache.batches('train_0001', 4))[0]['last']
                second = next(cache.batches('train_0002', 4))[0]['last']
                self.assertFalse(torch.equal(first, second))
                online = evaluate_final_linear_probe(encoder, *data, 2, 'cpu', epochs=3, batch_size=4, seed=7)
                with patch('src.linear_probe_final._pooled_features', side_effect=AssertionError('encoder used')):
                    cached = evaluate_final_linear_probe(None, None, None, None, 2, 'cpu',
                                                         epochs=3, batch_size=4, seed=7, feature_cache=cache)
                self.assertEqual(len(cached['trials']), trials)
                self.assertEqual(cached['trials'], online['trials'])
                self.assertEqual(cached['best'], online['best'])
                self.assertEqual(cached['protocol']['dropped_images_per_epoch'], [1, 1, 1])
                self.assertEqual(cached['best']['train_images'], 9)
                self.assertEqual(cached['best']['val_images'], 9)
                self.assertEqual(cached['protocol']['encoder_passes_during_evaluation'], 0)
                self.assertEqual(sum(len(y) for _, y in cache.batches('val', 4)), 9)

    def test_lite_reuses_one_crop_and_reports_weighted_validation_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'cache'
            prepare_feature_cache(ToyEncoder(depth=4), *loaders(), 2, 'cpu', path,
                                  epochs=1, last4=False)
            cache = FeatureCache(path)
            self.assertEqual(cache.metadata['representations'], ['last'])
            progress = []
            result = evaluate_final_linear_probe(
                None, None, None, None, 2, 'cpu', epochs=3, batch_size=4,
                feature_cache=cache, reuse_cached_features=True,
                representations=('last',), heads=('linear',), learning_rates=(0.01,),
                weight_decays=(0.0,), progress=progress.append)
            self.assertEqual(len(result['trials']), 1)
            self.assertEqual(len(result['history']), 3)
            self.assertEqual(len(progress), 3)
            self.assertAlmostEqual(result['history'][-1]['val_loss'], result['best']['val_loss'], places=6)
            self.assertEqual(result['best']['val_images'], 9)
            self.assertEqual(result['protocol']['name'], 'linear_lite_cached')

    def test_rejects_incomplete_truncated_and_insufficient_epoch_caches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                FeatureCache(root)
            data = loaders()
            path = root / 'complete'
            prepare_feature_cache(ToyEncoder(), *data, 2, 'cpu', path, epochs=1)
            cache = FeatureCache(path)
            with self.assertRaisesRegex(ValueError, 'exceed'):
                evaluate_final_linear_probe(None, None, None, None, 2, 'cpu', epochs=2,
                                             batch_size=4, feature_cache=cache)
            with self.assertRaises(FileExistsError):
                prepare_feature_cache(ToyEncoder(), *data, 2, 'cpu', path, epochs=1)
            (path / 'train_0001.npy').write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError, 'truncated'):
                FeatureCache(path)
            partial = root / 'partial'
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                prepare_feature_cache(ToyEncoder(fail_at=2), *data, 2, 'cpu', partial, epochs=1)
            self.assertFalse((partial / 'manifest.json').exists())

    def test_cli_preparation_then_evaluation_without_source_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ('train', 'val'):
                for label in ('a', 'b'):
                    folder = root / split / label
                    folder.mkdir(parents=True)
                    for index in range(3):
                        values = np.random.default_rng(index).integers(0, 256, (12, 12, 3), dtype=np.uint8)
                        Image.fromarray(values).save(folder / f'{index}.png')
            options = dict(image_size=8, patch_size=4, embed_dim=8, depth=4, num_heads=2)
            encoder = VisionTransformer(**options)
            checkpoint = root / 'checkpoint.pt'
            torch.save(dict(config={'encoder': options}, encoder=encoder.state_dict(), step=5, epoch=2), checkpoint)
            config = dict(checkpoint=str(checkpoint), train=str(root / 'train'), val=str(root / 'val'),
                          cache=str(root / 'features'), output=str(root / 'result.json'), epochs=2,
                          batch_size=2, probe_batch_size=4, num_workers=0, precision='fp32', device='cpu',
                          comet=False, seed=13)
            config_file = root / 'config.json'
            config_file.write_text(json.dumps(config))
            env = {key: value for key, value in os.environ.items() if not key.upper().startswith('COMET_')}
            env['PYTHONIOENCODING'] = 'utf-8'

            def run(script):
                result = subprocess.run([sys.executable, str(ROOT / 'scripts' / script), '--config', str(config_file)],
                                        cwd=ROOT, env=env, text=True, capture_output=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result

            run('prepare_linear_cache.py')
            full_config = dict(config)
            config.update(lite=True, cache=str(root / 'lite_features'), output=str(root / 'lite.json'))
            config_file.write_text(json.dumps(config))
            run('prepare_linear_cache.py')
            lite_manifest = json.loads((root / 'lite_features' / 'manifest.json').read_text())
            self.assertEqual(lite_manifest['epochs'], 1)
            self.assertEqual(lite_manifest['representations'], ['last'])
            config_file.write_text(json.dumps(full_config))
            online_output = root / 'online.json'
            online = subprocess.run([
                sys.executable, str(ROOT / 'scripts' / 'evaluate_linear.py'),
                '--checkpoint', str(checkpoint), '--train', config['train'], '--val', config['val'],
                '--output', str(online_output), '--probe-epochs', '2', '--probe-batch-size', '4',
                '--batch-size', '2', '--num-workers', '0', '--seed', '13', '--device', 'cpu',
            ], cwd=ROOT, env=env, text=True, capture_output=True, timeout=120)
            self.assertEqual(online.returncode, 0, online.stdout + online.stderr)
            # Prove evaluation needs neither checkpoint nor original image directories.
            checkpoint.unlink()
            (root / 'train').rename(root / 'train_unavailable')
            (root / 'val').rename(root / 'val_unavailable')
            result = run('evaluate_linear_cached.py')
            report = json.loads((root / 'result.json').read_text())
            best = json.loads((root / 'result.best.json').read_text())
            self.assertEqual(len(report['metrics']['trials']), 24)
            self.assertEqual(report['metrics']['trials'], json.loads(online_output.read_text())['metrics']['trials'])
            self.assertEqual(report['metrics']['best'], best['best'])
            self.assertEqual(best['best']['val_images'], 6)
            self.assertIn('Best linear probe', result.stdout)
            self.assertIn('Top-1 (%)', (root / 'result.best.md').read_text())
            self.assertEqual(len((root / 'result.trials.csv').read_text().splitlines()), 25)
            manifest = json.loads((root / 'features' / 'manifest.json').read_text())
            self.assertEqual(len(manifest['provenance']['checkpoint_sha256']), 64)
            self.assertEqual(manifest['provenance']['class_to_idx'], {'a': 0, 'b': 1})
            self.assertFalse((root / 'comet.json').exists())
            config_file.write_text(json.dumps(config))
            lite_result = run('evaluate_linear_cached.py')
            lite = json.loads((root / 'lite.json').read_text())['metrics']
            self.assertEqual(len(lite['trials']), 1)
            self.assertEqual(len(lite['history']), 2)
            self.assertIn('val_loss=', lite_result.stdout)
            self.assertAlmostEqual(lite['history'][-1]['val_loss'], lite['best']['val_loss'], places=6)


if __name__ == '__main__':
    unittest.main()
