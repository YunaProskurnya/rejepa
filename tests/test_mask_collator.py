"""Compatibility checks for the optional team collator and data loader."""

from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torchvision.transforms import ToTensor

from src.datasets.imagenet100 import MaskCollator, make_data_loader, make_imagenet100
from src.masks import pick_context_patches, pick_target_patches, sample_block_size
from src.reference.multiblock import MaskCollator as ReferenceCollator


ROOT = Path(__file__).resolve().parents[1]
OPTIONS = dict(
    input_size=224, patch_size=16, enc_mask_scale=(0.85, 1.0),
    pred_mask_scale=(0.15, 0.2), aspect_ratio=(0.75, 1.5), min_keep=10,
)


class TestMaskCollator(unittest.TestCase):
    def setUp(self):
        self.rng_state = torch.get_rng_state()

    def tearDown(self):
        torch.set_rng_state(self.rng_state)

    def test_24_batches_match_reference_including_rng_state(self):
        batch = [(torch.full((3, 224, 224), float(index)), index) for index in range(3)]
        for seed in (0, 1, 17):
            torch.manual_seed(seed)
            reference = ReferenceCollator(**OPTIONS, nenc=1, npred=4, allow_overlap=False)
            expected = [(reference(batch), torch.get_rng_state()) for _ in range(8)]
            torch.manual_seed(seed)
            custom = MaskCollator(**OPTIONS, M=4)
            for index, (reference_batch, reference_rng) in enumerate(expected):
                with self.subTest(seed=seed, batch=index):
                    actual = custom(batch)
                    self.assertEqual([len(group) for group in actual], [2, 1, 4])
                    for wanted_group, actual_group in zip(reference_batch, actual):
                        for wanted, value in zip(wanted_group, actual_group):
                            torch.testing.assert_close(value, wanted, rtol=0, atol=0)
                    for group in actual[1:]:
                        for mask in group:
                            self.assertEqual(mask.dtype, torch.long)
                            self.assertEqual(mask.device.type, "cpu")
                            self.assertEqual(mask.shape[0], len(batch))
                            self.assertGreater(mask.shape[1], OPTIONS["min_keep"])
                    self.assertTrue(torch.equal(torch.get_rng_state(), reference_rng))
                    self.assertEqual(custom._itr_counter.value, index)

    def test_legacy_seeded_helpers_leave_global_rng_unchanged(self):
        for kwargs in ({}, {"seed": 19}):
            with self.subTest(kwargs=kwargs):
                torch.manual_seed(37)
                before = torch.get_rng_state()
                targets = pick_target_patches(14, **kwargs)
                context = pick_context_patches(14, targets, **kwargs)
                repeated_targets = pick_target_patches(14, **kwargs)
                repeated_context = pick_context_patches(14, repeated_targets, **kwargs)
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                for first, second in zip(targets, repeated_targets):
                    self.assertTrue(torch.equal(first, second))
                self.assertTrue(torch.equal(context, repeated_context))

    def test_unseeded_helpers_advance_global_rng(self):
        generator = torch.Generator().manual_seed(0)
        target_size = sample_block_size(14, OPTIONS["pred_mask_scale"], OPTIONS["aspect_ratio"], generator)
        context_size = sample_block_size(14, OPTIONS["enc_mask_scale"], (1.0, 1.0), generator)
        before = torch.get_rng_state()
        targets = pick_target_patches(14, block_size=target_size, seed=None)
        after_targets = torch.get_rng_state()
        self.assertFalse(torch.equal(before, after_targets))
        pick_context_patches(14, targets, block_size=context_size, seed=None)
        self.assertFalse(torch.equal(after_targets, torch.get_rng_state()))

    def test_context_relaxes_full_exclusion_after_20_attempts_like_reference(self):
        reference = ReferenceCollator(input_size=128, patch_size=16, min_keep=10)
        torch.manual_seed(23)
        with patch("torch.randint", wraps=torch.randint) as draws:
            expected, _ = reference._sample_block_mask((7, 7), [torch.zeros(8, 8, dtype=torch.int32)])
        self.assertEqual(draws.call_count, 42)  # two coordinates per attempt, then success
        expected_rng = torch.get_rng_state()
        torch.manual_seed(23)
        with patch("torch.randint", wraps=torch.randint) as draws:
            actual = pick_context_patches(
                8, [torch.arange(64)], block_size=(7, 7), min_keep=10, seed=None,
            )
        self.assertEqual(draws.call_count, 42)
        self.assertEqual(actual.numel(), 49)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_invalid_settings_raise_without_hanging(self):
        # Bound this regression check even if impossible min_keep starts looping again.
        program = textwrap.dedent("""
            import torch
            from src.datasets.imagenet100 import MaskCollator
            from src.masks import pick_context_patches, pick_target_patches, sample_block_size
            batch = [(torch.zeros(3, 224, 224), 0)]
            cases = {
                'one-patch grid': lambda: MaskCollator(input_size=16, patch_size=16),
                'rectangular grid': lambda: MaskCollator(input_size=(224, 192)),
                'unaligned grid': lambda: MaskCollator(input_size=225),
                'zero patch': lambda: MaskCollator(patch_size=0),
                'zero targets': lambda: MaskCollator(M=0),
                'negative targets': lambda: MaskCollator(M=-1),
                'negative min_keep': lambda: MaskCollator(min_keep=-1),
                'invalid sampled grid': lambda: sample_block_size(1, (.15, .2), (.75, 1.5)),
                'empty target block': lambda: pick_target_patches(8, block_size=(0, 2)),
                'oversize target block': lambda: pick_target_patches(8, block_size=(8, 2)),
                'impossible context': lambda: pick_context_patches(8, [], block_size=(3, 3), min_keep=9),
                'negative context minimum': lambda: pick_context_patches(8, [], min_keep=-1),
                'target minimum': lambda: MaskCollator(pred_mask_scale=(.01, .01))(batch),
                'context minimum': lambda: MaskCollator(enc_mask_scale=(.01, .01))(batch),
            }
            for name, call in cases.items():
                try:
                    call()
                except ValueError:
                    continue
                raise AssertionError(name + ' did not raise ValueError')
            print('all invalid settings rejected')
        """)
        result = subprocess.run(
            [sys.executable, "-B", "-c", program], cwd=ROOT,
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("all invalid settings rejected", result.stdout)


class TestDataLoading(unittest.TestCase):
    def test_imagenet_splits_seed_and_epoch_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split, offset in (("train", 0), ("val", 64)):
                for index in range(6):
                    folder = root / split / f"class_{index // 3}"
                    folder.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (8, 8), (offset + index, 0, 0)).save(folder / f"{index}.png")
            options = dict(
                transform=ToTensor(), batch_size=2, root_path=root,
                num_workers=0, pin_mem=False, seed=31, drop_last=False,
            )
            train, loader, sampler = make_imagenet100(**options)
            self.assertEqual(Path(train.root), root / "train")
            self.assertEqual(len(train), 6)
            self.assertTrue(sampler.shuffle)
            self.assertIs(loader.sampler, sampler)
            self.assertEqual(loader.generator.initial_seed(), 31)
            first_order = list(sampler)
            self.assertEqual(first_order, torch.randperm(6, generator=torch.Generator().manual_seed(31)).tolist())
            _, _, repeated = make_imagenet100(**options)
            self.assertEqual(list(repeated), first_order)
            sampler.set_epoch(1)
            second_order = list(sampler)
            self.assertNotEqual(first_order, second_order)
            self.assertEqual(second_order, torch.randperm(6, generator=torch.Generator().manual_seed(32)).tolist())
            actual_images = torch.cat([images for images, _ in loader])
            torch.testing.assert_close(actual_images, torch.stack([train[index][0] for index in second_order]))

            validation, val_loader, val_sampler = make_imagenet100(**options, training=False)
            self.assertEqual(Path(validation.root), root / "val")
            self.assertFalse(val_sampler.shuffle)
            self.assertEqual(list(val_sampler), list(range(6)))
            val_sampler.set_epoch(5)
            self.assertEqual(list(val_sampler), list(range(6)))
            validation_images = torch.cat([images for images, _ in val_loader])
            self.assertEqual((validation_images[:, 0, 0, 0] * 255).round().tolist(), list(range(64, 70)))

    def test_data_loader_preserves_explicit_sampler_and_partial_batch(self):
        order = [5, 1, 4]
        generator = torch.Generator().manual_seed(7)
        loader = make_data_loader(
            torch.arange(6), batch_size=2, sampler=order, generator=generator,
            num_workers=0, pin_mem=False, drop_last=False,
        )
        self.assertIs(loader.sampler, order)
        self.assertIs(loader.generator, generator)
        self.assertEqual([batch.tolist() for batch in loader], [[5, 1], [4]])


if __name__ == "__main__":
    unittest.main()
