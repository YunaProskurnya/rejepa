"""Проверка MaskCollator и сэмплера масок по контракту I-JEPA."""

import math
import multiprocessing
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
import torchvision.transforms as T
from PIL import Image

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.imagenet100 import MaskCollator, make_imagenet100
from src.masks import pick_context_patches, pick_target_patches, sample_block_size
from src.models.patch_embed import PatchEmbed
from src.models.token_selection import select_tokens
from src.transforms import make_transform


PAPER_SCALE_CONTEXT = (0.85, 1.0)
PAPER_SCALE_TARGET = (0.15, 0.2)
PAPER_ASPECT = (0.75, 1.5)


def _block_size(grid, scale_range, aspect_range, unit):
    scale = scale_range[0] + unit * (scale_range[1] - scale_range[0])
    ratio = aspect_range[0] + unit * (aspect_range[1] - aspect_range[0])
    area = int(grid * grid * scale)
    height = int(round(math.sqrt(area * ratio)))
    width = int(round(math.sqrt(area / ratio)))
    while height >= grid:
        height -= 1
    while width >= grid:
        width -= 1
    return height, width


def _rectangle(grid, top, left, height, width):
    rows = torch.arange(top, top + height)
    cols = torch.arange(left, left + width)
    grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing="ij")
    return (grid_rows * grid + grid_cols).reshape(-1)


def _bounds(indices, grid):
    rows = torch.div(indices, grid, rounding_mode="floor")
    cols = indices % grid
    top = int(rows.min())
    left = int(cols.min())
    return int(rows.max()) - top + 1, int(cols.max()) - left + 1, top, left


@contextmanager
def _scripted_sampler(unit=0.0, location=0):
    units = iter(unit if isinstance(unit, (list, tuple)) else ())

    def rand(*args, **kwargs):
        value = next(units) if isinstance(unit, (list, tuple)) else unit
        return torch.tensor(float(value))

    def randint(low, high, *args, **kwargs):
        if isinstance(location, (list, tuple)):
            value = location[randint.calls]
            randint.calls += 1
        else:
            value = location
        if not (int(low) <= int(value) < int(high)):
            raise AssertionError(f"location {value} outside [{int(low)}, {int(high)})")
        return torch.tensor(int(value))

    randint.calls = 0
    with patch("src.masks.torch.rand", rand), patch("src.masks.torch.randint", randint):
        yield


_builtin_min = min


def _reduce_min(*args):
    if len(args) == 2 and isinstance(args[1], map):
        return _builtin_min(args[0], _builtin_min(args[1]))
    return _builtin_min(*args)


@contextmanager
def _reachable_collate():
    with patch("builtins.min", _reduce_min):
        yield


def _pairs(batch_size, image_size, value=0.0):
    images = [torch.full((3, image_size, image_size), value + index) for index in range(batch_size)]
    return [(image.clone(), index) for index, image in enumerate(images)], images


def _small_collator(**kwargs):
    params = dict(
        input_size=32,
        patch_size=8,
        enc_mask_scale=PAPER_SCALE_CONTEXT,
        pred_mask_scale=PAPER_SCALE_TARGET,
        aspect_ratio=PAPER_ASPECT,
        M=4,
        min_keep=1,
    )
    params.update(kwargs)
    return MaskCollator(**params)


def _apply_masks(tokens, masks):
    gathered = []
    for mask in masks:
        index = mask.unsqueeze(-1).repeat(1, 1, tokens.size(-1))
        gathered.append(torch.gather(tokens, dim=1, index=index))
    return torch.cat(gathered, dim=0)


def _impossible_context():
    pick_context_patches(
        4,
        [torch.arange(2)],
        block_size=(1, 1),
        scale=(0.0, 0.0),
        aspect_ratio=(1.0, 1.0),
        min_keep=10,
    )


class _Tensors(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return torch.full((3, 32, 32), float(index)), index


def _write_subset(root, count=4):
    folder = Path(root) / "train" / "n01440764"
    folder.mkdir(parents=True)
    for index in range(count):
        Image.new("RGB", (16, 16), (index, 32, 64)).save(folder / f"{index}.jpg")


class TestMaskSampling(unittest.TestCase):
    def test_target_blocks_share_size_and_row_major_order(self):
        grid = 16
        unit = 0.0
        height, width = _block_size(grid, PAPER_SCALE_TARGET, PAPER_ASPECT, unit)
        with _scripted_sampler(unit, 0):
            masks = pick_target_patches(
                grid, (height, width), M=4, scale=PAPER_SCALE_TARGET, aspect_ratio=PAPER_ASPECT
            )

        self.assertEqual(len(masks), 4)
        expected = _rectangle(grid, 0, 0, height, width)
        for mask in masks:
            self.assertEqual(mask.dtype, torch.long)
            self.assertEqual(tuple(_bounds(mask, grid)[:2]), (height, width))
            torch.testing.assert_close(mask, expected)
            self.assertEqual(torch.unique(mask).numel(), mask.numel())

    def test_target_blocks_may_overlap(self):
        grid = 16
        with _scripted_sampler(0.0, 0):
            masks = pick_target_patches(
                grid, _block_size(grid, PAPER_SCALE_TARGET, PAPER_ASPECT, 0.0), M=2
            )
        torch.testing.assert_close(masks[0], masks[1])

    def test_context_is_square_and_drops_target_patches(self):
        grid = 8
        target = _rectangle(grid, 0, 0, 3, 3)
        with _scripted_sampler(0.0, 0):
            context = pick_context_patches(
                grid,
                [target],
                block_size=(7, 7),
                aspect_ratio=(1.0, 1.0),
                scale=(1.0, 1.0),
                min_keep=30,
            )

        height, width, top, left = _bounds(context, grid)
        self.assertEqual((height, width, top, left), (7, 7, 0, 0))
        self.assertGreater(context.numel(), 30)
        self.assertTrue(set(target.tolist()).isdisjoint(context.tolist()))
        self.assertEqual(context.dtype, torch.long)
        torch.testing.assert_close(context, context.sort().values)

    def test_context_uses_strict_min_keep_and_then_relaxes(self):
        grid = 8
        target = _rectangle(grid, 0, 0, 3, 3)
        calls = {"n": 0}
        def randint(*args, **kwargs):
            calls["n"] += 1
            return torch.tensor(0)

        with patch("src.masks.torch.randint", randint):
            context = pick_context_patches(
                grid,
                [target],
                block_size=(7, 7),
                aspect_ratio=(1.0, 1.0),
                scale=(1.0, 1.0),
                min_keep=40,
            )

        self.assertEqual(calls["n"], 42)
        self.assertEqual(context.numel(), 49)
        self.assertTrue(set(target.tolist()) <= set(context.tolist()))

    def test_location_high_excludes_the_final_fitting_cell(self):
        highs = []

        def rand(*args, **kwargs):
            return torch.tensor(0.0)

        def randint(low, high, *args, **kwargs):
            highs.append((int(low), int(high)))
            return torch.tensor(0)

        grid = 16
        height, width = _block_size(grid, PAPER_SCALE_TARGET, PAPER_ASPECT, 0.0)
        with patch("src.masks.torch.rand", rand), patch("src.masks.torch.randint", randint):
            pick_target_patches(grid, (height, width), M=1)

        self.assertEqual(highs, [(0, grid - height), (0, grid - width)])

    def test_small_target_blocks_are_kept(self):
        with _scripted_sampler(0.0, 0):
            masks = pick_target_patches(16, (1, 1), M=2)
        self.assertEqual(len(masks), 2)
        self.assertEqual(masks[0].numel(), 1)

    def test_nonpositive_aspect_ratio_is_rejected(self):
        try:
            sample_block_size(8, (0.2, 0.3), (0.0, 0.0))
        except ValueError:
            return
        except Exception as exc:
            self.fail(f"expected ValueError, got {type(exc).__name__}: {exc}")
        self.fail("nonpositive aspect ratio returned a mask")

    def test_impossible_min_keep_stops_instead_of_hanging(self):
        proc = multiprocessing.get_context("fork").Process(target=_impossible_context)
        proc.start()
        proc.join(2)
        hung = proc.is_alive()
        if hung:
            proc.terminate()
            proc.join()
        self.assertFalse(hung, "context sampling hung instead of rejecting an impossible min_keep")
        self.assertNotEqual(proc.exitcode, 0)


class TestMaskCollator(unittest.TestCase):
    def test_vith14_configuration_is_stored(self):
        collator = MaskCollator(
            input_size=224,
            patch_size=14,
            enc_mask_scale=PAPER_SCALE_CONTEXT,
            pred_mask_scale=PAPER_SCALE_TARGET,
            aspect_ratio=PAPER_ASPECT,
            M=4,
            min_keep=10,
        )
        self.assertEqual((collator.height, collator.width), (16, 16))
        self.assertEqual(collator.patch_size, 14)
        self.assertEqual(collator.enc_mask_scale, PAPER_SCALE_CONTEXT)
        self.assertEqual(collator.pred_mask_scale, PAPER_SCALE_TARGET)
        self.assertEqual(collator.aspect_ratio, PAPER_ASPECT)
        self.assertEqual(collator.M, 4)
        self.assertEqual(collator.min_keep, 10)
        self.assertEqual(collator._itr_counter.value, -1)

        wide = MaskCollator(input_size=448, patch_size=16)
        self.assertEqual((wide.height, wide.width), (28, 28))

    def test_collator_passes_configured_target_arguments(self):
        seen = {}

        def wrapped(grid, M=4, aspect_ratio=PAPER_ASPECT, scale=PAPER_SCALE_TARGET, **kwargs):
            seen.update(grid=grid, M=M, aspect_ratio=aspect_ratio, scale=scale)
            return [torch.arange(4) for _ in range(M)]

        collator = _small_collator()
        batch, _ = _pairs(2, 32)
        with patch("src.datasets.imagenet100.pick_target_patches", wrapped):
            try:
                collator(batch)
            except TypeError:
                pass
        self.assertEqual(seen["M"], 4)
        self.assertEqual(seen["scale"], PAPER_SCALE_TARGET)
        self.assertEqual(seen["aspect_ratio"], PAPER_ASPECT)

    def test_collator_passes_the_grid_side(self):
        seen = {}

        def wrapped(grid, M=4, **kwargs):
            seen["grid"] = grid
            return [torch.arange(4) for _ in range(M)]

        collator = _small_collator()
        batch, _ = _pairs(2, 32)
        with patch("src.datasets.imagenet100.pick_target_patches", wrapped):
            try:
                collator(batch)
            except TypeError:
                pass
        self.assertEqual(seen["grid"], collator.height)
        self.assertEqual(seen["grid"], collator.width)

    def _context_sampling_record(self):
        seen = {}

        def target(grid, M=4, **kwargs):
            return [torch.arange(4) for _ in range(M)]

        def context(grid, targets, **kwargs):
            seen["grid"] = grid
            seen["kwargs"] = kwargs
            seen["targets"] = len(targets)
            return torch.arange(6)

        collator = _small_collator(min_keep=10)
        batch, _ = _pairs(1, 32)
        with patch("src.datasets.imagenet100.pick_target_patches", target), patch(
            "src.datasets.imagenet100.pick_context_patches", context
        ), _reachable_collate():
            collator(batch)
        return collator, seen

    def test_context_call_uses_square_aspect_and_min_keep(self):
        _, seen = self._context_sampling_record()
        kwargs = seen["kwargs"]
        self.assertEqual(kwargs["scale"], PAPER_SCALE_CONTEXT)
        self.assertEqual(kwargs["aspect_ratio"], (1.0, 1.0))
        self.assertEqual(kwargs["min_keep"], 10)
        self.assertNotIn("allow_overlap", kwargs)
        self.assertEqual(len(kwargs["block_size"]), 2)
        self.assertEqual(seen["targets"], 4)

    def test_context_call_uses_the_grid_side(self):
        collator, seen = self._context_sampling_record()
        self.assertEqual(seen["grid"], collator.height)
        self.assertEqual(seen["grid"], collator.width)

    def test_target_lists_truncate_to_one_shared_prefix(self):
        plans = iter([[9, 8, 10, 8], [7, 9, 8, 6]])

        def target(grid, M=4, **kwargs):
            return [torch.arange(length) for length in next(plans)]

        def context(grid, targets, **kwargs):
            return torch.arange(5)

        collator = _small_collator(M=4)
        batch, _ = _pairs(2, 32)
        with patch("src.datasets.imagenet100.pick_target_patches", target), patch(
            "src.datasets.imagenet100.pick_context_patches", context
        ), _reachable_collate():
            _, _, masks_pred = collator(batch)

        self.assertIsInstance(masks_pred, list)
        self.assertEqual(len(masks_pred), 4)
        expected = torch.arange(6).expand(2, 6)
        for mask in masks_pred:
            self.assertEqual(mask.dtype, torch.long)
            torch.testing.assert_close(mask, expected)

    def test_context_masks_are_one_list_entry_per_block(self):
        def target(grid, M=4, **kwargs):
            return [torch.arange(4) for _ in range(M)]

        def context(grid, targets, **kwargs):
            return torch.arange(6)

        collator = _small_collator()
        batch, _ = _pairs(2, 32)
        with patch("src.datasets.imagenet100.pick_target_patches", target), patch(
            "src.datasets.imagenet100.pick_context_patches", context
        ), _reachable_collate():
            _, masks_enc, _ = collator(batch)

        self.assertIsInstance(masks_enc, list)
        self.assertEqual(len(masks_enc), 1)
        self.assertEqual(tuple(masks_enc[0].shape), (2, 6))
        self.assertEqual(masks_enc[0].dtype, torch.long)

    def test_each_batch_advances_the_shared_counter(self):
        def target(grid, M=4, **kwargs):
            return [torch.arange(4) for _ in range(M)]

        def context(grid, targets, **kwargs):
            return torch.arange(6)

        collator = _small_collator()
        batch, _ = _pairs(1, 32)
        self.assertEqual(collator._itr_counter.value, -1)
        with patch("src.datasets.imagenet100.pick_target_patches", target), patch(
            "src.datasets.imagenet100.pick_context_patches", context
        ), _reachable_collate():
            collator(batch)
            collator(batch)
        self.assertEqual(collator._itr_counter.value, 1)

    def test_one_block_size_draw_per_batch(self):
        draws = {"n": 0}

        def rand(*args, **kwargs):
            draws["n"] += 1
            return torch.tensor(0.0)

        def randint(low, high, *args, **kwargs):
            return torch.tensor(0)

        collator = _small_collator()
        batch, _ = _pairs(2, 32)
        with patch("src.masks.torch.rand", rand), patch("src.masks.torch.randint", randint), _reachable_collate():
            collator(batch)
        self.assertEqual(draws["n"], 2)

    def test_scripted_batch_excludes_targets_and_fits_the_grid(self):
        collator = _small_collator(min_keep=1)
        batch, images = _pairs(2, 32, value=0.25)
        with _scripted_sampler(0.0, 0), _reachable_collate():
            collated, masks_enc, masks_pred = collator(batch)

        stacked, labels = collated
        torch.testing.assert_close(stacked, torch.stack(images))
        self.assertEqual(labels.tolist(), [0, 1])

        context = masks_enc[0] if isinstance(masks_enc, list) else masks_enc
        banned = torch.unique(torch.cat([mask.reshape(-1) for mask in masks_pred], dim=0))
        kept = torch.unique(context.reshape(-1))
        self.assertEqual(torch.isin(kept, banned).sum().item(), 0)

        n = collator.height * collator.width
        blocks = list(masks_pred)
        if isinstance(masks_enc, list):
            blocks = list(masks_enc) + blocks
        else:
            blocks.append(masks_enc)
        for mask in blocks:
            self.assertEqual(mask.dtype, torch.long)
            self.assertEqual(mask.shape[0], 2)
            self.assertGreater(mask.shape[1], collator.min_keep)
            self.assertTrue(torch.all((mask >= 0) & (mask < n)))
            for row in mask:
                self.assertEqual(row.numel(), torch.unique(row).numel())

    def test_training_batch_matches_the_reference_interface(self):
        torch.manual_seed(0)
        collator = MaskCollator(
            input_size=224,
            patch_size=14,
            enc_mask_scale=PAPER_SCALE_CONTEXT,
            pred_mask_scale=PAPER_SCALE_TARGET,
            aspect_ratio=PAPER_ASPECT,
            M=4,
            min_keep=10,
        )
        batch, images = _pairs(3, 224)
        collated, masks_enc, masks_pred = collator(batch)

        stacked, labels = collated
        torch.testing.assert_close(stacked, torch.stack(images))
        self.assertEqual(labels.tolist(), [0, 1, 2])
        self.assertEqual(tuple(stacked.shape), (3, 3, 224, 224))

        self.assertIsInstance(masks_enc, list)
        self.assertEqual(len(masks_enc), 1)
        self.assertIsInstance(masks_pred, list)
        self.assertEqual(len(masks_pred), 4)

        n = collator.height * collator.width
        for group in (masks_enc, masks_pred):
            self.assertEqual(len({tuple(mask.shape) for mask in group}), 1)
            for mask in group:
                self.assertEqual(mask.dtype, torch.long)
                self.assertEqual(mask.shape[0], 3)
                self.assertGreater(mask.shape[1], collator.min_keep)
                self.assertTrue(torch.all((mask >= 0) & (mask < n)))
                for row in mask:
                    self.assertEqual(row.numel(), torch.unique(row).numel())
                    self.assertTrue(torch.equal(row, row.sort().values))

        self.assertEqual(collator._itr_counter.value, 0)

        tokens = torch.arange(3 * n, dtype=torch.float32).reshape(3, n, 1)
        gathered = _apply_masks(tokens, masks_enc)
        self.assertEqual(tuple(gathered.shape), (3, masks_enc[0].shape[1], 1))
        pred = _apply_masks(tokens, masks_pred)
        self.assertEqual(tuple(pred.shape), (12, masks_pred[0].shape[1], 1))
        self.assertTrue(torch.isfinite(gathered).all())
        self.assertTrue(torch.isfinite(pred).all())

    def test_batch_size_one_keeps_a_leading_dimension(self):
        collator = _small_collator(min_keep=1)
        batch, _ = _pairs(1, 32)
        with _scripted_sampler(0.0, 0), _reachable_collate():
            _, masks_enc, masks_pred = collator(batch)

        context = masks_enc if isinstance(masks_enc, list) else [masks_enc]
        self.assertEqual(context[0].shape[0], 1)
        self.assertEqual(masks_pred[0].shape[0], 1)
        self.assertEqual(len(masks_pred), 4)

    def test_context_rows_follow_images_through_gather(self):
        cursor = {"i": 0}

        def target(grid, M=4, **kwargs):
            return [torch.arange(4) for _ in range(M)]

        def context(grid, targets, **kwargs):
            index = cursor["i"]
            cursor["i"] += 1
            return torch.tensor([index, index + 3])

        collator = _small_collator()
        batch, _ = _pairs(2, 32)
        with patch("src.datasets.imagenet100.pick_target_patches", target), patch(
            "src.datasets.imagenet100.pick_context_patches", context
        ), _reachable_collate():
            _, masks_enc, _ = collator(batch)

        context = masks_enc[0] if isinstance(masks_enc, list) else masks_enc
        tokens = torch.arange(2 * 16, dtype=torch.float32).reshape(2, 16, 1)
        selected = select_tokens(tokens, context)
        expected = torch.tensor([[[0.0], [3.0]], [[17.0], [20.0]]])
        torch.testing.assert_close(selected, expected)

        embed = PatchEmbed(patch_size=8, in_channels=3, embed_dim=4)
        images = torch.randn(2, 3, 32, 32, requires_grad=True)
        out = select_tokens(embed(images), context)
        self.assertEqual(tuple(out.shape), (2, 2, 4))
        self.assertTrue(torch.isfinite(out).all())
        out.sum().backward()
        self.assertGreater(images.grad.abs().sum().item(), 0)


class TestMaskIndexAndDataPath(unittest.TestCase):
    def test_patch_embedding_uses_row_times_width_plus_column(self):
        embed = PatchEmbed(patch_size=4, in_channels=3, embed_dim=1)
        with torch.no_grad():
            embed.proj.weight.zero_()
            embed.proj.bias.zero_()
            embed.proj.weight[:, :, 0, 0] = 1

        image = torch.zeros(1, 3, 8, 8)
        image[:, :, 4, 0] = 1
        tokens = embed(image).view(-1)
        self.assertEqual(int(tokens.argmax()), 1 * 2 + 0)
        self.assertEqual(float(tokens.sum()), 3.0)

    def test_excluded_target_patches_do_not_change_the_context_encoder(self):
        grid = 8
        target = _rectangle(grid, 0, 0, 3, 3)
        with _scripted_sampler(0.0, 0):
            context = pick_context_patches(
                grid,
                [target],
                block_size=(7, 7),
                aspect_ratio=(1.0, 1.0),
                scale=(1.0, 1.0),
                min_keep=30,
            )
        overlapped = torch.arange(grid * grid)

        embed = PatchEmbed(patch_size=4, in_channels=3, embed_dim=8)
        images = torch.randn(2, 3, 32, 32)
        changed = images.clone()
        changed[:, :, :12, :12] = 10
        indices = context.unsqueeze(0).expand(2, -1).contiguous()
        overlap = overlapped.unsqueeze(0).expand(2, -1).contiguous()

        def hidden(image, mask):
            return select_tokens(embed(image), mask)

        torch.testing.assert_close(hidden(images, indices), hidden(changed, indices), rtol=0, atol=0)
        self.assertGreater((hidden(images, overlap) - hidden(changed, overlap)).abs().sum().item(), 0)
        visible = images.detach().requires_grad_(True)
        loss = hidden(visible, indices).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(visible.grad[:, :, 12:, :].abs().sum().item(), 0)
        torch.testing.assert_close(
            visible.grad[:, :, :12, :12], torch.zeros_like(visible.grad[:, :, :12, :12]), rtol=0, atol=0
        )

    def test_transform_is_crop_tensor_and_imagenet_normalization(self):
        transform = make_transform()
        crop, to_tensor, normalize = transform.transforms
        default_crop = T.RandomResizedCrop(224)

        self.assertIsInstance(crop, T.RandomResizedCrop)
        self.assertEqual(tuple(crop.size), (224, 224))
        self.assertEqual(crop.scale, (0.3, 1.0))
        self.assertEqual(crop.ratio, default_crop.ratio)
        self.assertEqual(crop.interpolation, default_crop.interpolation)
        self.assertIsInstance(to_tensor, T.ToTensor)
        self.assertEqual(tuple(normalize.mean), (0.485, 0.456, 0.406))
        self.assertEqual(tuple(normalize.std), (0.229, 0.224, 0.225))
        self.assertEqual(
            [type(step) for step in transform.transforms],
            [T.RandomResizedCrop, T.ToTensor, T.Normalize],
        )

    def test_loader_uses_this_collator_without_worker_generator(self):
        collator = _small_collator()
        with tempfile.TemporaryDirectory() as root:
            _write_subset(root)
            _, loader, sampler = make_imagenet100(
                transform=make_transform(crop_size=32),
                batch_size=2,
                collator=collator,
                pin_mem=True,
                num_workers=2,
                world_size=1,
                rank=0,
                root_path=root,
                training=True,
                drop_last=True,
            )

        self.assertIs(loader.collate_fn, collator)
        self.assertTrue(loader.drop_last)
        self.assertFalse(loader.persistent_workers)
        self.assertIsNone(loader.worker_init_fn)
        self.assertIsNone(loader.generator)
        self.assertTrue(loader.pin_memory)
        self.assertEqual(loader.num_workers, 2)
        self.assertEqual(sampler.num_replicas, 1)
        self.assertEqual(sampler.rank, 0)

    def test_worker_loader_returns_a_batch(self):
        collator = _small_collator(patch_size=4, min_keep=1)
        loader = torch.utils.data.DataLoader(
            _Tensors(),
            batch_size=2,
            collate_fn=collator,
            num_workers=2,
            drop_last=True,
            persistent_workers=False,
            multiprocessing_context=multiprocessing.get_context("fork"),
        )
        batch = next(iter(loader))
        self.assertEqual(len(batch), 3)
        images, _ = batch[0]
        self.assertEqual(tuple(images.shape), (2, 3, 32, 32))

if __name__ == "__main__":
    unittest.main()
