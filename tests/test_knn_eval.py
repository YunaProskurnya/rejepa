import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

import torch

from scripts.evaluate_knn import feature_statistics, select_per_class, weighted_knn


class TestKnnEvaluation(unittest.TestCase):
    def test_joint_checkpoint_is_evaluated_with_shared_encoder(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkpoint_dir = folder / "run"
            training = subprocess.run(
                [sys.executable, str(root / "scripts/train.py"), "--smoke-test", "--mode", "joint",
                 "--max-steps", "1", "--output", str(checkpoint_dir)],
                cwd=root, capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(training.returncode, 0, training.stdout + training.stderr)

            for split in ("train", "val"):
                for name in ("a", "b"):
                    destination = folder / split / name
                    destination.mkdir(parents=True)
                    shutil.copyfile(root / "example/lenna.png", destination / "image.png")

            output = folder / "knn.json"
            evaluation = subprocess.run(
                [sys.executable, str(root / "scripts/evaluate_knn.py"),
                 "--checkpoint", str(checkpoint_dir / "checkpoint.pt"),
                 "--train", str(folder / "train"), "--val", str(folder / "val"),
                 "--reference-per-class", "1", "--queries-per-class", "1", "--k", "1",
                 "--compare-random", "--output", str(output)],
                cwd=root, capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(evaluation.returncode, 0, evaluation.stdout + evaluation.stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["config"]["evaluated_model"], "encoder")
            self.assertEqual(set(result["metrics"]), {"encoder", "random_encoder"})

    def test_chunked_cosine_knn_handles_noncontiguous_class_ids(self):
        reference = torch.tensor([[1., 0.], [3., 0.], [0., 2.], [0., 1.]])
        labels = torch.tensor([3, 3, 9, 9])
        queries = torch.tensor([[10., 1.], [1., 10.], [-0.1, 1.]])
        expected = torch.tensor([3, 9, 9])
        small_chunks = weighted_knn(reference, labels, queries, expected, k=2, chunk_size=1)
        large_chunks = weighted_knn(reference, labels, queries, expected, k=2, chunk_size=20)
        self.assertEqual(small_chunks, large_chunks)
        self.assertEqual(small_chunks["top1"], 100.)
        self.assertEqual(small_chunks["top5"], 100.)

    def test_temperature_changes_weighted_vote(self):
        reference = torch.tensor([[1., 0.], [0.8, 0.6], [0.8, -0.6]])
        labels = torch.tensor([0, 1, 1])
        query = torch.tensor([[1., 0.]])
        expected = torch.tensor([0])
        cold = weighted_knn(reference, labels, query, expected, k=3, temperature=0.07)
        warm = weighted_knn(reference, labels, query, expected, k=3, temperature=1.)
        self.assertEqual(cold["top1"], 100.)
        self.assertEqual(warm["top1"], 0.)

    def test_collapse_and_two_dimensional_features_have_expected_rank(self):
        collapsed = feature_statistics(torch.ones(4, 3))
        self.assertEqual(collapsed["centered_effective_rank"], 0.)
        self.assertEqual(collapsed["normalized_feature_std_mean"], 0.)
        spread = feature_statistics(torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]))
        self.assertAlmostEqual(spread["centered_effective_rank"], 2.)

    def test_balanced_subset_is_reproducible_and_obeys_selected_classes(self):
        dataset = types.SimpleNamespace(
            classes=["a", "b", "c"], class_to_idx={"a": 0, "b": 1, "c": 2},
            samples=[(str(i), i // 5) for i in range(15)],
        )
        first = select_per_class(dataset, ["a", "c"], 3, torch.Generator().manual_seed(7))
        repeat = select_per_class(dataset, ["a", "c"], 3, torch.Generator().manual_seed(7))
        self.assertEqual(first, repeat)
        self.assertEqual([dataset.samples[i][1] for i in first], [0, 0, 0, 2, 2, 2])
        self.assertEqual(len(set(first)), 6)
        with self.assertRaisesRegex(ValueError, "need 6"):
            select_per_class(dataset, ["a"], 6, torch.Generator().manual_seed(0))


if __name__ == "__main__":
    unittest.main()
