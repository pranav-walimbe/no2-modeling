"""Regression tests for AOI sampling and label-frequency loss weights."""

import unittest

import numpy as np
import polars as pl
import torch

from modeling.losses import HistogramWeightedHuberLoss, build_histogram_weight_config
from preprocessing.stratify_plants import _limit_splits
from preprocessing.stratify_utils import AOI_ID_COL


class StratificationPriorityTest(unittest.TestCase):
    """Verify deterministic train-only AOI prioritization."""

    def test_training_sampling_favors_power_and_coal(self) -> None:
        size = 10_000
        frame = pl.DataFrame(
            {
                "row": np.arange(size),
                AOI_ID_COL: np.arange(size),
                "avg_pwr_gen": np.arange(size, dtype=float),
                "num_coal_units": np.arange(size),
            }
        )

        first = _limit_splits({"train": frame}, {"train": 1_000})["train"]
        second = _limit_splits({"train": frame}, {"train": 1_000})["train"]

        self.assertEqual(first["row"].to_list(), second["row"].to_list())
        self.assertGreater(first["avg_pwr_gen"].mean(), frame["avg_pwr_gen"].mean())
        self.assertGreater(first["num_coal_units"].mean(), frame["num_coal_units"].mean())

    def test_validation_sampling_does_not_require_priority_columns(self) -> None:
        frame = pl.DataFrame({"row": np.arange(100)})

        result = _limit_splits({"val": frame}, {"val": 20})["val"]

        self.assertEqual(result.height, 20)


class HistogramWeightingTest(unittest.TestCase):
    """Verify signed inverse-frequency buckets and their cap."""

    def test_rare_bins_receive_larger_capped_weights(self) -> None:
        labels = np.concatenate((np.full(90, -0.1), np.full(8, 0.1), np.array([-2.0, 2.0])))

        config = build_histogram_weight_config(labels, bin_count=20, weight_cap=5.0)

        self.assertEqual(len(config.bin_counts), 20)
        self.assertIn(0.0, config.bin_edges)
        self.assertLessEqual(max(config.bin_weights), 5.0)
        occupied = [(count, weight) for count, weight in zip(config.bin_counts, config.bin_weights) if count]
        common_weight = min(occupied, key=lambda item: item[1])[1]
        rare_weight = max(occupied, key=lambda item: item[1])[1]
        self.assertGreater(rare_weight, common_weight)

    def test_weighted_huber_uses_target_bucket(self) -> None:
        labels = np.array([-2.0, -0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 2.0])
        config = build_histogram_weight_config(labels, bin_count=4, weight_cap=5.0)
        criterion = HistogramWeightedHuberLoss(config, delta=1.0, target_mean=1.0, target_std=2.0)

        rare_loss = criterion(torch.tensor([0.0]), torch.tensor([(2.0 - 1.0) / 2.0]))
        common_loss = criterion(torch.tensor([0.0]), torch.tensor([(0.1 - 1.0) / 2.0]))

        self.assertGreater(rare_loss.item(), common_loss.item())


if __name__ == "__main__":
    unittest.main()
