"""Regression tests for generated-dataset class balancing."""

import unittest
from datetime import date

import polars as pl

from config import LABEL_COL, MIN_TIMESTEP_NO2_FINITE_FRACTION
from preprocessing.generate_dataset_utils import MIN_NO2_FINITE_FRACTION_COL, select_final_records
from preprocessing.stratify_utils import AOI_ID_COL


class DatasetBalancingTest(unittest.TestCase):
    """Verify deterministic minority oversampling without majority loss."""

    def test_coverage_threshold_is_95_percent(self) -> None:
        self.assertEqual(MIN_TIMESTEP_NO2_FINITE_FRACTION, 0.95)

    def test_minority_rows_are_duplicated_to_majority_size(self) -> None:
        candidates = self._frame(labels=[0, 0, 1, 1, 1])

        selected = select_final_records(candidates)

        counts = dict(selected.group_by(LABEL_COL).len().iter_rows())
        self.assertEqual(counts, {0: 3, 1: 3})
        self.assertEqual(
            selected.filter(pl.col(LABEL_COL) == 1)["record_id"].sort().to_list(),
            [2, 3, 4],
        )
        minority_ids = selected.filter(pl.col(LABEL_COL) == 0)["record_id"].to_list()
        self.assertEqual(set(minority_ids), {0, 1})
        self.assertEqual(len(minority_ids), 3)

    def test_oversampling_is_deterministic(self) -> None:
        candidates = self._frame(labels=[0, 1, 1, 1, 1])

        first = select_final_records(candidates)
        second = select_final_records(candidates)

        self.assertTrue(first.equals(second))
        self.assertEqual(first.filter(pl.col(LABEL_COL) == 0).height, 4)

    def test_balancing_requires_both_classes(self) -> None:
        candidates = self._frame(labels=[1, 1])

        with self.assertRaisesRegex(ValueError, "without both classes"):
            select_final_records(candidates)

    @staticmethod
    def _frame(labels: list[int]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "record_id": list(range(len(labels))),
                AOI_ID_COL: [100 + index % 2 for index in range(len(labels))],
                "date": [date(2026, 1, index + 1) for index in range(len(labels))],
                "hour": [12 + index % 4 for index in range(len(labels))],
                MIN_NO2_FINITE_FRACTION_COL: [0.95 + 0.01 * (index % 3) for index in range(len(labels))],
                LABEL_COL: labels,
            }
        )


if __name__ == "__main__":
    unittest.main()
