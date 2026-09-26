"""Tests for AOI eligibility and class-balanced record selection."""

import unittest
from datetime import datetime, timedelta, timezone

import polars as pl
from preprocessing.stratify_plants import (
    DELTA_CATEGORY_COL,
    EMA_INNOVATION_COL,
    HYBRID_THRESHOLD_COL,
    filter_aoi_class_floor,
    select_balanced_ema_records,
)
from preprocessing.stratify_utils import AOI_ID_COL


class StratificationSelectionTest(unittest.TestCase):
    def test_aoi_floor_requires_twenty_records_in_every_class(self) -> None:
        rows = []
        for aoi_id, decrease_count in ((1, 20), (2, 19)):
            for class_name, count in (
                ("decrease", decrease_count),
                ("steady", 20),
                ("increase", 20),
            ):
                rows.extend({AOI_ID_COL: aoi_id, DELTA_CATEGORY_COL: class_name} for _ in range(count))

        eligible, audit = filter_aoi_class_floor(pl.DataFrame(rows), minimum_per_class=20)

        self.assertEqual(eligible[AOI_ID_COL].unique().to_list(), [1])
        self.assertTrue(audit.filter(pl.col(AOI_ID_COL) == 1)["meets_class_floor"].item())
        self.assertFalse(audit.filter(pl.col(AOI_ID_COL) == 2)["meets_class_floor"].item())

    def test_steady_downsampling_favors_innovations_near_zero(self) -> None:
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        rows = []
        innovations = [-300.0] * 50 + [0.0] * 500 + [198.0] * 500 + [300.0] * 50
        categories = ["decrease"] * 50 + ["steady"] * 1_000 + ["increase"] * 50
        for index, (innovation, category) in enumerate(zip(innovations, categories, strict=True)):
            timestamp = base + timedelta(minutes=index)
            rows.append(
                {
                    AOI_ID_COL: index % 7,
                    "emissions_hour_utc": timestamp,
                    "t3_timestamp": timestamp,
                    EMA_INNOVATION_COL: innovation,
                    HYBRID_THRESHOLD_COL: 200.0,
                    DELTA_CATEGORY_COL: category,
                }
            )

        selected = select_balanced_ema_records(pl.DataFrame(rows), "train")
        steady = selected.filter(pl.col(DELTA_CATEGORY_COL) == "steady")

        self.assertEqual(steady.height, 50)
        self.assertLess(steady[EMA_INNOVATION_COL].abs().mean(), 25.0)


if __name__ == "__main__":
    unittest.main()
