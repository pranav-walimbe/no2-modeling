"""Tests for the AOI score percentile visualization."""

import tempfile
import unittest
from pathlib import Path

import polars as pl
from preprocessing.stratify_plants import _plot_aoi_score_percentiles


class AoiScorePlotTest(unittest.TestCase):
    def test_plot_is_written_for_coal_containing_aois(self) -> None:
        features = pl.DataFrame(
            {
                "num_coal_units": [0, 1, 2],
                "avg_coal_nox": [500.0, 10.0, 20.0],
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "aoi-scores.png"
            _plot_aoi_score_percentiles(features, output_path)

            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
