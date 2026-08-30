"""Tests for normalized hourly regional power-price collection."""

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from collection.power_price_sources import _parse_ercot_rtm_intervals
from collection.scrape_power_prices import (
    aggregate_hourly,
    month_windows,
    normalize_prices,
    scrape_iso,
    write_partition,
)


class MonthWindowsTest(unittest.TestCase):
    """Verify inclusive CLI dates become half-open UTC partitions."""

    def test_partial_months_preserve_requested_bounds(self) -> None:
        windows = month_windows(date(2023, 8, 15), date(2023, 9, 2))

        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].start_utc, pd.Timestamp("2023-08-15", tz="UTC"))
        self.assertEqual(windows[0].end_utc, pd.Timestamp("2023-09-01", tz="UTC"))
        self.assertEqual(windows[1].end_utc, pd.Timestamp("2023-09-03", tz="UTC"))


class HourlyAggregationTest(unittest.TestCase):
    """Verify native intervals become duration-weighted UTC hours."""

    def test_aggregates_four_fifteen_minute_prices(self) -> None:
        starts = pd.date_range("2024-01-01 00:00", periods=4, freq="15min", tz="US/Central")
        raw = pd.DataFrame(
            {
                "Interval Start": starts,
                "Interval End": starts + pd.Timedelta(minutes=15),
                "Location": ["HB_NORTH"] * 4,
                "Location Type": ["Trading Hub"] * 4,
                "SPP": [20.0, 20.0, 40.0, 40.0],
            },
        )
        normalized = normalize_prices(
            raw,
            iso="ERCOT",
            market="real_time",
            settlement_status="published",
            retrieved_at=pd.Timestamp("2024-01-02", tz="UTC"),
        )

        hourly = aggregate_hourly(normalized)

        self.assertEqual(len(hourly), 1)
        self.assertEqual(hourly.loc[0, "interval_start_utc"], pd.Timestamp("2024-01-01 06:00", tz="UTC"))
        self.assertEqual(hourly.loc[0, "price_usd_mwh"], 30.0)
        self.assertEqual(hourly.loc[0, "location_type"], "hub")
        self.assertEqual(hourly.loc[0, "source_interval_minutes"], 15.0)
        self.assertEqual(hourly.loc[0, "interval_count"], 4)

    def test_rejects_incomplete_hour(self) -> None:
        starts = pd.date_range("2024-01-01 00:00", periods=3, freq="15min", tz="UTC")
        raw = pd.DataFrame(
            {
                "Interval Start": starts,
                "Interval End": starts + pd.Timedelta(minutes=15),
                "Location": ["ZONE_A"] * 3,
                "Location Type": ["Load Zone"] * 3,
                "LMP": [10.0, 20.0, 30.0],
            },
        )
        normalized = normalize_prices(
            raw,
            iso="TEST",
            market="real_time",
            settlement_status="final",
            retrieved_at=pd.Timestamp("2024-01-02", tz="UTC"),
        )

        with self.assertRaisesRegex(ValueError, "incomplete hourly"):
            aggregate_hourly(normalized)

    def test_ignores_unused_categorical_location_types(self) -> None:
        raw = pd.DataFrame(
            {
                "Interval Start": [pd.Timestamp("2024-01-01", tz="UTC")],
                "Interval End": [pd.Timestamp("2024-01-01 01:00", tz="UTC")],
                "Location": ["LZ_NORTH"],
                "Location Type": pd.Categorical(
                    ["Load Zone"],
                    categories=["Load Zone", "Load Zone Energy Weighted"],
                ),
                "LMP": [20.0],
            },
        )

        normalized = normalize_prices(
            raw,
            iso="ERCOT",
            market="real_time",
            settlement_status="published",
            retrieved_at=pd.Timestamp("2024-01-02", tz="UTC"),
        )

        self.assertEqual(normalized["location_type"].tolist(), ["load_zone"])


class ERCOTArchiveParsingTest(unittest.TestCase):
    """Verify integer ERCOT delivery hours parse with current pandas."""

    def test_parses_fifteen_minute_delivery_intervals(self) -> None:
        raw = pd.DataFrame(
            {
                "Delivery Date": ["2026-01-01"] * 4,
                "Delivery Hour": [1] * 4,
                "Delivery Interval": [1, 2, 3, 4],
                "Repeated Hour Flag": ["N"] * 4,
                "Settlement Point Type": ["LZ"] * 4,
            },
        )

        parsed = _parse_ercot_rtm_intervals(raw, "US/Central")

        expected = pd.date_range("2026-01-01 00:00", periods=4, freq="15min", tz="US/Central")
        self.assertEqual(parsed["Interval Start"].tolist(), expected.tolist())
        self.assertEqual(parsed["SettlementPointType"].unique().tolist(), ["LZ"])


class PartitionWriteTest(unittest.TestCase):
    """Verify Parquet partitions are written without a lingering partial file."""

    def test_writes_atomic_parquet_partition(self) -> None:
        frame = pd.DataFrame(
            {
                "iso": ["TEST"],
                "market": ["day_ahead"],
                "location_id": ["ZONE_A"],
                "interval_start_utc": [pd.Timestamp("2024-01-01", tz="UTC")],
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_path = root / "hourly" / "prices.parquet"

            write_partition(frame, output_path, root / "temporary")

            self.assertTrue(output_path.is_file())
            self.assertEqual(pd.read_parquet(output_path).loc[0, "location_id"], "ZONE_A")
            self.assertEqual(list((root / "temporary").glob("*.part")), [])


class ScrapeISOTest(unittest.TestCase):
    """Verify a source backfill writes both markets and metadata."""

    @patch("collection.scrape_power_prices.time.sleep")
    @patch("collection.scrape_power_prices.create_source")
    def test_writes_resumable_month_partition(self, create_source, _sleep) -> None:
        starts = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        raw = pd.DataFrame(
            {
                "Interval Start": starts,
                "Interval End": starts + pd.Timedelta(hours=1),
                "Location": ["ZONE_A"] * 24,
                "Location Type": ["Load Zone"] * 24,
                "LMP": range(24),
            },
        )
        source = create_source.return_value
        source.timezone = "UTC"
        source.fetch.return_value = raw
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            scrape_iso(
                iso="TEST",
                start=date(2024, 1, 1),
                end=date(2024, 1, 1),
                hourly_dir=root / "hourly",
                metadata_dir=root / "metadata",
                temporary_dir=root / "temporary",
            )

            for market in ("day_ahead", "real_time"):
                output = (
                    root
                    / "hourly"
                    / "iso=TEST"
                    / f"market={market}"
                    / "year=2024"
                    / "month=01"
                    / "prices.parquet"
                )
                self.assertEqual(len(pd.read_parquet(output)), 24)
            self.assertTrue((root / "metadata" / "locations" / "iso=TEST" / "locations.parquet").is_file())
            self.assertTrue((root / "metadata" / "manifests" / "TEST.json").is_file())


if __name__ == "__main__":
    unittest.main()
