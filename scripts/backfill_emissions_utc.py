#!/usr/bin/env python3
"""Backfill UTC timestamps in the existing enriched emissions Parquet."""

import argparse
from pathlib import Path

from collection.scrape_locations import backfill_emissions_utc
from config import FULL_DATA_PARQUET


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path(FULL_DATA_PARQUET))
    return parser.parse_args()


def main() -> None:
    """Convert the configured enriched emissions archive in place."""
    args = parse_args()
    row_count = backfill_emissions_utc(args.path)
    print(f"Backfilled UTC time fields for {row_count:,} rows in {args.path}")


if __name__ == "__main__":
    main()
