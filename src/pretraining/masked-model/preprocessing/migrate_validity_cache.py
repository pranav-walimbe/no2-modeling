"""Migrate legacy validity-cache files into a compact source-cache index."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from dataset_generation_utils import (
    INVALID_STATUS,
    VALID_STATUS,
    VALIDITY_INDEX_SCHEMA,
    candidate_cache_tasks,
    merge_validity_frames,
    write_parquet_atomic,
)

from config import (
    DATASET_TEMPO_CACHE_DIR,
    DATASET_WEATHER_CACHE_DIR,
    HRRR_DIR,
    MASKED_PRETRAINING_VALIDITY_CACHE_DIR,
    MASKED_PRETRAINING_VALIDITY_INDEX,
    MASKED_PRETRAINING_WORK_DIR,
    TEMPO_DIR,
)

SPLITS = ("train", "val", "test")
PROGRESS_INTERVAL = 50_000


def parse_args() -> argparse.Namespace:
    """Parse migration options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delete-legacy-files",
        action="store_true",
        help="delete legacy valid NPZ and invalid JSON files after the index is safely published",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing validity index")
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _legacy_valid_paths() -> list[Path]:
    return sorted((Path(MASKED_PRETRAINING_VALIDITY_CACHE_DIR) / "valid").glob("*/*.npz"))


def _legacy_invalid_paths() -> list[Path]:
    return sorted((Path(MASKED_PRETRAINING_VALIDITY_CACHE_DIR) / "invalid").glob("*/*.json"))


def _valid_results(split: str) -> pl.DataFrame:
    paths = sorted((Path(MASKED_PRETRAINING_WORK_DIR) / "validity-results" / split).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No legacy validity results found for {split}")
    return (
        pl.concat([pl.read_parquet(path) for path in paths], how="vertical")
        .select("candidate_index", "cache_key")
        .unique(subset="cache_key", keep="first")
    )


def _mapped_valid_rows(split: str) -> list[dict[str, object]]:
    results = _valid_results(split)
    candidate_path = Path(MASKED_PRETRAINING_WORK_DIR) / "candidates" / f"{split}.parquet"
    candidates = (
        pl.scan_parquet(candidate_path)
        .join(results.lazy(), on="candidate_index", how="inner")
        .collect(engine="streaming")
    )
    if candidates.height != results.height:
        raise ValueError(f"[{split}] mapped {candidates.height:,}/{results.height:,} legacy valid results")

    rows: list[dict[str, object]] = []
    lookup_started = time.monotonic()
    for index, candidate in enumerate(candidates.iter_rows(named=True), start=1):
        scan, weather = candidate_cache_tasks(
            candidate,
            tempo_root=Path(TEMPO_DIR),
            tempo_cache_dir=Path(DATASET_TEMPO_CACHE_DIR),
            hrrr_root=Path(HRRR_DIR),
            weather_cache_dir=Path(DATASET_WEATHER_CACHE_DIR),
        )
        expected_key = str(candidate["cache_key"])
        if scan.cache_key != expected_key:
            raise ValueError(f"[{split}] candidate cache key changed: {expected_key}")
        if not Path(scan.cache_path).is_file():
            raise FileNotFoundError(f"Missing TEMPO cache entry for {expected_key}: {scan.cache_path}")
        if not Path(weather.cache_path).is_file():
            raise FileNotFoundError(f"Missing weather cache entry for {expected_key}: {weather.cache_path}")
        rows.append(
            {
                "cache_key": expected_key,
                "status": VALID_STATUS,
                "tempo_cache_path": scan.cache_path,
                "weather_cache_path": weather.cache_path,
                "reason": None,
            }
        )
        if index % PROGRESS_INTERVAL == 0 or index == candidates.height:
            elapsed = time.monotonic() - lookup_started
            print(
                f"[{_timestamp()}] [{split}] verified {index:,}/{candidates.height:,} source-cache pairs "
                f"in {elapsed:.1f}s"
            )
    return rows


def _remove_legacy_files(paths: list[Path]) -> None:
    started = time.monotonic()
    for index, path in enumerate(paths, start=1):
        path.unlink()
        if index % PROGRESS_INTERVAL == 0 or index == len(paths):
            print(f"[{_timestamp()}] deleted {index:,}/{len(paths):,} legacy files")
    for directory in sorted({path.parent for path in paths}):
        directory.rmdir()
    print(f"[{_timestamp()}] legacy deletion finished in {time.monotonic() - started:.1f}s")


def main() -> None:
    """Build the validity index and optionally remove its legacy files."""
    args = parse_args()
    destination = Path(MASKED_PRETRAINING_VALIDITY_INDEX)
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"Validity index already exists: {destination}. Use --overwrite to replace it.")

    valid_paths = _legacy_valid_paths()
    invalid_paths = _legacy_invalid_paths()
    print(
        f"[{_timestamp()}] legacy inventory: {len(valid_paths):,} valid rasters and "
        f"{len(invalid_paths):,} invalid records"
    )
    valid_rows = [row for split in SPLITS for row in _mapped_valid_rows(split)]
    mapped_keys = {str(row["cache_key"]) for row in valid_rows}
    legacy_valid_keys = {path.stem for path in valid_paths}
    if mapped_keys != legacy_valid_keys:
        missing = legacy_valid_keys - mapped_keys
        extra = mapped_keys - legacy_valid_keys
        raise ValueError(
            f"Legacy/index mismatch: {len(missing):,} unmapped rasters and {len(extra):,} missing legacy rasters"
        )

    invalid_rows = [
        {
            "cache_key": path.stem,
            "status": INVALID_STATUS,
            "tempo_cache_path": None,
            "weather_cache_path": None,
            "reason": "legacy invalid scene",
        }
        for path in invalid_paths
        if path.stem not in mapped_keys
    ]
    index = merge_validity_frames(
        [
            pl.DataFrame(valid_rows, schema=VALIDITY_INDEX_SCHEMA),
            pl.DataFrame(invalid_rows, schema=VALIDITY_INDEX_SCHEMA),
        ]
    )
    write_parquet_atomic(index, destination)
    print(f"[{_timestamp()}] published {index.height:,} entries to {destination}")

    if args.delete_legacy_files:
        _remove_legacy_files([*valid_paths, *invalid_paths])
        for directory in (
            Path(MASKED_PRETRAINING_VALIDITY_CACHE_DIR) / "valid",
            Path(MASKED_PRETRAINING_VALIDITY_CACHE_DIR) / "invalid",
        ):
            directory.rmdir()
        print(f"[{_timestamp()}] removed the legacy validity-cache tree")
    else:
        print("Legacy files retained. Re-run with --overwrite --delete-legacy-files after reviewing the index.")


if __name__ == "__main__":
    main()
