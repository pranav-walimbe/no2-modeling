"""Map coal and natural-gas facilities to ISO regional price locations."""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

from config import FULL_DATA_PARQUET, PLANT_ZONE_MAPPING, POWER_PRICE_METADATA_DIR

DEFAULT_BATCH_SIZE = 100_000
DEFAULT_EIA860_YEAR = 2024
EIA860_URL = "https://www.eia.gov/electricity/data/eia860/xls/eia860{year}.zip"

PLANT_COLUMNS = [
    "facilityId",
    "facilityName",
    "stateCode",
    "county",
    "lat",
    "lon",
    "nercRegion",
    "primaryFuelInfo",
]

BA_TO_ISO = {
    "CISO": "CAISO",
    "ERCO": "ERCOT",
    "ISNE": "ISONE",
    "MISO": "MISO",
    "NYIS": "NYISO",
    "PJM": "PJM",
    "SWPP": "SPP",
}

ISONE_STATE_ZONES = {
    "CT": ".Z.CONNECTICUT",
    "ME": ".Z.MAINE",
    "NH": ".Z.NEWHAMPSHIRE",
    "RI": ".Z.RHODEISLAND",
    "VT": ".Z.VERMONT",
}
ISONE_MASSACHUSETTS_COUNTIES = {
    ".Z.SEMASS": {"barnstable", "bristol", "dukes", "nantucket", "plymouth"},
    ".Z.WCMASS": {"berkshire", "franklin", "hampden", "hampshire", "worcester"},
    ".Z.NEMASSBOST": {"essex", "middlesex", "suffolk"},
}

NYISO_COUNTY_ZONES = {
    "WEST": {"allegany", "cattaraugus", "chautauqua", "erie", "niagara"},
    "GENESE": {"genesee", "livingston", "monroe", "ontario", "orleans", "wyoming"},
    "CENTRL": {
        "broome",
        "cayuga",
        "chenango",
        "chemung",
        "cortland",
        "onondaga",
        "oswego",
        "schuyler",
        "seneca",
        "steuben",
        "tioga",
        "tompkins",
        "wayne",
        "yates",
    },
    "NORTH": {
        "clinton",
        "essex",
        "franklin",
        "hamilton",
        "jefferson",
        "lewis",
        "st. lawrence",
        "st lawrence",
    },
    "MHK VL": {"fulton", "herkimer", "madison", "montgomery", "oneida", "otsego", "schoharie"},
    "CAPITL": {"albany", "columbia", "greene", "rensselaer", "saratoga", "schenectady", "warren", "washington"},
    "HUD VL": {"delaware", "dutchess", "orange", "putnam", "rockland", "sullivan", "ulster"},
    "DUNWOD": {"westchester"},
    "N.Y.C.": {"bronx", "kings", "new york", "queens", "richmond"},
    "LONGIL": {"nassau", "suffolk"},
}

CAISO_OWNER_ZONES = {
    "pacific gas and electric": "DLAP_PGAE-APND",
    "pacific gas & electric": "DLAP_PGAE-APND",
    "southern california edison": "DLAP_SCE-APND",
    "san diego gas": "DLAP_SDGE-APND",
    "valley electric": "DLAP_VEA-APND",
}

MISO_STATE_HUBS = {
    "AR": "ARKANSAS.HUB",
    "IL": "ILLINOIS.HUB",
    "IN": "INDIANA.HUB",
    "LA": "LOUISIANA.HUB",
    "MI": "MICHIGAN.HUB",
    "MN": "MINN.HUB",
    "MS": "MS.HUB",
    "TX": "TEXAS.HUB",
}

SPP_NORTH_STATES = {"IA", "KS", "MO", "ND", "NE", "SD"}
SPP_SOUTH_STATES = {"AR", "LA", "NM", "OK", "TX"}

ZONE_CENTROIDS = {
    "CAISO": {
        "DLAP_PGAE-APND": (37.6, -121.5),
        "DLAP_SCE-APND": (34.0, -117.8),
        "DLAP_SDGE-APND": (32.8, -117.0),
        "DLAP_VEA-APND": (36.0, -115.5),
    },
    "ERCOT": {
        "LZ_AEN": (32.0, -96.8),
        "LZ_CPS": (29.4, -98.5),
        "LZ_HOUSTON": (29.8, -95.4),
        "LZ_LCRA": (30.3, -97.7),
        "LZ_NORTH": (32.8, -97.2),
        "LZ_RAYBN": (33.0, -96.0),
        "LZ_SOUTH": (27.8, -98.0),
        "LZ_WEST": (31.5, -102.0),
    },
}

PJM_OWNER_ZONES = {
    "atlantic city electric": "AECO",
    "american electric power": "AEP",
    "baltimore gas": "BGE",
    "commonwealth edison": "COMED",
    "dayton power": "DAY",
    "delmarva power": "DPL",
    "dominion energy": "DOM",
    "duquesne light": "DUQ",
    "jersey central": "JCPL",
    "metropolitan edison": "METED",
    "peco energy": "PECO",
    "pennsylvania electric": "PENELEC",
    "ppl electric": "PPL",
    "public service electric": "PSEG",
}

NAME_STOPWORDS = {
    "energy",
    "generating",
    "generation",
    "plant",
    "power",
    "project",
    "station",
    "steam",
}


@dataclass(frozen=True)
class RegionalMapping:
    """One regional-price assignment and its provenance."""

    price_location_name: str | None
    mapping_method: str
    confidence: str
    needs_review: bool
    note: str
    distance_km: float | None = None


def collect_unique_fossil_plants(input_path: Path, batch_size: int) -> pd.DataFrame:
    """Collect one coal/gas row per facility without loading hourly data at once."""
    parquet = pq.ParquetFile(input_path)
    missing = set(PLANT_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Input parquet is missing columns: {sorted(missing)}")

    records: dict[int, dict[str, Any]] = {}
    for batch in parquet.iter_batches(batch_size=batch_size, columns=PLANT_COLUMNS, use_threads=True):
        fuel = pc.utf8_lower(pc.fill_null(batch.column("primaryFuelInfo"), ""))
        fossil = pc.or_(pc.match_substring(fuel, "coal"), pc.match_substring(fuel, "natural gas"))
        filtered = batch.filter(fossil)
        if filtered.num_rows == 0:
            continue

        frame = filtered.to_pandas().dropna(subset=["facilityId"])
        normalized = frame["primaryFuelInfo"].astype("string").str.casefold()
        coal = frame.loc[normalized.str.contains("coal", na=False)].assign(fuel_type="coal")
        gas = frame.loc[normalized.str.contains("natural gas", na=False)].assign(fuel_type="natural_gas")
        frame = pd.concat([coal, gas], ignore_index=True)
        frame = frame.drop_duplicates(["facilityId", "fuel_type"])
        for row in frame.itertuples(index=False):
            facility_id = int(row.facilityId)
            record = records.setdefault(
                facility_id,
                {
                    "facility_id": facility_id,
                    "facility_name": row.facilityName,
                    "fuel_types": set(),
                    "state": row.stateCode,
                    "county": row.county,
                    "latitude": row.lat,
                    "longitude": row.lon,
                    "nerc_region": row.nercRegion,
                },
            )
            record["fuel_types"].add(str(row.fuel_type))

    if not records:
        raise RuntimeError("No coal or natural-gas facilities were found")
    rows = []
    for record in records.values():
        rows.append({**record, "fuel_type": "|".join(sorted(record["fuel_types"]))})
        rows[-1].pop("fuel_types")
    return pd.DataFrame(rows).sort_values("facility_id").reset_index(drop=True)


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                shutil.copyfileobj(response.raw, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path.replace(destination)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise


def load_eia860_plants(archive_path: Path, year: int, allow_download: bool) -> pd.DataFrame:
    """Read the EIA-860 plant table used to identify each plant's ISO."""
    if not archive_path.exists():
        if not allow_download:
            raise FileNotFoundError(f"EIA-860 archive not found: {archive_path}")
        url = EIA860_URL.format(year=year)
        print(f"Downloading {url}")
        _download_file(url, archive_path)

    columns = {
        "Plant Code": "facility_id",
        "Plant Name": "eia_plant_name",
        "Balancing Authority Code": "balancing_authority_code",
        "Balancing Authority Name": "balancing_authority_name",
        "Transmission or Distribution System Owner": "transmission_owner",
        "Transmission or Distribution System Owner ID": "transmission_owner_id",
    }
    with ZipFile(archive_path) as archive:
        candidates = [name for name in archive.namelist() if "Plant" in name and name.endswith(".xlsx")]
        if len(candidates) != 1:
            raise ValueError(f"Expected one EIA-860 plant workbook, found {candidates}")
        with archive.open(candidates[0]) as workbook:
            plants = pd.read_excel(
                workbook,
                sheet_name="Plant",
                skiprows=1,
                usecols=lambda column: column in columns,
            )

    missing = set(columns) - set(plants.columns)
    if missing:
        raise ValueError(f"EIA-860 plant table is missing columns: {sorted(missing)}")
    result = plants[list(columns)].rename(columns=columns)
    result["facility_id"] = pd.to_numeric(result["facility_id"], errors="coerce").astype("Int64")
    return result.dropna(subset=["facility_id"]).drop_duplicates("facility_id")


def load_price_locations(metadata_dir: Path) -> pd.DataFrame:
    """Load the regional price locations already collected by the scraper."""
    paths = sorted((metadata_dir / "locations").glob("iso=*/locations.parquet"))
    if not paths:
        raise FileNotFoundError(f"No price-location metadata under {metadata_dir / 'locations'}")
    tables = [pq.read_table(path) for path in paths]
    frame = pa.concat_tables(tables, promote_options="default").to_pandas()
    return frame.drop_duplicates(["iso", "location_name", "location_type"])


def _county_mapping(county: object, zones: dict[str, set[str]]) -> str | None:
    normalized = str(county).strip().casefold()
    for suffix in (" county", " parish", " borough"):
        if normalized.endswith(suffix):
            normalized = normalized.removesuffix(suffix)
            break
    return next((zone for zone, counties in zones.items() if normalized in counties), None)


def _normalize_name(value: object) -> str:
    # Remove generic plant terms before comparing ISO location names
    tokens = re.findall(r"[a-z0-9]+", str(value).casefold())
    return "".join(token for token in tokens if token not in NAME_STOPWORDS)


def _best_location_name_match(row: pd.Series, location_names: set[str]) -> tuple[str, float] | None:
    # Require a clear winner to limit false plant-to-node name matches
    plant_names = {_normalize_name(row.get("facility_name")), _normalize_name(row.get("eia_plant_name"))}
    plant_names = {name for name in plant_names if len(name) >= 5}
    candidates: list[tuple[float, str]] = []
    for location_name in location_names:
        if location_name in {"SPPNORTH_HUB", "SPPSOUTH_HUB"}:
            continue
        normalized_location = _normalize_name(location_name)
        if len(normalized_location) < 5:
            continue
        score = 0.0
        for plant_name in plant_names:
            similarity = SequenceMatcher(None, plant_name, normalized_location).ratio()
            if plant_name in normalized_location:
                similarity = max(similarity, 0.9 + 0.09 * len(plant_name) / len(normalized_location))
            score = max(score, similarity)
        candidates.append((score, location_name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    best_score, best_name = candidates[0]
    second_score = candidates[1][0] if len(candidates) > 1 else 0.0
    if best_score < 0.72 or best_score - second_score < 0.04:
        return None
    return best_name, best_score


def _haversine_km(latitude: float, longitude: float, target_latitude: float, target_longitude: float) -> float:
    # Calculate great-circle distance between two coordinate pairs
    radius_km = 6371.0088
    latitude_1 = math.radians(latitude)
    latitude_2 = math.radians(target_latitude)
    latitude_delta = math.radians(target_latitude - latitude)
    longitude_delta = math.radians(target_longitude - longitude)
    haversine = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(latitude_1) * math.cos(latitude_2) * math.sin(longitude_delta / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(haversine))


def _nearest_zone(
    row: pd.Series,
    iso: str,
    location_names: set[str],
) -> tuple[str, float] | None:
    # Use approximate zone centroids only when plant coordinates are present
    latitude = pd.to_numeric(row.get("latitude"), errors="coerce")
    longitude = pd.to_numeric(row.get("longitude"), errors="coerce")
    if pd.isna(latitude) or pd.isna(longitude):
        return None
    candidates = {name: coordinates for name, coordinates in ZONE_CENTROIDS[iso].items() if name in location_names}
    if not candidates:
        return None
    distances = {
        name: _haversine_km(float(latitude), float(longitude), *coordinates) for name, coordinates in candidates.items()
    }
    location_name = min(distances, key=distances.get)
    return location_name, distances[location_name]


def _map_regional_location(row: pd.Series, locations_by_iso: dict[str, set[str]]) -> RegionalMapping:
    iso = row.get("iso")
    state = str(row.get("state") or "").upper()
    county = row.get("county")
    location_names = locations_by_iso.get(str(iso), set())

    if pd.isna(iso):
        return RegionalMapping(None, "eia860_balancing_authority", "none", True, "Plant is outside the studied ISOs")

    if iso == "ISONE":
        location = ISONE_STATE_ZONES.get(state)
        method = "isone_state"
        if state == "MA":
            location = _county_mapping(county, ISONE_MASSACHUSETTS_COUNTIES)
            method = "isone_massachusetts_county"
        if location:
            if state == "MA":
                return RegionalMapping(
                    location,
                    method,
                    "medium",
                    True,
                    "County proxy should be checked against the ISO-NE pricing-node table",
                )
            return RegionalMapping(location, method, "high", False, "State maps uniquely to this ISO-NE load zone")
        return RegionalMapping(None, method, "none", True, "Massachusetts boundary is ambiguous at county resolution")

    if iso == "NYISO":
        location = _county_mapping(county, NYISO_COUNTY_ZONES)
        if location:
            return RegionalMapping(
                location,
                "nyiso_county",
                "medium",
                True,
                "County rule should be checked against a NYISO generator roster",
            )
        return RegionalMapping(
            None, "nyiso_county", "none", True, "County does not uniquely identify a NYISO load zone"
        )

    if iso == "CAISO":
        owner = str(row.get("transmission_owner") or "").casefold()
        location = next((zone for token, zone in CAISO_OWNER_ZONES.items() if token in owner), None)
        if location in location_names:
            return RegionalMapping(
                location, "caiso_transmission_owner", "medium", True, "EIA transmission owner used as a DLAP proxy"
            )
        nearest = _nearest_zone(row, "CAISO", location_names)
        if nearest:
            location, distance = nearest
            return RegionalMapping(
                location,
                "caiso_nearest_dlap_centroid",
                "low",
                True,
                "Nearest approximate DLAP centroid",
                round(distance, 1),
            )
        return RegionalMapping(None, "caiso_nearest_dlap_centroid", "none", True, "No DLAP proxy could be assigned")

    if iso == "MISO":
        location = MISO_STATE_HUBS.get(state)
        if location:
            return RegionalMapping(
                location,
                "miso_state_hub",
                "low",
                True,
                "State hub is an analytical proxy, not a generator load-zone assignment",
            )
        return RegionalMapping(None, "miso_state_hub", "none", True, "No retained state hub is available")

    if iso == "SPP":
        name_match = _best_location_name_match(row, location_names)
        if name_match:
            location, score = name_match
            return RegionalMapping(
                location,
                "spp_plant_name_match",
                "medium",
                True,
                f"Plant and settlement-location name similarity {score:.3f}",
            )
        if state in SPP_NORTH_STATES:
            location = "SPPNORTH_HUB"
        elif state in SPP_SOUTH_STATES:
            location = "SPPSOUTH_HUB"
        else:
            location = None
        if location in location_names:
            return RegionalMapping(location, "spp_state_hub", "low", True, "State is a proxy for SPP hub membership")
        return RegionalMapping(None, "spp_state_hub", "none", True, "No collected SPP location could be assigned")

    if iso == "ERCOT":
        nearest = _nearest_zone(row, "ERCOT", location_names)
        if nearest:
            location, distance = nearest
            return RegionalMapping(
                location,
                "ercot_nearest_load_zone_centroid",
                "low",
                True,
                "Nearest approximate ERCOT load-zone centroid",
                round(distance, 1),
            )
        return RegionalMapping(
            None, "ercot_nearest_load_zone_centroid", "none", True, "No ERCOT load-zone proxy could be assigned"
        )
    if iso == "PJM":
        owner = str(row.get("transmission_owner") or "").casefold()
        location = next((zone for token, zone in PJM_OWNER_ZONES.items() if token in owner), None)
        if location in location_names:
            return RegionalMapping(
                location, "pjm_transmission_owner", "medium", True, "EIA transmission owner used as a PJM zone proxy"
            )
        return RegionalMapping(
            None, "pjm_location_unavailable", "none", True, "No collected PJM zone could be assigned"
        )
    return RegionalMapping(None, "unsupported_iso", "none", True, f"No mapper is configured for {iso}")


def apply_manual_overrides(mapping: pd.DataFrame, override_path: Path | None) -> pd.DataFrame:
    """Apply reviewed facility-level ISO node/zone assignments when supplied."""
    if override_path is None or not override_path.exists():
        return mapping
    overrides = pd.read_csv(override_path)
    required = {"facility_id", "price_location_name"}
    missing = required - set(overrides.columns)
    if missing:
        raise ValueError(f"Manual override file is missing columns: {sorted(missing)}")
    if overrides["facility_id"].duplicated().any():
        raise ValueError("Manual override file contains duplicate facility_id values")

    override_columns = [
        "price_location_name",
        "iso_node_id",
        "iso_node_name",
        "mapping_note",
    ]
    indexed = overrides.set_index("facility_id")
    for column in override_columns:
        if column in indexed:
            values = mapping["facility_id"].map(indexed[column])
            mapping[column] = values.combine_first(mapping.get(column))
    overridden = mapping["facility_id"].isin(indexed.index)
    mapping.loc[overridden, "mapping_method"] = "manual_reviewed_override"
    mapping.loc[overridden, "confidence"] = "high"
    mapping.loc[overridden, "needs_review"] = False
    return mapping


def build_mapping(
    plants: pd.DataFrame,
    eia_plants: pd.DataFrame,
    price_locations: pd.DataFrame,
    eia860_year: int,
    override_path: Path | None,
) -> pd.DataFrame:
    """Join plant metadata, assign regional locations, and validate them."""
    mapping = plants.merge(eia_plants, on="facility_id", how="left", validate="many_to_one")
    mapping["iso"] = mapping["balancing_authority_code"].map(BA_TO_ISO).astype("string")
    locations_by_iso = {
        str(iso): set(group["location_name"].dropna().astype(str))
        for iso, group in price_locations.groupby("iso", observed=True)
    }

    assignments = mapping.apply(_map_regional_location, axis=1, locations_by_iso=locations_by_iso)
    mapping["price_location_name"] = [assignment.price_location_name for assignment in assignments]
    mapping["mapping_method"] = [assignment.mapping_method for assignment in assignments]
    mapping["confidence"] = [assignment.confidence for assignment in assignments]
    mapping["needs_review"] = [assignment.needs_review for assignment in assignments]
    mapping["mapping_note"] = [assignment.note for assignment in assignments]
    mapping["mapping_distance_km"] = [assignment.distance_km for assignment in assignments]
    mapping["iso_node_id"] = pd.Series(pd.NA, index=mapping.index, dtype="string")
    mapping["iso_node_name"] = pd.Series(pd.NA, index=mapping.index, dtype="string")
    mapping = apply_manual_overrides(mapping, override_path)

    location_lookup = price_locations.drop_duplicates(["iso", "location_name"])[
        ["iso", "location_id", "location_name", "location_type"]
    ].rename(columns={"location_name": "price_location_name", "location_id": "price_location_id"})
    mapping = mapping.merge(
        location_lookup,
        on=["iso", "price_location_name"],
        how="left",
        validate="many_to_one",
    )
    missing_location = mapping["price_location_name"].notna() & mapping["location_type"].isna()
    mapping.loc[missing_location, "mapping_note"] = "Assigned location is absent from collected price metadata"
    mapping.loc[missing_location, "confidence"] = "none"
    mapping.loc[missing_location, "needs_review"] = True

    mapping["eia860_year"] = eia860_year
    mapping["eia860_source_url"] = EIA860_URL.format(year=eia860_year)
    mapping["created_at_utc"] = datetime.now(timezone.utc)
    mapping["has_price_location"] = mapping["location_type"].notna() & mapping["price_location_name"].notna()
    mapping["is_analysis_ready"] = mapping["has_price_location"] & mapping["confidence"].isin(["high", "medium"])
    columns = [
        "facility_id",
        "facility_name",
        "fuel_type",
        "state",
        "county",
        "latitude",
        "longitude",
        "nerc_region",
        "eia_plant_name",
        "balancing_authority_code",
        "balancing_authority_name",
        "transmission_owner",
        "transmission_owner_id",
        "iso",
        "iso_node_id",
        "iso_node_name",
        "price_location_id",
        "price_location_name",
        "location_type",
        "mapping_method",
        "mapping_distance_km",
        "confidence",
        "needs_review",
        "has_price_location",
        "is_analysis_ready",
        "mapping_note",
        "eia860_year",
        "eia860_source_url",
        "created_at_utc",
    ]
    return mapping[columns].sort_values(["iso", "facility_id", "fuel_type"], na_position="last").reset_index(drop=True)


def write_mapping(mapping: pd.DataFrame, output_path: Path) -> None:
    """Atomically write the plant-zone mapping as Zstd-compressed Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent) as temporary_dir:
        temporary = Path(temporary_dir)
        parquet_path = temporary / output_path.name
        mapping.to_parquet(parquet_path, index=False, compression="zstd")
        parquet_path.replace(output_path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(FULL_DATA_PARQUET))
    parser.add_argument("--output", type=Path, default=Path(PLANT_ZONE_MAPPING))
    parser.add_argument("--metadata-dir", type=Path, default=Path(POWER_PRICE_METADATA_DIR))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eia860-year", type=int, default=DEFAULT_EIA860_YEAR)
    parser.add_argument("--eia860-zip", type=Path)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--overrides", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    """Build and save the plant-node-zone mapping."""
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    reference_dir = args.output.parent / "reference"
    eia860_zip = args.eia860_zip or reference_dir / f"eia860{args.eia860_year}.zip"
    override_path = args.overrides or args.output.parent / "manual_node_zone_overrides.csv"

    print(f"Scanning {args.input} in batches of {args.batch_size:,} rows")
    plants = collect_unique_fossil_plants(args.input, args.batch_size)
    print(f"Found {len(plants):,} unique coal or natural-gas plants")
    eia_plants = load_eia860_plants(eia860_zip, args.eia860_year, allow_download=not args.no_download)
    price_locations = load_price_locations(args.metadata_dir)
    mapping = build_mapping(plants, eia_plants, price_locations, args.eia860_year, override_path)
    write_mapping(mapping, args.output)

    assigned = int(mapping["has_price_location"].sum())
    analysis_ready = int(mapping["is_analysis_ready"].sum())
    print(f"Wrote {len(mapping):,} rows to {args.output}")
    print(f"Candidate regional-price assignments: {assigned:,}/{len(mapping):,}")
    print(f"Reviewed/high-confidence assignments: {analysis_ready:,}/{len(mapping):,}")
    print(f"Rows requiring review: {int(mapping['needs_review'].sum()):,}")


if __name__ == "__main__":
    main()
