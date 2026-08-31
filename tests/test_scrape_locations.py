"""Tests for CAMPD facility and unit attribute augmentation."""

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests

from collection.scrape_locations import _build_attribute_frames, _fetch_attribute_page, write_augmented_parquet


class AttributeFramesTest(unittest.TestCase):
    """Verify facility and unit features remain available for hourly joins."""

    def test_preserves_rich_facility_and_unit_attributes(self) -> None:
        records = [
            {
                "facilityId": 3,
                "unitId": " 1 ",
                "year": 2026,
                "latitude": 31.0,
                "longitude": -88.0,
                "epaRegion": 4,
                "county": "Mobile",
                "nercRegion": "SERC",
                "ownerOperator": "Example Utility (Owner)",
                "operatingStatus": "OP",
                "commercialOperationDate": "1954-02-12",
                "associatedGeneratorsAndNameplateCapacity": "1 (153)",
                "primaryFuelInfo": "Coal",
                "unitType": "Boiler",
                "noxControlInfo": "Low NOx Burner",
            },
        ]

        facility_attributes, unit_attributes = _build_attribute_frames(records)

        self.assertEqual(facility_attributes.loc[0, "facilityAttributeYear"], 2026)
        self.assertEqual(facility_attributes.loc[0, "county"], "Mobile")
        self.assertEqual(facility_attributes.loc[0, "ownerOperator"], "Example Utility (Owner)")
        self.assertEqual(unit_attributes.loc[0, "unitIdKey"], "1")
        self.assertEqual(unit_attributes.loc[0, "unitOperatingStatus"], "OP")
        self.assertEqual(unit_attributes.loc[0, "generatorAndNameplateCapacity"], "1 (153)")
        self.assertEqual(unit_attributes.loc[0, "attributePrimaryFuelInfo"], "Coal")
        self.assertEqual(unit_attributes.loc[0, "noxControlInfo"], "Low NOx Burner")

    def test_streams_augmented_rows_to_compressed_parquet(self) -> None:
        records = [
            {
                "facilityId": 3,
                "unitId": "1",
                "year": 2026,
                "latitude": 31.0,
                "longitude": -88.0,
                "epaRegion": 4,
                "county": "Mobile",
                "operatingStatus": "OP",
                "primaryFuelInfo": "Coal",
            },
        ]
        facility_attributes, unit_attributes = _build_attribute_frames(records)
        emissions = pd.DataFrame(
            {
                "stateCode": ["AL", "AL"],
                "facilityName": ["Barry", "Unknown"],
                "facilityId": [3, 999],
                "unitId": ["1", "1"],
                "date": ["2024-01-01", "2024-01-01"],
                "hour": [0, 0],
                "opTime": [1.0, 0.0],
                "noxMass": [2.5, 0.0],
                "noxRate": [0.1, 0.0],
                "grossLoad": [25.0, 0.0],
                "primaryFuelInfo": ["Coal", "Coal"],
                "unitType": ["Boiler", "Boiler"],
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "emissions.csv"
            output_path = root / "nox_emissions_full.parquet"
            emissions.to_csv(input_path, index=False)

            row_count = write_augmented_parquet(
                input_path,
                output_path,
                facility_attributes,
                unit_attributes,
                chunk_size=1,
            )

            result = pd.read_parquet(output_path)
            self.assertEqual(row_count, 1)
            self.assertEqual(result.loc[0, "facilityId"], 3)
            self.assertEqual(result.loc[0, "county"], "Mobile")
            self.assertEqual(list(root.glob("*.part")), [])

    @patch("collection.scrape_locations.time.sleep")
    @patch("collection.scrape_locations.requests.get")
    def test_request_failure_does_not_log_api_key(self, mock_get, _mock_sleep) -> None:
        secret = "secret-campd-key"
        mock_get.side_effect = requests.ConnectionError(f"failed URL?api_key={secret}")

        with patch("collection.scrape_locations.require_campd_credentials", return_value=secret):
            output = StringIO()
            with redirect_stdout(output):
                result = _fetch_attribute_page(facility_id=3, year=2026, page=1)

        self.assertIsNone(result)
        self.assertNotIn(secret, output.getvalue())


if __name__ == "__main__":
    unittest.main()
