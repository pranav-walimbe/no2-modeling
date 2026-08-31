"""Tests for CAMPD hourly emissions collection."""

import unittest
from datetime import date
from unittest.mock import Mock, patch

from collection.scrape_emissions import fetch_chunk, latest_completed_quarter_end


class LatestCompletedQuarterEndTest(unittest.TestCase):
    """Verify CAMPD reporting-period date bounds."""

    def test_returns_previous_quarter_end(self) -> None:
        self.assertEqual(latest_completed_quarter_end(date(2026, 8, 30)), date(2026, 6, 30))
        self.assertEqual(latest_completed_quarter_end(date(2026, 1, 1)), date(2025, 12, 31))


class FetchChunkTest(unittest.TestCase):
    """Verify permanent API errors do not trigger retries."""

    @patch("collection.scrape_emissions.require_campd_credentials", return_value="secret-key")
    @patch("collection.scrape_emissions.requests.get")
    def test_does_not_retry_bad_request_or_expose_api_key(self, get: Mock, _credentials: Mock) -> None:
        response = Mock(status_code=400)
        response.json.return_value = {"message": ["End date exceeds the latest reporting quarter"]}
        get.return_value = response

        with self.assertRaisesRegex(RuntimeError, "HTTP 400") as raised:
            fetch_chunk("AL", "2026-07-01", "2026-07-31")

        self.assertEqual(get.call_count, 1)
        self.assertNotIn("secret-key", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
