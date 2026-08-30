"""GridStatus adapters for regional day-ahead and real-time power prices."""

import os
from abc import ABC, abstractmethod
from typing import Literal

import pandas as pd
from gridstatus import CAISO, ISONE, MISO, NYISO, PJM, SPP, Ercot
from gridstatus import utils as gridstatus_utils
from gridstatus.base import Markets
from gridstatus.ercot import HISTORICAL_RTM_LOAD_ZONE_AND_HUB_PRICES_RTID
from gridstatus.isone_api.isone_api import ISONEAPI
from gridstatus.miso_api import MISOAPI
from gridstatus.nyiso import NYISOLocationType

MarketName = Literal["day_ahead", "real_time"]

HUB_LOCATION_TYPES = {"hub", "trading hub"}
ZONE_LOCATION_TYPES = {"zone", "load zone", "loadzone", "dlap"}
REGIONAL_LOCATION_TYPES = HUB_LOCATION_TYPES | ZONE_LOCATION_TYPES
MISO_API_CUTOVER = pd.Timestamp("2025-12-12", tz="EST")
CAISO_DLAP_LOCATIONS = [
    "DLAP_PGAE-APND",
    "DLAP_SCE-APND",
    "DLAP_SDGE-APND",
    "DLAP_VEA-APND",
]


def _require_environment(name: str) -> str:
    # Validate credentials only for the source that needs them
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required for this ISO; add it to the repository .env file")
    return value


def _filter_location_types(frame: pd.DataFrame, accepted: set[str]) -> pd.DataFrame:
    # Keep stable regional aggregates instead of generator and bus nodes
    if "Location Type" not in frame.columns:
        raise ValueError("GridStatus response is missing Location Type")
    location_types = frame["Location Type"].astype("string").str.strip().str.casefold()
    return frame.loc[location_types.isin(accepted)].copy()


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    # Preserve an empty response when a split date range has no applicable segment
    populated = [frame for frame in frames if not frame.empty]
    return pd.concat(populated, ignore_index=True) if populated else pd.DataFrame()


def _parse_ercot_rtm_intervals(frame: pd.DataFrame, timezone: str) -> pd.DataFrame:
    # Work around a GridStatus 0.36 conversion that is incompatible with current pandas
    data = frame.rename(
        columns={
            "Delivery Date": "DeliveryDate",
            "Delivery Hour": "HourEnding",
            "Delivery Interval": "DeliveryInterval",
            "Repeated Hour Flag": "DSTFlag",
            "Settlement Point Type": "SettlementPointType",
        },
    ).copy()
    data = data.dropna(subset=["HourEnding", "DeliveryInterval"], how="all")
    hour_offset = pd.to_timedelta(pd.to_numeric(data["HourEnding"]) - 1, unit="h")
    interval_offset = pd.to_timedelta((pd.to_numeric(data["DeliveryInterval"]) - 1) * 15, unit="m")
    interval_start = pd.to_datetime(data["DeliveryDate"]) + hour_offset + interval_offset
    ambiguous: str | pd.Series = "infer"
    if "DSTFlag" in data:
        flags = data["DSTFlag"]
        ambiguous = ~flags if flags.dtype == bool else flags.astype("string").eq("N")
    data["Interval Start"] = interval_start.dt.tz_localize(timezone, ambiguous=ambiguous)
    data["Interval End"] = data["Interval Start"] + pd.Timedelta(minutes=15)
    data["Time"] = data["Interval Start"]
    return data


class PowerPriceSource(ABC):
    """Interface implemented by each ISO-specific GridStatus adapter."""

    iso: str
    timezone: str

    @abstractmethod
    def fetch(self, market: MarketName, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Fetch regional prices over a local-time half-open interval.

        Args:
            market: Normalized day-ahead or real-time market name.
            start: Inclusive query timestamp in the ISO timezone.
            end: Exclusive query timestamp in the ISO timezone.

        Returns:
            GridStatus price records at their source resolution.
        """


class PJMSource(PowerPriceSource):
    """PJM hourly hub and load-zone prices."""

    iso = "PJM"
    timezone = "US/Eastern"

    def __init__(self) -> None:
        self.client = PJM(api_key=_require_environment("PJM_API_KEY"))

    def fetch(self, market: MarketName, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Fetch verified hourly PJM hub and zone LMPs."""
        gridstatus_market = Markets.DAY_AHEAD_HOURLY if market == "day_ahead" else Markets.REAL_TIME_HOURLY
        frames = [
            self.client.get_lmp(
                date=start,
                end=end,
                market=gridstatus_market,
                locations=None,
                location_type=location_type,
            )
            for location_type in ("HUB", "ZONE")
        ]
        return _filter_location_types(_concat_frames(frames), REGIONAL_LOCATION_TYPES)


class MISOSource(PowerPriceSource):
    """MISO final hourly hub and load-zone prices across its API migration."""

    iso = "MISO"
    timezone = "EST"

    def __init__(self) -> None:
        self.legacy_client = MISO()
        self.api_client: MISOAPI | None = None

    def _current_client(self) -> MISOAPI:
        # Construct lazily so older public report history does not require a key
        if self.api_client is None:
            key = _require_environment("MISO_API_PRICING_SUBSCRIPTION_KEY")
            self.api_client = MISOAPI(pricing_api_key=key, initial_sleep_seconds=2, max_retries=5)
        return self.api_client

    def _fetch_legacy(self, market: MarketName, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        # MISO's legacy reports expose final RT and ex-post DA hourly files
        gridstatus_market = (
            Markets.DAY_AHEAD_HOURLY if market == "day_ahead" else Markets.REAL_TIME_HOURLY_FINAL
        )
        return self.legacy_client.get_lmp(date=start, end=end, market=gridstatus_market, locations="ALL")

    def _fetch_current(self, market: MarketName, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        # MISO Data Exchange superseded the legacy pricing reports in December 2025
        client = self._current_client()
        if market == "day_ahead":
            return client.get_lmp_day_ahead_hourly_ex_post(date=start, end=end)
        return client.get_lmp_real_time_hourly_ex_post_final(date=start, end=end)

    def fetch(self, market: MarketName, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Fetch final hourly MISO hub and load-zone LMPs."""
        frames: list[pd.DataFrame] = []
        if start < MISO_API_CUTOVER:
            frames.append(self._fetch_legacy(market, start, min(end, MISO_API_CUTOVER)))
        if end > MISO_API_CUTOVER:
            frames.append(self._fetch_current(market, max(start, MISO_API_CUTOVER), end))
        return _filter_location_types(_concat_frames(frames), REGIONAL_LOCATION_TYPES)


class SPPSource(PowerPriceSource):
    """SPP hourly hub prices."""

    iso = "SPP"
    timezone = "US/Central"

    def __init__(self) -> None:
        self.client = SPP()

    def fetch(self, market: MarketName, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Fetch SPP hub prices at hourly DA or five-minute RT resolution."""
        if market == "day_ahead":
            return self.client.get_lmp_day_ahead_hourly(
                date=start,
                end=end,
                location_type="Hub",
            )
        return self.client.get_lmp_real_time_5_min_by_location(
            date=start,
            end=end,
            location_type="Hub",
            use_daily_files=True,
        )


class ERCOTSource(PowerPriceSource):
    """ERCOT settlement prices for trading hubs and load zones."""

    iso = "ERCOT"
    timezone = "US/Central"

    def __init__(self) -> None:
        self.client = Ercot()
        self.year_cache: dict[tuple[MarketName, int], pd.DataFrame] = {}

    def _fetch_completed_year(self, market: MarketName, year: int) -> pd.DataFrame:
        # Annual and year-to-date archives already exclude resource nodes
        cache_key = (market, year)
        if cache_key not in self.year_cache:
            if market == "day_ahead":
                self.year_cache[cache_key] = self.client.get_dam_spp(year)
            else:
                doc_info = self.client._get_document(
                    report_type_id=HISTORICAL_RTM_LOAD_ZONE_AND_HUB_PRICES_RTID,
                    constructed_name_contains=f"{year}.zip",
                )
                workbook = gridstatus_utils.get_zip_file(doc_info.url)
                sheets = pd.read_excel(workbook, sheet_name=None)
                populated_sheets = [sheet for sheet in sheets.values() if not sheet.empty and not sheet.isna().all().all()]
                parsed = _parse_ercot_rtm_intervals(pd.concat(populated_sheets), self.timezone)
                self.year_cache[cache_key] = self.client._finalize_spp_df(
                    parsed,
                    market=Markets.REAL_TIME_15_MIN,
                )
        return self.year_cache[cache_key]

    def fetch(self, market: MarketName, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Fetch ERCOT hub and load-zone settlement prices."""
        frames: list[pd.DataFrame] = []
        final_year = (end - pd.Timedelta(microseconds=1)).year
        for year in range(start.year, final_year + 1):
            year_start = pd.Timestamp(year=year, month=1, day=1, tz=self.timezone)
            year_end = year_start + pd.DateOffset(years=1)
            query_start = max(start, year_start)
            query_end = min(end, year_end)
            frame = self._fetch_completed_year(market, year)
            interval_start = pd.to_datetime(frame["Interval Start"], utc=True)
            lower = query_start.tz_convert("UTC")
            upper = query_end.tz_convert("UTC")
            frames.append(frame.loc[(interval_start >= lower) & (interval_start < upper)].copy())
        return _filter_location_types(_concat_frames(frames), REGIONAL_LOCATION_TYPES)


class CAISOSource(PowerPriceSource):
    """CAISO trading-hub and default-load-aggregation-point prices."""

    iso = "CAISO"
    timezone = "US/Pacific"

    def __init__(self) -> None:
        self.client = CAISO()

    def fetch(self, market: MarketName, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Fetch CAISO hub and DLAP prices at hourly DA or 15-minute RT resolution."""
        gridstatus_market = Markets.DAY_AHEAD_HOURLY if market == "day_ahead" else Markets.REAL_TIME_15_MIN
        locations = sorted(set(self.client.trading_hub_locations) | set(CAISO_DLAP_LOCATIONS))
        frame = self.client.get_lmp(
            date=start,
            end=end,
            market=gridstatus_market,
            locations=locations,
            sleep=5,
        )
        return _filter_location_types(frame, REGIONAL_LOCATION_TYPES)


class NYISOSource(PowerPriceSource):
    """NYISO hourly load-zone prices."""

    iso = "NYISO"
    timezone = "US/Eastern"

    def __init__(self) -> None:
        self.client = NYISO()

    def fetch(self, market: MarketName, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Fetch hourly NYISO zonal LBMPs."""
        gridstatus_market = Markets.DAY_AHEAD_HOURLY if market == "day_ahead" else Markets.REAL_TIME_HOURLY
        return self.client.get_lmp(
            date=start,
            end=end,
            market=gridstatus_market,
            locations=None,
            location_type=NYISOLocationType.ZONE,
        )


class ISONESource(PowerPriceSource):
    """ISO-NE day-ahead and final real-time core hub and zone prices."""

    iso = "ISONE"
    timezone = "US/Eastern"

    def __init__(self) -> None:
        self.public_client = ISONE()
        self.final_client: ISONEAPI | None = None

    def _final_client(self) -> ISONEAPI:
        # The final hourly endpoint requires ISO-NE Web Services credentials
        _require_environment("ISONE_API_USERNAME")
        _require_environment("ISONE_API_PASSWORD")
        if self.final_client is None:
            self.final_client = ISONEAPI(sleep_seconds=2, max_retries=5)
        return self.final_client

    def fetch(self, market: MarketName, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Fetch ISO-NE core hub and zone prices."""
        if market == "day_ahead":
            locations = list(self.public_client.hubs) + list(self.public_client.zones)
            frame = self.public_client.get_lmp(
                date=start,
                end=end,
                market=Markets.DAY_AHEAD_HOURLY,
                locations=locations,
                include_id=True,
            )
        else:
            frame = self._final_client().get_lmp_real_time_hourly_final(date=start, end=end)
        return _filter_location_types(frame, REGIONAL_LOCATION_TYPES)


SOURCE_FACTORIES: dict[str, type[PowerPriceSource]] = {
    "CAISO": CAISOSource,
    "ERCOT": ERCOTSource,
    "ISONE": ISONESource,
    "MISO": MISOSource,
    "NYISO": NYISOSource,
    "PJM": PJMSource,
    "SPP": SPPSource,
}


def create_source(iso: str) -> PowerPriceSource:
    """Construct the configured source adapter for an ISO.

    Args:
        iso: Normalized ISO identifier.

    Returns:
        Source adapter for the requested ISO.
    """
    try:
        return SOURCE_FACTORIES[iso.upper()]()
    except KeyError as error:
        supported = ", ".join(SOURCE_FACTORIES)
        raise ValueError(f"Unsupported ISO {iso!r}; choose one of: {supported}") from error
