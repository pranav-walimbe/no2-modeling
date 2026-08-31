"""GridStatus adapters for regional day-ahead and real-time power prices."""

import io
import os
import zipfile
from abc import ABC, abstractmethod
from typing import BinaryIO, Literal
from urllib.parse import urlencode

import pandas as pd
import requests
from gridstatus import CAISO, ISONE, MISO, NYISO, PJM, Ercot
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
MISO_API_CHUNK_DURATION = pd.Timedelta(days=1)
CAISO_DLAP_LOCATIONS = [
    "DLAP_PGAE-APND",
    "DLAP_SCE-APND",
    "DLAP_SDGE-APND",
    "DLAP_VEA-APND",
]
SPP_DOWNLOAD_URL = "https://portal.spp.org/file-browser-api/download"
SPP_ENDPOINTS = {
    "day_ahead": "da-lmp-by-settlement-location",
    "real_time": "rtbm-lmp-by-location",
}
SPP_MONTHLY_PREFIXES = {"day_ahead": "DA-LMP-MONTHLY-SL", "real_time": "RTBM-LMP-MONTHLY-SL"}
SPP_DAILY_PREFIXES = {"day_ahead": "DA-LMP-SL", "real_time": "RTBM-LMP-DAILY-SL"}
SPP_PRICE_COLUMNS = {"LMP": "LMP", "MEC": "Energy", "MCC": "Congestion", "MLC": "Loss"}


class _HTTPRangeReader(io.RawIOBase):
    """Expose a remote ZIP as seekable without downloading it in full."""

    def __init__(self, session: requests.Session, url: str) -> None:
        self.session = session
        self.url = url
        response = session.head(url, timeout=(10, 60))
        response.raise_for_status()
        if "bytes" not in response.headers.get("Accept-Ranges", "").lower():
            raise RuntimeError("SPP archive does not support bounded byte-range requests")
        self.length = int(response.headers["Content-Length"])
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_CUR:
            offset += self.position
        elif whence == io.SEEK_END:
            offset += self.length
        if not 0 <= offset <= self.length:
            raise ValueError("seek outside SPP archive")
        self.position = offset
        return offset

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.length or size == 0:
            return b""
        end = self.length - 1 if size < 0 else min(self.position + size - 1, self.length - 1)
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={self.position}-{end}"},
            stream=True,
            timeout=(10, 120),
        )
        try:
            if response.status_code != 206:
                raise RuntimeError(f"SPP ignored bounded range request (HTTP {response.status_code})")
            payload = response.content
        finally:
            response.close()
        expected = end - self.position + 1
        if len(payload) != expected:
            raise OSError(f"SPP returned {len(payload)} bytes for a {expected}-byte range")
        self.position += len(payload)
        return payload


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


def _parse_spp_monthly_report(stream: BinaryIO) -> pd.DataFrame:
    # Monthly reports store one row per location and price component
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(stream, chunksize=100_000, skipinitialspace=True):
        chunk.columns = chunk.columns.str.strip()
        frames.append(chunk)
    data = _concat_frames(frames)
    if data.empty:
        return data

    hour_columns = [column for column in data if column.upper().startswith("HE")]
    data = data.melt(
        id_vars=["Date", "Settlement Location Name", "Price Type"],
        value_vars=hour_columns,
        var_name="Hour Ending",
        value_name="Price",
    ).dropna(subset=["Price"])
    hour = data["Hour Ending"].str.extract(r"(\d+)", expand=False).astype(int)
    interval_start = pd.to_datetime(data["Date"]) + pd.to_timedelta(hour - 1, unit="h")
    repeated = data["Hour Ending"].str.upper().str.contains("X") | data["Hour Ending"].str.endswith(".1")
    data["Interval Start"] = interval_start.dt.tz_localize(
        "US/Central",
        ambiguous=~repeated,
        nonexistent="NaT",
    ).dt.tz_convert("UTC")
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")
    data = data.loc[data["Price Type"].isin(SPP_PRICE_COLUMNS)].dropna(subset=["Interval Start", "Price"])
    result = data.pivot_table(
        index=["Interval Start", "Settlement Location Name"],
        columns="Price Type",
        values="Price",
        aggfunc="first",
    ).reset_index()
    result = result.rename(columns={"Settlement Location Name": "Location", **SPP_PRICE_COLUMNS})
    for column in ("LMP", "Congestion", "Loss"):
        if column not in result:
            result[column] = pd.NA
    derived_energy = result["LMP"] - result["Congestion"] - result["Loss"]
    if "Energy" in result:
        result["Energy"] = result["Energy"].fillna(derived_energy)
    else:
        result["Energy"] = derived_energy
    result["Interval End"] = result["Interval Start"] + pd.Timedelta(hours=1)
    result["Location Type"] = "Settlement Location"
    columns = ["Interval Start", "Interval End", "Location", "Location Type", *SPP_PRICE_COLUMNS.values()]
    return result[columns]


def _parse_spp_daily_report(stream: BinaryIO, market: MarketName) -> pd.DataFrame:
    # Daily data repair holes in monthly reports, including repeated DST hours
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(stream, chunksize=100_000, skipinitialspace=True):
        chunk.columns = chunk.columns.str.strip()
        chunk = chunk.rename(
            columns={
                "Settlement Location Name": "Settlement Location",
                "GMT Interval": "GMTIntervalEnd",
            },
        )
        frames.append(chunk)
    data = _concat_frames(frames)
    if data.empty:
        return data
    data["Interval End"] = pd.to_datetime(data["GMTIntervalEnd"], utc=True)
    data["Interval Start"] = data["Interval End"] - pd.Timedelta(
        minutes=60 if market == "day_ahead" else 5,
    )
    data = data.rename(
        columns={"Settlement Location": "Location", "MEC": "Energy", "MCC": "Congestion", "MLC": "Loss"},
    )
    data["Location Type"] = "Settlement Location"
    columns = ["Interval Start", "Interval End", "Location", "Location Type", "LMP", "Energy", "Congestion", "Loss"]
    return data[columns]


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
        gridstatus_market = Markets.DAY_AHEAD_HOURLY if market == "day_ahead" else Markets.REAL_TIME_HOURLY_FINAL
        return self.legacy_client.get_lmp(date=start, end=end, market=gridstatus_market, locations="ALL")

    def _fetch_current(self, market: MarketName, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        # MISO Data Exchange superseded the legacy pricing reports in December 2025
        client = self._current_client()
        frames: list[pd.DataFrame] = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + MISO_API_CHUNK_DURATION, end)
            if market == "day_ahead":
                frame = client.get_lmp_day_ahead_hourly_ex_post(date=chunk_start, end=chunk_end)
            else:
                frame = client.get_lmp_real_time_hourly_ex_post_final(date=chunk_start, end=chunk_end)
            frames.append(_filter_location_types(frame, REGIONAL_LOCATION_TYPES))
            chunk_start = chunk_end
        return _concat_frames(frames)

    def fetch(self, market: MarketName, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Fetch final hourly MISO hub and load-zone LMPs."""
        frames: list[pd.DataFrame] = []
        if start < MISO_API_CUTOVER:
            frames.append(self._fetch_legacy(market, start, min(end, MISO_API_CUTOVER)))
        if end > MISO_API_CUTOVER:
            frames.append(self._fetch_current(market, max(start, MISO_API_CUTOVER), end))
        return _filter_location_types(_concat_frames(frames), REGIONAL_LOCATION_TYPES)


class SPPSource(PowerPriceSource):
    """SPP prices at every published settlement location."""

    iso = "SPP"
    timezone = "US/Central"

    def __init__(self) -> None:
        self.session = requests.Session()
        self.month_cache: dict[tuple[MarketName, int, int], pd.DataFrame] = {}
        self.day_cache: dict[tuple[MarketName, str], pd.DataFrame] = {}
        self.archive_cache: dict[tuple[MarketName, int], zipfile.ZipFile] = {}

    def _url(self, market: MarketName, path: str) -> str:
        # Keep SPP paths URL-encoded, including the leading slash
        endpoint = SPP_ENDPOINTS[market]
        return f"{SPP_DOWNLOAD_URL}/{endpoint}?{urlencode({'path': path})}"

    def _load_annual_member(self, market: MarketName, year: int, member: str) -> bytes:
        # ZipFile seeks through bounded ranges and reads only the selected member
        key = (market, year)
        if key not in self.archive_cache:
            reader = _HTTPRangeReader(self.session, self._url(market, f"/{year}/{year}.zip"))
            self.archive_cache[key] = zipfile.ZipFile(reader)
        return self.archive_cache[key].read(member)

    def _load_monthly_report(self, market: MarketName, year: int, month: int) -> pd.DataFrame:
        # SPP retires older loose CSVs into a yearly archive
        key = (market, year, month)
        if key in self.month_cache:
            return self.month_cache[key]
        filename = f"{SPP_MONTHLY_PREFIXES[market]}-{year}{month:02d}.csv"
        path = f"/{year}/{month:02d}/{filename}"
        response = self.session.get(self._url(market, path), stream=True, timeout=(10, 120))
        try:
            if response.status_code == 404:
                member = f"{year}/{month:02d}/{filename}"
                frame = _parse_spp_monthly_report(io.BytesIO(self._load_annual_member(market, year, member)))
            else:
                response.raise_for_status()
                response.raw.decode_content = True
                frame = _parse_spp_monthly_report(response.raw)
        finally:
            response.close()
        self.month_cache[key] = frame
        return frame

    def _load_daily_report(self, market: MarketName, day: pd.Timestamp) -> pd.DataFrame:
        # Fetch one day only when the monthly report is incomplete
        day_key = day.strftime("%Y%m%d")
        key = (market, day_key)
        if key in self.day_cache:
            return self.day_cache[key]
        suffix = f"{day_key}0100" if market == "day_ahead" else day_key
        filename = f"{SPP_DAILY_PREFIXES[market]}-{suffix}.csv"
        path = f"/{day.year}/{day.month:02d}/By_Day/{filename}"
        response = self.session.get(self._url(market, path), stream=True, timeout=(10, 120))
        try:
            if response.status_code == 404:
                member = f"{day.year}/{day.month:02d}/By_Day/{filename}"
                frame = _parse_spp_daily_report(io.BytesIO(self._load_annual_member(market, day.year, member)), market)
            else:
                response.raise_for_status()
                response.raw.decode_content = True
                frame = _parse_spp_daily_report(response.raw, market)
        finally:
            response.close()
        self.day_cache[key] = frame
        return frame

    def fetch(self, market: MarketName, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Fetch SPP settlement locations, repairing only missing hours."""
        frames: list[pd.DataFrame] = []
        month = start.normalize().replace(day=1)
        while month < end:
            frames.append(self._load_monthly_report(market, month.year, month.month))
            month += pd.DateOffset(months=1)
        frame = _concat_frames(frames)
        lower, upper = start.tz_convert("UTC"), end.tz_convert("UTC")
        frame = frame.loc[(frame["Interval Start"] >= lower) & (frame["Interval Start"] < upper)].copy()

        expected_hours = pd.date_range(lower, upper, freq="h", inclusive="left")
        locations = frame["Location"].drop_duplicates().sort_values()
        expected = pd.MultiIndex.from_product([locations, expected_hours], names=["Location", "Hour"])
        observed = pd.MultiIndex.from_arrays(
            [frame["Location"], frame["Interval Start"].dt.floor("h")],
            names=["Location", "Hour"],
        )
        missing = expected.difference(observed)
        if len(missing):
            missing_table = missing.to_frame(index=False)
            local_days = missing_table["Hour"].dt.tz_convert(self.timezone).dt.normalize().unique()
            daily = _concat_frames(
                [self._load_daily_report(market, pd.Timestamp(day)) for day in local_days],
            )
            daily["Hour"] = daily["Interval Start"].dt.floor("h")
            repair = daily.merge(missing_table, on=["Location", "Hour"], how="inner").drop(columns="Hour")
            frame = pd.concat([frame, repair], ignore_index=True)
        return frame.sort_values(["Location", "Interval Start"]).reset_index(drop=True)


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
                populated_sheets = [
                    sheet for sheet in sheets.values() if not sheet.empty and not sheet.isna().all().all()
                ]
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

DEFAULT_ISOS: tuple[str, ...] = tuple(iso for iso in SOURCE_FACTORIES if iso != "PJM")


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
