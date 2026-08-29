"""Command-specific prerequisite validation."""

from pathlib import Path

from no2_modeling.config import (
    CAMPD_API_KEY,
    EARTHDATA_PASSWORD,
    EARTHDATA_USERNAME,
)


def require_campd_credentials() -> str:
    """Return the CAMPD API key or raise a useful configuration error."""
    if not CAMPD_API_KEY:
        raise RuntimeError("CAMPD_API_KEY is not set; add it to the repository .env file")
    return CAMPD_API_KEY


def require_earthdata_credentials() -> None:
    """Validate credentials used by earthaccess's environment strategy."""
    if not EARTHDATA_USERNAME or not EARTHDATA_PASSWORD:
        raise RuntimeError("EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set in the repository .env file")


def require_cds_credentials(path: Path | None = None) -> Path:
    """Return the CDS configuration path or raise a useful error."""
    config_path = path or Path.home() / ".cdsapirc"
    if not config_path.is_file():
        raise RuntimeError(f"{config_path} not found; configure CDS API credentials first")
    return config_path
