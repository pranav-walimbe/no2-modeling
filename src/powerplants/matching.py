"""Deterministic matching helpers for pipeline records and satellite tiles."""

from collections.abc import Iterable
from datetime import datetime, timedelta


def select_tempo_after(
    target_dt: datetime,
    candidates: Iterable[tuple[datetime, str]],
    window_minutes: int,
) -> str | None:
    """Return the closest candidate after a target within the given window."""
    window = timedelta(minutes=window_minutes)
    eligible = [
        (curr_dt - target_dt, fname) for curr_dt, fname in candidates if timedelta(0) < curr_dt - target_dt <= window
    ]
    return min(eligible, default=(None, None), key=lambda item: item[0])[1]
