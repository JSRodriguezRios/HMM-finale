"""Timezone helpers for normalizing timestamps to UTC."""

from __future__ import annotations

import datetime as dt

import pandas as pd


def ensure_utc(value: dt.datetime) -> dt.datetime:
    """Return ``value`` as an aware UTC datetime."""

    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def ensure_utc_index(index: pd.Index | pd.Series) -> pd.DatetimeIndex:
    """Convert timestamps to a UTC-aware :class:`~pandas.DatetimeIndex`."""

    if isinstance(index, pd.Series):
        series = pd.to_datetime(index)
        if series.dt.tz is None:
            series = series.dt.tz_localize("UTC")
        else:
            series = series.dt.tz_convert("UTC")
        return pd.DatetimeIndex(series)

    if not isinstance(index, pd.DatetimeIndex):
        index = pd.to_datetime(index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    return index


def hourly_range(start: dt.datetime, end: dt.datetime) -> pd.DatetimeIndex:
    """Return a UTC hourly index covering ``[start, end)``."""

    start_utc = ensure_utc(start)
    end_utc = ensure_utc(end)
    if end_utc <= start_utc:
        raise ValueError("end must be later than start for hourly_range")
    # ``inclusive='left'`` ensures the range stops just before ``end``.
    return pd.date_range(start=start_utc, end=end_utc, freq="h", inclusive="left", tz="UTC")


def nearest_hour_floor(timestamp: dt.datetime) -> dt.datetime:
    """Floor ``timestamp`` down to the nearest UTC hour."""

    ts = ensure_utc(timestamp)
    return ts.replace(minute=0, second=0, microsecond=0)


__all__ = ["ensure_utc", "ensure_utc_index", "hourly_range", "nearest_hour_floor"]
