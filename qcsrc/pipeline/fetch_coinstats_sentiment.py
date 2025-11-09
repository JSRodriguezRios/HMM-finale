"""Fetch CoinStats sentiment scores and align to hourly cadence."""

from __future__ import annotations

import datetime as dt
from datetime import timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from qcsrc.io import ensure_directory, get_data_path, load_assets_config
from qcsrc.io.schemas import CoinStatsResponse
from qcsrc.util.logging_utils import get_logger
from qcsrc.util.secrets import CredentialSet, MissingCredentialError, load_credentials

_LOGGER = get_logger(__name__)
_BASE_URL = "https://api.coinstats.app/public/v1/sentiment"


def _ensure_utc(timestamp: dt.datetime) -> dt.datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _load_credentials() -> CredentialSet:
    try:
        return load_credentials()
    except MissingCredentialError as exc:
        raise RuntimeError("CoinStats credentials are required for fetches") from exc


def _build_output_path(symbol: str, date: dt.date, output_dir: Optional[Path]) -> Path:
    if output_dir:
        base = ensure_directory(output_dir)
    else:
        base = ensure_directory(get_data_path("external", "coinstats_sentiment", symbol))
    return base / f"{date.isoformat()}.csv"


def fetch_coinstats_sentiment(
    symbol: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    session: Optional[requests.Session] = None,
    persist: bool = True,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch CoinStats sentiment for ``symbol`` between ``start`` and ``end``."""

    if start >= end:
        raise ValueError("start must be earlier than end")

    assets = load_assets_config()
    asset_meta = assets.get(symbol)
    if not asset_meta:
        raise KeyError(f"Unknown symbol {symbol!r} in assets config")

    credentials = _load_credentials()
    headers = {"X-API-KEY": credentials.coinstats_api_key}

    session = session or requests.Session()
    start_utc = _ensure_utc(start)
    end_utc = _ensure_utc(end)
    params = {
        "symbol": asset_meta["external_symbol"],
        "start": start_utc.replace(microsecond=0).isoformat(),
        "end": end_utc.replace(microsecond=0).isoformat(),
    }

    response = session.get(_BASE_URL, headers=headers, params=params, timeout=30)
    if response.status_code >= 400:
        raise requests.HTTPError(
            f"CoinStats error {response.status_code}: {response.text}",
            response=response,
        )

    payload = response.json()
    data = CoinStatsResponse.model_validate(payload)
    if not data.data:
        _LOGGER.warning("CoinStats returned no data for %s", symbol)
        return pd.DataFrame(columns=["timestamp", "fear_greed_score", "confidence"])

    frame = pd.DataFrame([entry.model_dump() for entry in data.data])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame.sort_values("timestamp", inplace=True)
    frame.reset_index(drop=True, inplace=True)

    hourly_index = pd.date_range(start_utc, end_utc, freq="1h", inclusive="left", tz=timezone.utc)
    frame.set_index("timestamp", inplace=True)
    frame = frame.reindex(hourly_index, method="pad")
    frame.index.name = "timestamp"
    frame.reset_index(inplace=True)

    if persist and not frame.empty:
        by_date = frame.groupby(frame["timestamp"].dt.date)
        for date, group in by_date:
            output_path = _build_output_path(symbol, date, output_dir)
            group.to_csv(output_path, index=False)
            _LOGGER.info("Wrote CoinStats sentiment data to %s", output_path)

    return frame


__all__ = ["fetch_coinstats_sentiment"]
