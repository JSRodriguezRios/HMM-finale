"""Fetch OHLCV data from the CryptoQuant API."""

from __future__ import annotations

import datetime as dt
from datetime import timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import requests
from requests import Response

from qcsrc.io import ensure_directory, get_data_path, load_assets_config
from qcsrc.io.schemas import CryptoQuantResponse
from qcsrc.util.logging_utils import get_logger
from qcsrc.util.secrets import CredentialSet, MissingCredentialError, load_credentials

_LOGGER = get_logger(__name__)
_BASE_URL = "https://api.cryptoquant.com/v1/markets"


def _ensure_utc(timestamp: dt.datetime) -> dt.datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _request(
    session: requests.Session,
    endpoint: str,
    headers: Dict[str, str],
    params: Dict[str, str],
) -> Response:
    response = session.get(endpoint, headers=headers, params=params, timeout=30)
    if response.status_code >= 400:
        raise requests.HTTPError(
            f"CryptoQuant error {response.status_code}: {response.text}",
            response=response,
        )
    return response


def _build_output_path(symbol: str, date: dt.date, output_dir: Optional[Path]) -> Path:
    if output_dir:
        base = ensure_directory(output_dir)
    else:
        base = ensure_directory(get_data_path("external", "cryptoquant", symbol))
    return base / f"{date.isoformat()}.csv"


def _load_credentials() -> CredentialSet:
    try:
        return load_credentials()
    except MissingCredentialError as exc:
        raise RuntimeError("CryptoQuant credentials are required for fetches") from exc


def fetch_cryptoquant(
    symbol: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    window: str = "hour",
    session: Optional[requests.Session] = None,
    persist: bool = True,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch OHLCV candles for ``symbol`` between ``start`` and ``end``."""

    if start >= end:
        raise ValueError("start must be earlier than end")

    assets = load_assets_config()
    asset_meta = assets.get(symbol)
    if not asset_meta:
        raise KeyError(f"Unknown symbol {symbol!r} in assets config")

    credentials = _load_credentials()
    headers = {"x-api-key": credentials.cryptoquant_api_key}

    external_symbol = asset_meta["external_symbol"]
    start_utc = _ensure_utc(start)
    end_utc = _ensure_utc(end)
    params = {
        "symbol": external_symbol,
        "window": window,
        "from": start_utc.replace(microsecond=0).isoformat(),
        "to": end_utc.replace(microsecond=0).isoformat(),
    }

    session = session or requests.Session()
    endpoint = f"{_BASE_URL}/{external_symbol}/ohlcv"
    response = _request(session, endpoint, headers=headers, params=params)
    payload = response.json()

    data = CryptoQuantResponse.parse_obj(payload)
    if not data.result:
        _LOGGER.warning("CryptoQuant returned no data for %s", symbol)
        return pd.DataFrame(columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
        ])

    frame = pd.DataFrame([entry.dict() for entry in data.result])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame.sort_values("timestamp", inplace=True)
    frame.reset_index(drop=True, inplace=True)

    if persist:
        dates: Iterable[dt.date] = frame["timestamp"].dt.date.unique()
        for date in dates:
            day_frame = frame[frame["timestamp"].dt.date == date]
            output_path = _build_output_path(symbol, date, output_dir)
            day_frame.to_csv(output_path, index=False)
            _LOGGER.info("Wrote CryptoQuant data to %s", output_path)

    return frame


__all__ = ["fetch_cryptoquant"]
