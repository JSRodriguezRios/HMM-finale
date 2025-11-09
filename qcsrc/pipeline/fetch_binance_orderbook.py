"""Fetch Binance order book snapshots and derive liquidity metrics."""

from __future__ import annotations

import datetime as dt
from datetime import timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import pandas as pd
import requests

from qcsrc.io import ensure_directory, get_data_path, load_assets_config
from qcsrc.io.schemas import BinanceOrderBookEntry
from qcsrc.util.logging_utils import get_logger
from qcsrc.util.secrets import CredentialSet, MissingCredentialError, load_credentials

_LOGGER = get_logger(__name__)
_BASE_URL = "https://api.binance.com/api/v3/depth"


def _ensure_utc(timestamp: dt.datetime) -> dt.datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _load_credentials() -> CredentialSet:
    try:
        return load_credentials()
    except MissingCredentialError as exc:
        raise RuntimeError("Binance credentials are required for fetches") from exc


def _build_output_path(symbol: str, timestamp: dt.datetime, output_dir: Optional[Path]) -> Path:
    if output_dir:
        base = ensure_directory(output_dir)
    else:
        base = ensure_directory(get_data_path("external", "binance_orderbook", symbol))
    return base / f"{timestamp:%Y-%m-%d}.csv"


def _normalize_orders(raw: List[List[str]]) -> List[Tuple[float, float]]:
    return [(float(price), float(quantity)) for price, quantity in raw]


def _depth_within_pct(
    orders: Sequence[Tuple[float, float]], mid_price: float, pct: float, side: str
) -> float:
    if not orders:
        return 0.0

    if side == "bid":
        threshold = mid_price * (1 - pct / 100)
        volumes = [qty for price, qty in orders if price >= threshold]
    else:
        threshold = mid_price * (1 + pct / 100)
        volumes = [qty for price, qty in orders if price <= threshold]
    return float(sum(volumes))


def _orderbook_metrics(
    bids: Sequence[Tuple[float, float]],
    asks: Sequence[Tuple[float, float]],
    timestamp: dt.datetime,
) -> BinanceOrderBookEntry:
    if not bids or not asks:
        raise ValueError("Binance order book must contain both bids and asks")

    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid = (best_bid + best_ask) / 2

    depth_bid_1 = _depth_within_pct(bids, mid, 1, "bid")
    depth_ask_1 = _depth_within_pct(asks, mid, 1, "ask")
    depth_bid_5 = _depth_within_pct(bids, mid, 5, "bid")
    depth_ask_5 = _depth_within_pct(asks, mid, 5, "ask")

    depth_pct_1 = depth_bid_1 + depth_ask_1
    depth_pct_5 = depth_bid_5 + depth_ask_5

    total_bid = sum(qty for _, qty in bids)
    total_ask = sum(qty for _, qty in asks)
    denominator = total_bid + total_ask
    order_imbalance = float((total_bid - total_ask) / denominator) if denominator else 0.0

    return BinanceOrderBookEntry(
        timestamp=timestamp,
        best_bid=best_bid,
        best_ask=best_ask,
        bid_ask_spread=best_ask - best_bid,
        depth_pct_1=depth_pct_1,
        depth_pct_5=depth_pct_5,
        order_imbalance=order_imbalance,
    )


def fetch_binance_orderbook(
    symbol: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    limit: int = 100,
    session: Optional[requests.Session] = None,
    persist: bool = True,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch Binance order book snapshots between ``start`` and ``end``."""

    if start >= end:
        raise ValueError("start must be earlier than end")

    assets = load_assets_config()
    asset_meta = assets.get(symbol)
    if not asset_meta:
        raise KeyError(f"Unknown symbol {symbol!r} in assets config")

    credentials = _load_credentials()
    headers = {"X-MBX-APIKEY": credentials.binance_api_key}

    session = session or requests.Session()

    start_utc = _ensure_utc(start)
    end_utc = _ensure_utc(end)
    timestamps: List[dt.datetime] = list(
        pd.date_range(start_utc, end_utc, freq="1h", inclusive="left", tz=timezone.utc)
    )
    rows = []

    for ts in timestamps:
        params = {"symbol": asset_meta["qc_ticker"], "limit": str(limit)}
        response = session.get(_BASE_URL, headers=headers, params=params, timeout=30)
        if response.status_code >= 400:
            raise requests.HTTPError(
                f"Binance error {response.status_code}: {response.text}",
                response=response,
            )

        payload = response.json()
        bids = _normalize_orders(payload.get("bids", []))
        asks = _normalize_orders(payload.get("asks", []))
        entry = _orderbook_metrics(bids, asks, ts)
        rows.append(entry.model_dump())

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame.sort_values("timestamp", inplace=True)
    frame.reset_index(drop=True, inplace=True)

    if persist and not frame.empty:
        by_date = frame.groupby(frame["timestamp"].dt.date)
        for date, group in by_date:
            output_path = _build_output_path(symbol, group["timestamp"].iloc[0], output_dir)
            if output_path.exists():
                existing = pd.read_csv(output_path, parse_dates=["timestamp"])
                existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
                group = pd.concat([existing, group], ignore_index=True)
                group.drop_duplicates(subset="timestamp", inplace=True)
                group.sort_values("timestamp", inplace=True)
            group.to_csv(output_path, index=False)
            _LOGGER.info("Wrote Binance order book data to %s", output_path)

    return frame


__all__ = ["fetch_binance_orderbook"]
