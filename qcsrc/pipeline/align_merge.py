"""Align external data sources into a single hourly frame."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import pandas as pd

from qcsrc.io import ensure_directory, get_data_path
from qcsrc.io.schemas import validate_interim_frame
from qcsrc.util import ensure_utc, ensure_utc_index, get_logger, hourly_range

_LOGGER = get_logger(__name__)


def _load_daily_csv(base_dir: Path, symbol: str, date: dt.date) -> pd.DataFrame:
    path = base_dir / symbol / f"{date.isoformat()}.csv"
    if not path.exists():
        _LOGGER.warning("Missing data file: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _prepare_ohlcv(frame: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["timestamp"] = ensure_utc_index(frame["timestamp"])
    frame.sort_values("timestamp", inplace=True)
    frame.drop_duplicates("timestamp", keep="last", inplace=True)
    frame.set_index("timestamp", inplace=True)
    frame = frame.reindex(index)
    frame = frame.ffill(limit=1)
    frame = frame.dropna()
    return frame


def _prepare_liquidity(frame: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["timestamp"] = ensure_utc_index(frame["timestamp"])
    frame.sort_values("timestamp", inplace=True)
    frame.drop_duplicates("timestamp", keep="last", inplace=True)
    frame.set_index("timestamp", inplace=True)
    numeric_cols = [
        "best_bid",
        "best_ask",
        "bid_ask_spread",
        "depth_pct_1",
        "depth_pct_5",
        "order_imbalance",
    ]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
    frame = frame.reindex(index)
    for column in ["depth_pct_1", "depth_pct_5", "order_imbalance"]:
        frame[column] = frame[column].fillna(0.0)
    frame["best_bid"] = frame["best_bid"].ffill()
    frame["best_ask"] = frame["best_ask"].ffill()
    frame["bid_ask_spread"] = frame["bid_ask_spread"].ffill()
    frame = frame.dropna(subset=["best_bid", "best_ask", "bid_ask_spread"])
    return frame


def _prepare_sentiment(frame: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["timestamp"] = ensure_utc_index(frame["timestamp"])
    frame.sort_values("timestamp", inplace=True)
    frame.drop_duplicates("timestamp", keep="last", inplace=True)
    frame.set_index("timestamp", inplace=True)
    frame = frame.reindex(index)
    frame = frame.ffill()
    frame = frame.dropna()
    return frame


def align_and_merge(
    symbol: str,
    when: dt.datetime,
    *,
    persist: bool = True,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Align OHLCV, liquidity, and sentiment data for ``symbol`` on ``when``."""

    day = ensure_utc(when).date()
    start = dt.datetime.combine(day, dt.time.min, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(days=1)
    index = hourly_range(start, end)

    base = get_data_path("external")
    ohlcv = _load_daily_csv(base / "cryptoquant", symbol, day)
    liquidity = _load_daily_csv(base / "binance_orderbook", symbol, day)
    sentiment = _load_daily_csv(base / "coinstats_sentiment", symbol, day)

    if ohlcv.empty or liquidity.empty or sentiment.empty:
        _LOGGER.warning(
            "Insufficient data to align %s on %s: ohlcv=%s liquidity=%s sentiment=%s",
            symbol,
            day,
            not ohlcv.empty,
            not liquidity.empty,
            not sentiment.empty,
        )
        return pd.DataFrame(
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "best_bid",
                "best_ask",
                "bid_ask_spread",
                "depth_pct_1",
                "depth_pct_5",
                "order_imbalance",
                "fear_greed_score",
                "confidence",
            ]
        )

    ohlcv_prepared = _prepare_ohlcv(ohlcv, index)
    liquidity_prepared = _prepare_liquidity(liquidity, index)
    sentiment_prepared = _prepare_sentiment(sentiment, index)

    merged = ohlcv_prepared.join(liquidity_prepared, how="inner")
    merged = merged.join(sentiment_prepared, how="inner")
    merged.reset_index(inplace=True)
    merged.rename(columns={"index": "timestamp"}, inplace=True)

    if merged.empty:
        _LOGGER.warning("Merged frame empty for %s on %s", symbol, day)
        return merged

    validate_interim_frame(merged)

    if persist:
        target_dir = output_dir or get_data_path("interim")
        ensure_directory(target_dir)
        output_path = Path(target_dir) / f"{symbol}.parquet"
        if output_path.exists():
            existing = pd.read_parquet(output_path)
            merged = pd.concat([existing, merged], ignore_index=True)
            merged.drop_duplicates(subset="timestamp", keep="last", inplace=True)
            merged.sort_values("timestamp", inplace=True)
        merged.to_parquet(output_path, index=False)
        _LOGGER.info("Wrote interim data for %s to %s", symbol, output_path)

    return merged


__all__ = ["align_and_merge"]
