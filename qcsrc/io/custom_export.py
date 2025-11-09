"""Utilities for exporting pipeline outputs to QuantConnect custom data format."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from qcsrc.io.file_locator import ensure_directory, get_qc_data_path
from qcsrc.util.logging_utils import get_logger

_LOGGER = get_logger(__name__)

_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
_PROBABILITY_COLUMNS: Sequence[str] = (
    "prob_bullish",
    "prob_bearish",
    "prob_consolidation",
)
_LIQUIDITY_COLUMNS: Sequence[str] = (
    "bid_ask_spread",
    "depth_pct_1",
    "depth_pct_5",
    "order_imbalance",
)
_SENTIMENT_COLUMNS: Sequence[str] = (
    "fear_greed_score",
    "confidence",
)


def _prepare_directory(directory: Optional[Path], subdirs: Iterable[str]) -> Path:
    """Resolve and create the export directory."""

    if directory is None:
        directory = get_qc_data_path("custom", *subdirs)
    return ensure_directory(Path(directory))


def _normalize_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure frames use a UTC datetime index."""

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("export frame must be indexed by pandas.DatetimeIndex")

    index = frame.index
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")

    normalized = frame.copy()
    normalized.index = index
    return normalized


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    payload = frame.reset_index().rename(columns={frame.index.name or "index": "time"})
    payload["time"] = payload["time"].dt.strftime(_TIME_FORMAT)
    payload.to_csv(path, index=False)


def export_probability_series(
    symbol: str,
    probabilities: pd.DataFrame,
    *,
    output_dir: Optional[Path] = None,
) -> Path:
    """Persist posterior probabilities for ``symbol`` to QuantConnect CSV layout."""

    normalized = _normalize_index(probabilities)
    missing = set(_PROBABILITY_COLUMNS).difference(normalized.columns)
    if missing:
        raise ValueError(
            f"probabilities frame missing required columns: {sorted(missing)}"
        )

    ordered = normalized.loc[:, list(_PROBABILITY_COLUMNS)]
    directory = _prepare_directory(output_dir, ("HMMStateProba",))
    path = directory / f"{symbol.lower()}.csv"

    _write_csv(path, ordered)
    _LOGGER.info("Exported probability series for %s to %s", symbol, path)
    return path


def export_liquidity_frame(
    symbol: str,
    liquidity: pd.DataFrame,
    *,
    output_dir: Optional[Path] = None,
) -> Path:
    """Export liquidity metrics to QuantConnect custom data layout."""

    normalized = _normalize_index(liquidity)
    missing = set(_LIQUIDITY_COLUMNS).difference(normalized.columns)
    if missing:
        raise ValueError(
            f"liquidity frame missing required columns: {sorted(missing)}"
        )

    ordered = normalized.loc[:, list(_LIQUIDITY_COLUMNS)]
    directory = _prepare_directory(output_dir, ("LiquidityBitAsk",))
    path = directory / f"{symbol.lower()}.csv"

    _write_csv(path, ordered)
    _LOGGER.info("Exported liquidity series for %s to %s", symbol, path)
    return path


def export_sentiment_frame(
    symbol: str,
    sentiment: pd.DataFrame,
    *,
    output_dir: Optional[Path] = None,
) -> Path:
    """Export sentiment metrics to QuantConnect custom data layout."""

    normalized = _normalize_index(sentiment)
    missing = set(_SENTIMENT_COLUMNS).difference(normalized.columns)
    if missing:
        raise ValueError(
            f"sentiment frame missing required columns: {sorted(missing)}"
        )

    ordered = normalized.loc[:, list(_SENTIMENT_COLUMNS)]
    directory = _prepare_directory(output_dir, ("MarketSentiment",))
    path = directory / f"{symbol.lower()}.csv"

    _write_csv(path, ordered)
    _LOGGER.info("Exported sentiment series for %s to %s", symbol, path)
    return path


__all__ = [
    "export_liquidity_frame",
    "export_probability_series",
    "export_sentiment_frame",
]
