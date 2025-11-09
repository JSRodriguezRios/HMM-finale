"""Orchestrate external data fetches and feature generation for configured assets."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from qcsrc.features import build_feature_matrix
from qcsrc.io import load_assets_config
from qcsrc.util.logging_utils import get_logger

from qcsrc.io.file_locator import get_data_path

from .align_merge import align_and_merge
from .evaluate import evaluate_symbol
from .fetch_binance_orderbook import fetch_binance_orderbook
from .fetch_coinstats_sentiment import fetch_coinstats_sentiment
from .fetch_cryptoquant import fetch_cryptoquant
from .infer_proba import infer_probabilities_for_symbol
from .train_hmm import train_hmm_for_symbol

_LOGGER = get_logger(__name__)


def _iter_assets(symbols: Optional[Iterable[str]] = None) -> Iterable[str]:
    assets = load_assets_config()
    if symbols is None:
        return assets.keys()
    return symbols


def run_pipeline(
    start: dt.datetime,
    end: dt.datetime,
    *,
    symbols: Optional[Iterable[str]] = None,
) -> None:
    """Run all data fetchers for the configured assets."""

    if start >= end:
        raise ValueError("start must be earlier than end")

    for symbol in _iter_assets(symbols):
        _LOGGER.info("Fetching CryptoQuant OHLCV for %s", symbol)
        fetch_cryptoquant(symbol, start, end)

        _LOGGER.info("Fetching Binance order book for %s", symbol)
        fetch_binance_orderbook(symbol, start, end)

        _LOGGER.info("Fetching CoinStats sentiment for %s", symbol)
        fetch_coinstats_sentiment(symbol, start, end)

        current = start
        while current < end:
            _LOGGER.info("Aligning data for %s at %s", symbol, current.date())
            align_and_merge(symbol, current)
            current += dt.timedelta(days=1)

        interim_path = Path(get_data_path("interim")) / f"{symbol}.parquet"
        if interim_path.exists():
            _LOGGER.info("Building feature matrix for %s", symbol)
            frame = pd.read_parquet(interim_path)
            build_feature_matrix(symbol, frame)

            try:
                _LOGGER.info("Training HMM for %s", symbol)
                train_hmm_for_symbol(symbol)
            except ValueError as error:
                _LOGGER.warning("Skipping HMM training for %s: %s", symbol, error)
            except FileNotFoundError as error:
                _LOGGER.warning("Missing artifacts for %s: %s", symbol, error)

            try:
                _LOGGER.info("Running probability inference for %s", symbol)
                infer_probabilities_for_symbol(symbol)
            except (ValueError, FileNotFoundError) as error:
                _LOGGER.warning("Skipping probability export for %s: %s", symbol, error)
                continue

            try:
                _LOGGER.info("Evaluating probability performance for %s", symbol)
                evaluate_symbol(symbol)
            except (ValueError, FileNotFoundError) as error:
                _LOGGER.warning("Skipping evaluation for %s: %s", symbol, error)
        else:
            _LOGGER.warning(
                "Skipping feature build for %s because interim data is missing at %s",
                symbol,
                interim_path,
            )


def main() -> None:
    end = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start = end - dt.timedelta(hours=24)
    run_pipeline(start, end)


if __name__ == "__main__":
    main()
