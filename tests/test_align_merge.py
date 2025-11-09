from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from qcsrc.pipeline.align_merge import align_and_merge


@pytest.fixture
def project_root(monkeypatch, tmp_path):
    from qcsrc.io import file_locator

    monkeypatch.setattr(file_locator, "_PROJECT_ROOT", tmp_path)
    return tmp_path


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_align_and_merge_creates_hourly_frame(project_root):
    symbol = "BTCUSD"
    date = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)

    base = project_root / "data" / "external"
    _write_csv(
        base / "cryptoquant" / symbol / "2024-01-01.csv",
        [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "open": 100,
                "high": 110,
                "low": 95,
                "close": 105,
                "volume": 12,
                "quote_volume": 1200,
            },
            {
                "timestamp": "2024-01-01T01:00:00Z",
                "open": 105,
                "high": 112,
                "low": 100,
                "close": 108,
                "volume": 14,
                "quote_volume": 1400,
            },
            {
                "timestamp": "2024-01-01T01:00:00Z",
                "open": 106,
                "high": 113,
                "low": 101,
                "close": 109,
                "volume": 15,
                "quote_volume": 1500,
            },
            {
                "timestamp": "2024-01-01T03:00:00Z",
                "open": 110,
                "high": 118,
                "low": 107,
                "close": 116,
                "volume": 18,
                "quote_volume": 1800,
            },
        ],
    )

    _write_csv(
        base / "binance_orderbook" / symbol / "2024-01-01.csv",
        [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "best_bid": 99.5,
                "best_ask": 100.5,
                "bid_ask_spread": 1.0,
                "depth_pct_1": 5.0,
                "depth_pct_5": 12.0,
                "order_imbalance": 0.1,
            },
            {
                "timestamp": "2024-01-01T01:00:00Z",
                "best_bid": 100.0,
                "best_ask": 101.0,
                "bid_ask_spread": 1.0,
                "depth_pct_1": 6.0,
                "depth_pct_5": 13.0,
                "order_imbalance": 0.05,
            },
            {
                "timestamp": "2024-01-01T03:00:00Z",
                "best_bid": 102.0,
                "best_ask": 103.0,
                "bid_ask_spread": 1.0,
                "depth_pct_1": 7.0,
                "depth_pct_5": 14.0,
                "order_imbalance": 0.02,
            },
        ],
    )

    _write_csv(
        base / "coinstats_sentiment" / symbol / "2024-01-01.csv",
        [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "fear_greed_score": 40,
                "confidence": 0.4,
            },
            {
                "timestamp": "2024-01-01T03:00:00Z",
                "fear_greed_score": 55,
                "confidence": 0.6,
            },
        ],
    )

    result = align_and_merge(symbol, date)

    assert not result.empty
    assert set(
        [
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
    ) == set(result.columns)
    assert result["timestamp"].dt.tz == dt.timezone.utc

    # Hour with missing liquidity depth should be zero-filled
    zero_depth_rows = result[result["timestamp"] == dt.datetime(2024, 1, 1, 2, tzinfo=dt.timezone.utc)]
    if not zero_depth_rows.empty:
        assert zero_depth_rows["depth_pct_1"].iloc[0] == 0.0
        assert zero_depth_rows["depth_pct_5"].iloc[0] == 0.0

    interim_path = project_root / "data" / "interim" / f"{symbol}.parquet"
    assert interim_path.exists()


def test_align_and_merge_returns_empty_when_source_missing(project_root):
    symbol = "ETHUSD"
    date = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)

    base = project_root / "data" / "external"
    _write_csv(
        base / "cryptoquant" / symbol / "2024-01-01.csv",
        [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "open": 200,
                "high": 210,
                "low": 195,
                "close": 205,
                "volume": 22,
                "quote_volume": 2200,
            }
        ],
    )

    result = align_and_merge(symbol, date)
    assert result.empty
