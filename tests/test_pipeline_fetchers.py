from __future__ import annotations

import datetime as dt
import json
from datetime import timezone
from typing import List

import pytest

pytest.importorskip("pandas")
pytest.importorskip("pydantic")
pytest.importorskip("requests")

from qcsrc.pipeline.fetch_binance_orderbook import fetch_binance_orderbook
from qcsrc.pipeline.fetch_coinstats_sentiment import fetch_coinstats_sentiment
from qcsrc.pipeline.fetch_cryptoquant import fetch_cryptoquant


class DummyResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


class DummySession:
    def __init__(self, responses: List[DummyResponse]):
        self._responses = responses
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "params": params})
        if not self._responses:
            raise AssertionError("No more dummy responses configured")
        return self._responses.pop(0)


@pytest.fixture(autouse=True)
def _mock_credentials(monkeypatch):
    monkeypatch.setenv("CRYPTOQUANT_API_KEY", "cryptoquant-key")
    monkeypatch.setenv("COINSTATS_API_KEY", "coinstats-key")
    monkeypatch.setenv("BINANCE_API_KEY", "binance-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "binance-secret")


def _utc(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def test_fetch_cryptoquant_returns_hourly_frame(monkeypatch, tmp_path):
    payload = {
        "result": [
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
                "high": 108,
                "low": 101,
                "close": 102,
                "volume": 9,
                "quote_volume": 910,
            },
        ]
    }

    dummy_session = DummySession([DummyResponse(payload)])
    monkeypatch.setattr("requests.Session", lambda: dummy_session)

    start = _utc("2024-01-01T00:00:00Z")
    end = _utc("2024-01-01T02:00:00Z")
    frame = fetch_cryptoquant(
        "BTCUSD",
        start,
        end,
        persist=False,
        output_dir=tmp_path,
    )

    assert list(frame.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
    ]
    assert frame.shape == (2, 7)
    assert frame["timestamp"].iloc[0] == start


def test_fetch_binance_orderbook_computes_metrics(monkeypatch, tmp_path):
    payload = {
        "lastUpdateId": 1,
        "bids": [["100.0", "2.0"], ["99.5", "3.0"]],
        "asks": [["101.0", "1.5"], ["101.5", "2.0"]],
    }

    dummy_session = DummySession([DummyResponse(payload), DummyResponse(payload)])
    monkeypatch.setattr("requests.Session", lambda: dummy_session)

    start = _utc("2024-01-01T00:00:00Z")
    end = _utc("2024-01-01T02:00:00Z")
    frame = fetch_binance_orderbook(
        "BTCUSD",
        start,
        end,
        limit=10,
        persist=False,
        output_dir=tmp_path,
    )

    assert frame.shape == (2, 7)
    assert frame["best_ask"].iloc[0] == pytest.approx(101.0)
    assert frame["best_bid"].iloc[0] == pytest.approx(100.0)
    assert frame["bid_ask_spread"].iloc[0] == pytest.approx(1.0)
    assert frame["depth_pct_1"].iloc[0] == pytest.approx(8.5)
    assert frame["depth_pct_5"].iloc[0] == pytest.approx(8.5)
    assert frame["order_imbalance"].iloc[0] == pytest.approx((5.0 - 3.5) / 8.5)


def test_fetch_coinstats_sentiment_interpolates(monkeypatch, tmp_path):
    payload = {
        "data": {
            "items": [
                {
                    "timestamp": "2024-01-01T00:00:00Z",
                    "value": 40,
                    "confidence": 0.5,
                },
                {
                    "timestamp": "2024-01-01T03:00:00Z",
                    "value": 50,
                    "confidence": 0.6,
                },
            ]
        }
    }

    dummy_session = DummySession([DummyResponse(payload)])
    monkeypatch.setattr("requests.Session", lambda: dummy_session)

    start = _utc("2024-01-01T00:00:00Z")
    end = _utc("2024-01-01T04:00:00Z")
    frame = fetch_coinstats_sentiment(
        "BTCUSD",
        start,
        end,
        persist=False,
        output_dir=tmp_path,
    )

    assert frame.shape[0] == 4
    assert frame["timestamp"].iloc[0] == start
    assert frame["fear_greed_score"].iloc[1] == pytest.approx(40)
    assert frame["fear_greed_score"].iloc[-1] == pytest.approx(50)
