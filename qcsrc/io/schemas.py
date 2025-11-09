"""Pydantic schemas describing external data payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CryptoQuantEntry(BaseModel):
    """Schema for CryptoQuant OHLCV payloads."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = Field(..., alias="volume")
    quote_volume: float

    model_config = ConfigDict(populate_by_name=True)


class BinanceOrderBookEntry(BaseModel):
    """Schema describing derived Binance order book metrics."""

    timestamp: datetime
    best_bid: float
    best_ask: float
    bid_ask_spread: float
    depth_pct_1: float
    depth_pct_5: float
    order_imbalance: float


class CoinStatsSentimentEntry(BaseModel):
    """Schema for CoinStats sentiment time series."""

    timestamp: datetime
    fear_greed_score: float
    confidence: float = 1.0

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _coerce_timestamp(cls, value: Any):
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str) and value.isdigit():
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        return value

    @field_validator("fear_greed_score", mode="before")
    @classmethod
    def _coerce_score(cls, value: Any):
        if isinstance(value, dict):
            for key in ("value", "score", "index"):
                if key in value:
                    return value[key]
        if value is None:
            raise ValueError("CoinStats fear/greed score missing from entry")
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def _default_confidence(cls, value: Any):
        if value is None:
            return 1.0
        return value


class CryptoQuantResponse(BaseModel):
    result: List[CryptoQuantEntry]


class CoinStatsResponse(BaseModel):
    data: List[CoinStatsSentimentEntry]

    @field_validator("data", mode="before")
    @classmethod
    def _extract_data(cls, value: Any):
        if isinstance(value, dict):
            if "items" in value:
                value = value["items"]
            elif "data" in value:
                inner = value["data"]
                if isinstance(inner, dict) and "items" in inner:
                    value = inner["items"]
                else:
                    value = inner
        entries = []
        for entry in value or []:
            if isinstance(entry, dict):
                entries.append(
                    {
                        "timestamp": entry.get("timestamp")
                        or entry.get("time")
                        or entry.get("date"),
                        "fear_greed_score": entry.get("fear_greed_score")
                        or entry.get("score")
                        or entry.get("value"),
                        "confidence": entry.get("confidence")
                        or entry.get("confidence_score")
                        or entry.get("scoreConfidence"),
                    }
                )
            else:
                entries.append(entry)
        return entries


__all__ = [
    "BinanceOrderBookEntry",
    "CoinStatsResponse",
    "CoinStatsSentimentEntry",
    "CryptoQuantEntry",
    "CryptoQuantResponse",
]
