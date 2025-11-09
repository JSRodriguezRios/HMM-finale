"""Pydantic schemas describing external data payloads."""

from __future__ import annotations

from datetime import datetime
from typing import List

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
    confidence: float


class CryptoQuantResponse(BaseModel):
    result: List[CryptoQuantEntry]


class CoinStatsResponse(BaseModel):
    data: List[CoinStatsSentimentEntry]

    @field_validator("data", mode="before")
    @classmethod
    def _extract_data(cls, value):
        if isinstance(value, dict) and "items" in value:
            return value["items"]
        return value


__all__ = [
    "BinanceOrderBookEntry",
    "CoinStatsResponse",
    "CoinStatsSentimentEntry",
    "CryptoQuantEntry",
    "CryptoQuantResponse",
]
