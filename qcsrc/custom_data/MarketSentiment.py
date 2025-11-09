"""QuantConnect custom data definition for CoinStats sentiment metrics."""

from __future__ import annotations

from qcsrc.custom_data._base import BaseCustomSeries, CustomDataColumn


class MarketSentiment(BaseCustomSeries):
    """Expose hourly fear/greed sentiment scores to QuantConnect."""

    csv_columns = (
        CustomDataColumn("fear_greed_score", "FearGreedScore"),
        CustomDataColumn("confidence", "Confidence"),
    )
    csv_headers = ("time",) + tuple(column.column_name for column in csv_columns)
    subdirectory = ("MarketSentiment",)
    primary_value_column = "FearGreedScore"

    FearGreedScore: float = 0.0
    Confidence: float = 0.0


__all__ = ["MarketSentiment"]
