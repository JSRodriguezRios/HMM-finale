"""QuantConnect custom data definition for liquidity metrics."""

from __future__ import annotations

from qcsrc.custom_data._base import BaseCustomSeries, CustomDataColumn


class LiquidityBitAsk(BaseCustomSeries):
    """Custom data class exposing hourly Binance order book metrics."""

    csv_columns = (
        CustomDataColumn("bid_ask_spread", "BidAskSpread"),
        CustomDataColumn("depth_pct_1", "DepthPct1"),
        CustomDataColumn("depth_pct_5", "DepthPct5"),
        CustomDataColumn("order_imbalance", "OrderImbalance"),
    )
    csv_headers = ("time",) + tuple(column.column_name for column in csv_columns)
    subdirectory = ("LiquidityBitAsk",)
    primary_value_column = "BidAskSpread"

    BidAskSpread: float = 0.0
    DepthPct1: float = 0.0
    DepthPct5: float = 0.0
    OrderImbalance: float = 0.0


__all__ = ["LiquidityBitAsk"]
