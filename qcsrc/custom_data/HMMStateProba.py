"""QuantConnect custom data definition for HMM posterior probabilities."""

from __future__ import annotations

from enum import Enum

from qcsrc.custom_data._base import BaseCustomSeries, CustomDataColumn
from qcsrc.models.thresholds import should_flat, should_long, should_short


class SignalType(str, Enum):
    """Trading signals derived from HMM state probabilities."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class HMMStateProba(BaseCustomSeries):
    """Expose hourly posterior probabilities for each hidden state."""

    csv_columns = (
        CustomDataColumn("prob_bullish", "ProbBullish"),
        CustomDataColumn("prob_bearish", "ProbBearish"),
        CustomDataColumn("prob_consolidation", "ProbConsolidation"),
    )
    csv_headers = ("time",) + tuple(column.column_name for column in csv_columns)
    subdirectory = ("HMMStateProba",)
    primary_value_column = "ProbBullish"

    ProbBullish: float = 0.0
    ProbBearish: float = 0.0
    ProbConsolidation: float = 0.0

    def GetSignal(self, threshold: float = 0.7) -> SignalType:
        """Return the trading signal implied by the probability vector."""

        probabilities = (self.ProbBullish, self.ProbBearish, self.ProbConsolidation)
        if should_long(probabilities, threshold):
            return SignalType.LONG
        if should_short(probabilities, threshold):
            return SignalType.SHORT
        if should_flat(probabilities, threshold):
            return SignalType.FLAT
        return SignalType.FLAT


__all__ = ["HMMStateProba", "SignalType"]
