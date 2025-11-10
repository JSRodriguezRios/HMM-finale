"""Evaluation utilities for comparing predicted vs. realized returns."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


@dataclass(frozen=True)
class ErrorMetrics:
    """Aggregate error statistics for probability-implied returns."""

    mae: float
    rmse: float
    mape: float
    direction_accuracy: float
    count: int


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator`` guarding against zero denominators."""

    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_error_metrics(pairs: Sequence[Tuple[float, float]]) -> ErrorMetrics:
    """Compute MAE, RMSE, MAPE, and directional accuracy for ``pairs``."""

    if not pairs:
        return ErrorMetrics(0.0, 0.0, 0.0, 0.0, 0)

    absolute_errors = [abs(pred - actual) for pred, actual in pairs]
    squared_errors = [(pred - actual) ** 2 for pred, actual in pairs]

    mae = sum(absolute_errors) / len(pairs)
    rmse = math.sqrt(sum(squared_errors) / len(pairs))

    ape_values = []
    directional_hits = 0
    evaluated = 0
    for predicted, actual in pairs:
        if actual != 0:
            ape_values.append(abs((actual - predicted) / actual))
        # Treat zero as neutral for accuracy; only count when actual != 0.
        if actual != 0:
            evaluated += 1
            if math.copysign(1, predicted or 0.0) == math.copysign(1, actual):
                directional_hits += 1

    mape = sum(ape_values) / len(ape_values) if ape_values else 0.0
    direction_accuracy = _safe_divide(directional_hits, evaluated)

    return ErrorMetrics(mae, rmse, mape, direction_accuracy, len(pairs))


def align_predictions(
    predictions: Iterable[Tuple[float, float]],
) -> Sequence[Tuple[float, float]]:
    """Normalize ``predictions`` into a list of ``(predicted, actual)`` pairs."""

    normalized = []
    for predicted, actual in predictions:
        if predicted is None or actual is None:
            continue
        normalized.append((float(predicted), float(actual)))
    return normalized


__all__ = ["ErrorMetrics", "align_predictions", "compute_error_metrics"]
