"""Evaluate exported HMM probabilities against realized returns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from qcsrc.io import ensure_directory, get_data_path, get_qc_data_path, load_assets_config
from qcsrc.models import align_predictions, compute_error_metrics
from qcsrc.util import ensure_utc_index, get_logger

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    """Wrap the computed metrics for a single symbol."""

    symbol: str
    mae: float
    rmse: float
    mape: float
    direction_accuracy: float
    count: int


def _load_probability_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"probability export at {path} is empty")
    frame["time"] = ensure_utc_index(frame["time"])
    frame.sort_values("time", inplace=True)
    frame["expected_return"] = frame["prob_bullish"] - frame["prob_bearish"]
    return frame


def _load_feature_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"feature matrix at {path} is empty")
    frame["timestamp"] = ensure_utc_index(frame["timestamp"])
    frame.sort_values("timestamp", inplace=True)
    frame["future_return"] = frame["return_log_1h"].shift(-1)
    return frame[["timestamp", "future_return"]]


def evaluate_symbol(
    symbol: str,
    *,
    feature_path: Optional[Path] = None,
    probability_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> EvaluationResult:
    """Compute error metrics for ``symbol`` and persist them to diagnostics."""

    feature_path = feature_path or Path(get_data_path("processed")) / f"{symbol}_features.parquet"
    probability_path = probability_path or Path(get_qc_data_path("custom", "HMMStateProba")) / f"{symbol.lower()}.csv"

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature matrix missing for {symbol} at {feature_path}")
    if not probability_path.exists():
        raise FileNotFoundError(f"Probability export missing for {symbol} at {probability_path}")

    features = _load_feature_frame(feature_path)
    probabilities = _load_probability_frame(probability_path)

    merged = probabilities.merge(
        features, left_on="time", right_on="timestamp", how="inner", suffixes=("_prob", "_feat")
    )
    merged.dropna(subset=["future_return"], inplace=True)
    pairs = align_predictions(zip(merged["expected_return"], merged["future_return"]))

    metrics = compute_error_metrics(pairs)
    result = EvaluationResult(
        symbol=symbol,
        mae=metrics.mae,
        rmse=metrics.rmse,
        mape=metrics.mape,
        direction_accuracy=metrics.direction_accuracy,
        count=metrics.count,
    )

    payload = {
        "symbol": symbol,
        "count": result.count,
        "mae": result.mae,
        "rmse": result.rmse,
        "mape": result.mape,
        "direction_accuracy": result.direction_accuracy,
    }

    target_dir = ensure_directory(output_dir or get_data_path("models", "diagnostics"))
    output_path = Path(target_dir) / f"{symbol}_metrics.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _LOGGER.info("Wrote evaluation metrics for %s to %s", symbol, output_path)

    return result


def evaluate_all(symbols: Optional[Iterable[str]] = None) -> list[EvaluationResult]:
    assets = load_assets_config()
    selected = symbols or assets.keys()
    results = []
    for symbol in selected:
        try:
            results.append(evaluate_symbol(symbol))
        except (FileNotFoundError, ValueError) as exc:
            _LOGGER.warning("Skipping evaluation for %s: %s", symbol, exc)
    return results


def main() -> None:
    evaluate_all()


if __name__ == "__main__":
    main()
