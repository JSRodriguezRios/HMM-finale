from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from qcsrc.pipeline.evaluate import evaluate_symbol


def _write_probability_csv(path, timestamps, bullish, bearish, consolidation):
    frame = pd.DataFrame(
        {
            "time": timestamps,
            "prob_bullish": bullish,
            "prob_bearish": bearish,
            "prob_consolidation": consolidation,
        }
    )
    frame.to_csv(path, index=False)


def _write_feature_parquet(path, timestamps, returns):
    frame = pd.DataFrame({"timestamp": timestamps, "return_log_1h": returns})
    frame.to_parquet(path, index=False)


def test_evaluate_symbol_persists_metrics(tmp_path):
    timestamps = [
        datetime(2023, 1, 1, hour, tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        for hour in range(3)
    ]
    probability_path = tmp_path / "btc.csv"
    feature_path = tmp_path / "btc_features.parquet"
    output_dir = tmp_path / "diagnostics"

    _write_probability_csv(
        probability_path,
        timestamps,
        bullish=[0.8, 0.2, 0.7],
        bearish=[0.1, 0.6, 0.2],
        consolidation=[0.1, 0.2, 0.1],
    )

    feature_timestamps = [datetime(2023, 1, 1, hour, tzinfo=timezone.utc) for hour in range(3)]
    returns = [0.05, -0.02, 0.03]
    _write_feature_parquet(feature_path, feature_timestamps, returns)

    result = evaluate_symbol(
        "BTC",
        feature_path=feature_path,
        probability_path=probability_path,
        output_dir=output_dir,
    )

    assert result.symbol == "BTC"
    assert result.count > 0

    metrics_path = output_dir / "BTC_metrics.json"
    assert metrics_path.exists()

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["symbol"] == "BTC"
    assert payload["count"] == result.count
