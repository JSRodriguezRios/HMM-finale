from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from qcsrc.io.custom_export import export_probability_series


def _synthetic_feature_frame(samples_per_state: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    index = pd.date_range("2023-01-01", periods=samples_per_state * 3, freq="h", tz="UTC")
    regimes = np.repeat(["bearish", "consolidation", "bullish"], samples_per_state)

    base_returns = np.select(
        [regimes == "bullish", regimes == "bearish", regimes == "consolidation"],
        [0.003, -0.002, 0.0],
    )
    returns = base_returns + rng.normal(scale=0.0005, size=index.size)

    frame = pd.DataFrame(
        {
            "return_log_1h": returns,
            "return_log_4h": pd.Series(returns).rolling(4, min_periods=1).mean().values,
            "vol_6h": np.abs(rng.normal(0.01, 0.002, size=index.size)),
            "volume_zscore": rng.normal(0.0, 1.0, size=index.size),
            "liquidity_order_imbalance_z": rng.normal(0.0, 1.0, size=index.size),
            "sentiment_fear_greed_z": rng.normal(0.0, 1.0, size=index.size),
        },
        index=index,
    )
    frame.index.name = "timestamp"
    return frame


def test_export_probability_series_writes_csv(tmp_path):
    index = pd.date_range("2023-01-01", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "prob_bullish": [0.7, 0.1, 0.2],
            "prob_bearish": [0.2, 0.8, 0.2],
            "prob_consolidation": [0.1, 0.1, 0.6],
        },
        index=index,
    )
    frame.index.name = "timestamp"

    path = export_probability_series("BTCUSD", frame, output_dir=tmp_path)
    assert path.exists()

    exported = pd.read_csv(path)
    assert list(exported.columns) == [
        "time",
        "prob_bullish",
        "prob_bearish",
        "prob_consolidation",
    ]
    assert exported.iloc[0]["prob_bullish"] == pytest.approx(0.7)


def test_infer_probabilities_for_symbol(tmp_path, monkeypatch):
    from qcsrc.pipeline import infer_proba as infer_module
    from qcsrc.pipeline import train_hmm as train_module

    def fake_get_data_path(*parts):
        base = tmp_path / "data"
        for part in parts:
            base = base / str(part)
        return base

    monkeypatch.setattr(train_module, "get_data_path", fake_get_data_path)
    monkeypatch.setattr(infer_module, "get_data_path", fake_get_data_path)

    processed_dir = fake_get_data_path("processed")
    model_dir = fake_get_data_path("models", "hmm")
    diagnostics_dir = fake_get_data_path("models", "diagnostics")

    processed_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    frame = _synthetic_feature_frame(samples_per_state=80)
    scaler = StandardScaler().fit(frame.values)
    scaled = pd.DataFrame(
        scaler.transform(frame.values),
        columns=frame.columns,
        index=frame.index,
    )
    scaled.index.name = "timestamp"

    feature_path = processed_dir / "BTC_features.parquet"
    scaled.reset_index().to_parquet(feature_path, index=False)

    scaler_path = model_dir / "BTC_scaler.pkl"
    with scaler_path.open("wb") as handle:
        pickle.dump(scaler, handle)

    train_module.train_hmm_for_symbol(
        "BTC",
        lookback_hours=200,
        min_samples=100,
        model_dir=model_dir,
        diagnostics_dir=diagnostics_dir,
    )

    export_dir = tmp_path / "exports"
    artifacts = infer_module.infer_probabilities_for_symbol(
        "BTC",
        processed_dir=processed_dir,
        model_dir=model_dir,
        output_dir=export_dir,
    )

    assert artifacts.export_path.exists()
    assert artifacts.probabilities.shape[0] == frame.shape[0]
    np.testing.assert_allclose(
        artifacts.probabilities.sum(axis=1).values,
        1.0,
        rtol=1e-5,
    )

    exported = pd.read_csv(artifacts.export_path)
    assert exported.shape[0] == frame.shape[0]
    assert set(exported.columns) == {
        "time",
        "prob_bullish",
        "prob_bearish",
        "prob_consolidation",
    }
