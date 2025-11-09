from __future__ import annotations

import json
import math
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from qcsrc.models import GaussianHMMWrapper, TrainingConfig, map_states
from qcsrc.models.thresholds import should_flat, should_long, should_short


def _generate_regime_series(samples_per_state: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2023-01-01", periods=samples_per_state * 3, freq="h", tz="UTC")
    regimes = np.repeat([0, 1, 2], samples_per_state)

    base_returns = np.select([regimes == 0, regimes == 1, regimes == 2], [-0.002, 0.0, 0.003])
    returns = base_returns + rng.normal(scale=0.0005, size=regimes.size)

    frame = pd.DataFrame(
        {
            "return_log_1h": returns,
            "return_log_4h": pd.Series(returns).rolling(4, min_periods=1).mean().values,
            "vol_6h": np.abs(rng.normal(0.01, 0.002, size=regimes.size)),
            "volume_zscore": rng.normal(0.0, 1.0, size=regimes.size),
            "liquidity_order_imbalance_z": rng.normal(0.0, 1.0, size=regimes.size),
            "sentiment_fear_greed_z": rng.normal(0.0, 1.0, size=regimes.size),
        },
        index=timestamps,
    )
    return frame


def test_gaussian_hmm_wrapper_fit_predict_proba():
    frame = _generate_regime_series()
    scaler = StandardScaler().fit(frame.values)
    scaled = scaler.transform(frame.values)

    config = TrainingConfig(n_components=3, n_iter=50, random_state=7)
    wrapper = GaussianHMMWrapper(config)
    wrapper.fit(scaled)

    posterior = wrapper.predict_proba(scaled)
    assert posterior.shape == (frame.shape[0], 3)
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, rtol=1e-5)

    mapping = map_states(wrapper.model, list(frame.columns), scaler=scaler)
    assert set(mapping.values()) == {"bullish", "bearish", "consolidation"}

    restored_means = wrapper.model.means_ * scaler.scale_ + scaler.mean_
    bullish_state = next(state for state, label in mapping.items() if label == "bullish")
    bearish_state = next(state for state, label in mapping.items() if label == "bearish")
    assert restored_means[bullish_state, 0] > restored_means[bearish_state, 0]


def test_threshold_helpers():
    probabilities = np.array([0.75, 0.2, 0.05])
    assert should_long(probabilities, 0.7) is True
    assert should_short(probabilities, 0.7) is False
    assert should_flat(probabilities, 0.7) is False

    probabilities = np.array([0.1, 0.8, 0.1])
    assert should_short(probabilities, 0.7) is True

    probabilities = np.array([0.1, 0.2, 0.9])
    assert should_flat(probabilities, 0.7) is True


def test_train_hmm_for_symbol(tmp_path, monkeypatch):
    from qcsrc.pipeline import train_hmm as train_module

    def fake_get_data_path(*parts):
        base = tmp_path / "data"
        for part in parts:
            base = base / str(part)
        return base

    monkeypatch.setattr(train_module, "get_data_path", fake_get_data_path)

    processed_dir = fake_get_data_path("processed")
    model_dir = fake_get_data_path("models", "hmm")
    diagnostics_dir = fake_get_data_path("models", "diagnostics")

    processed_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    frame = _generate_regime_series(samples_per_state=120)
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

    artifacts = train_module.train_hmm_for_symbol(
        "BTC",
        lookback_hours=300,
        min_samples=100,
        model_dir=model_dir,
        diagnostics_dir=diagnostics_dir,
    )

    assert artifacts.model_path.exists()
    assert artifacts.diagnostics_path.exists()
    assert {"mae", "rmse", "mape"}.issubset(artifacts.metrics.keys())
    assert all(math.isfinite(value) for value in artifacts.metrics.values())

    with artifacts.diagnostics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["symbol"] == "BTC"
    expected_samples = min(300, frame.shape[0])
    assert len(payload["posterior_probabilities"]) == expected_samples


