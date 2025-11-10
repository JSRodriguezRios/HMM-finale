from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("sklearn")

from qcsrc.features import build_feature_matrix


@pytest.fixture
def project_root(monkeypatch, tmp_path):
    from qcsrc.io import file_locator

    monkeypatch.setattr(file_locator, "_PROJECT_ROOT", tmp_path)
    return tmp_path


def _write_features_config(config_dir):
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "features.yaml").write_text(
        """
returns:
  windows: [1, 4]
volatility:
  windows: [6]
liquidity:
  depth_pct_levels: [1, 5]
  include_bid_ask_spread: true
  zscore_window: 6
sentiment:
  lags: [1, 3]
  zscore_window: 6
"""
    )


def _build_interim_frame(hours: int) -> pd.DataFrame:
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    timestamps = [start + dt.timedelta(hours=i) for i in range(hours)]
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": np.linspace(100, 100 + hours - 1, hours),
            "volume": np.linspace(1000, 1000 + hours - 1, hours),
            "best_bid": np.linspace(99, 99 + hours - 1, hours),
            "best_ask": np.linspace(101, 101 + hours - 1, hours),
            "bid_ask_spread": np.full(hours, 2.0),
            "depth_pct_1": np.linspace(5, 5 + hours - 1, hours),
            "depth_pct_5": np.linspace(10, 10 + hours - 1, hours),
            "order_imbalance": np.linspace(0.1, 0.1 + (hours - 1) * 0.01, hours),
            "fear_greed_score": np.linspace(40, 40 + hours - 1, hours),
            "confidence": np.linspace(0.3, 0.3 + (hours - 1) * 0.01, hours),
        }
    )
    return frame


def test_build_feature_matrix_scales_features(project_root):
    _write_features_config(project_root / "config")
    frame = _build_interim_frame(30)

    artifacts = build_feature_matrix("BTCUSD", frame)

    assert not artifacts.features.empty
    assert "return_log_4h" in artifacts.features.columns
    assert "vol_6h" in artifacts.features.columns
    assert (project_root / "data" / "processed" / "BTCUSD_features.parquet").exists()
    assert (project_root / "data" / "models" / "hmm" / "BTCUSD_scaler.pkl").exists()

    means = artifacts.features.mean().abs()
    assert (means < 1e-6).all()


def test_build_feature_matrix_requires_columns(project_root):
    _write_features_config(project_root / "config")
    frame = _build_interim_frame(5).drop(columns=["order_imbalance"])

    with pytest.raises(ValueError):
        build_feature_matrix("ETHUSD", frame, persist=False)
