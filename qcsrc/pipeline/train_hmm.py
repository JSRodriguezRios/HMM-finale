"""Train Gaussian HMMs for configured assets and persist diagnostics."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from qcsrc.io import ensure_directory, get_data_path, load_settings_config
from qcsrc.models import GaussianHMMWrapper, TrainingConfig, map_states
from qcsrc.util import get_logger

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class TrainingArtifacts:
    symbol: str
    model_path: Path
    diagnostics_path: Path
    metrics: Dict[str, float]
    state_mapping: Dict[int, str]


def _load_feature_frame(symbol: str, *, lookback_hours: int) -> pd.DataFrame:
    path = Path(get_data_path("processed")) / f"{symbol}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed feature matrix missing for {symbol} at {path}")

    frame = pd.read_parquet(path)
    if "timestamp" not in frame.columns:
        raise ValueError("feature parquet must include a timestamp column")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame.sort_values("timestamp", inplace=True)
    if lookback_hours > 0:
        frame = frame.tail(lookback_hours)
    frame.set_index("timestamp", inplace=True)
    return frame


def _load_scaler(symbol: str) -> StandardScaler:
    path = Path(get_data_path("models", "hmm")) / f"{symbol}_scaler.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Scaler artifact missing for {symbol} at {path}")

    with path.open("rb") as handle:
        scaler = pickle.load(handle)
    if not isinstance(scaler, StandardScaler):
        raise TypeError("Loaded scaler artifact must be a sklearn StandardScaler")
    return scaler


def _compute_metrics(
    posterior: np.ndarray,
    *,
    model: GaussianHMMWrapper,
    scaler: StandardScaler,
    feature_columns: list[str],
    frame: pd.DataFrame,
) -> Dict[str, float]:
    return_idx = feature_columns.index("return_log_1h")
    restored_means = model.model.means_ * scaler.scale_ + scaler.mean_
    state_means = restored_means[:, return_idx]

    actual_features = scaler.inverse_transform(frame.values)
    actual_returns = actual_features[:, return_idx]

    predicted_returns = posterior @ state_means
    if len(predicted_returns) <= 1:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}

    predicted_next = predicted_returns[:-1]
    actual_next = actual_returns[1:]

    mae = float(np.mean(np.abs(predicted_next - actual_next)))
    rmse = float(np.sqrt(np.mean((predicted_next - actual_next) ** 2)))

    denominator = np.where(actual_next == 0.0, 1e-8, np.abs(actual_next))
    mape = float(np.mean(np.abs((predicted_next - actual_next) / denominator)))

    return {"mae": mae, "rmse": rmse, "mape": mape}


def _count_parameters(hmm: GaussianHMMWrapper) -> int:
    model = hmm.model
    n_components = model.n_components
    n_features = model.n_features

    start_params = n_components - 1
    transition_params = n_components * (n_components - 1)
    mean_params = n_components * n_features

    covariance_type = model.covariance_type
    if covariance_type == "full":
        cov_params = int(n_components * (n_features * (n_features + 1) / 2))
    elif covariance_type == "diag":
        cov_params = n_components * n_features
    elif covariance_type == "spherical":
        cov_params = n_components
    elif covariance_type == "tied":
        cov_params = int(n_features * (n_features + 1) / 2)
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")

    return start_params + transition_params + mean_params + cov_params


def train_hmm_for_symbol(
    symbol: str,
    *,
    lookback_hours: Optional[int] = None,
    min_samples: Optional[int] = None,
    model_dir: Optional[Path] = None,
    diagnostics_dir: Optional[Path] = None,
) -> TrainingArtifacts:
    """Train a Gaussian HMM for ``symbol`` and persist model artifacts."""

    settings = load_settings_config()
    lookback = lookback_hours or int(settings.get("lookback_hours", 720))
    min_required = min_samples or int(settings.get("min_samples", 200))
    n_components = int(settings.get("max_states", 3))

    frame = _load_feature_frame(symbol, lookback_hours=lookback)
    if len(frame) < min_required:
        raise ValueError(
            f"Not enough samples to train HMM for {symbol}: "
            f"{len(frame)} available, require {min_required}"
        )

    scaler = _load_scaler(symbol)
    observations = frame.values

    config = TrainingConfig(n_components=n_components)
    wrapper = GaussianHMMWrapper(config)
    wrapper.fit(observations)

    posterior = wrapper.predict_proba(observations)
    log_likelihood = wrapper.score(observations)

    n_params = _count_parameters(wrapper)
    n_samples = observations.shape[0]
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_samples) - 2 * log_likelihood

    feature_columns = list(frame.columns)
    state_mapping = map_states(wrapper.model, feature_columns, scaler=scaler)

    metrics = _compute_metrics(
        posterior,
        model=wrapper,
        scaler=scaler,
        feature_columns=feature_columns,
        frame=frame,
    )

    target_model_dir = ensure_directory(model_dir or Path(get_data_path("models", "hmm")))
    model_path = target_model_dir / f"{symbol}_hmm.pkl"
    wrapper.save(model_path)

    target_diag_dir = ensure_directory(
        diagnostics_dir or Path(get_data_path("models", "diagnostics"))
    )
    diagnostics_path = target_diag_dir / f"{symbol}_report.json"

    diagnostics_payload = {
        "symbol": symbol,
        "samples": n_samples,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "bic": bic,
        "state_mapping": {str(state): label for state, label in state_mapping.items()},
        "metrics": metrics,
        "timestamps": [ts.isoformat() for ts in frame.index],
        "posterior_probabilities": posterior.tolist(),
    }

    with diagnostics_path.open("w", encoding="utf-8") as handle:
        json.dump(diagnostics_payload, handle, indent=2)

    _LOGGER.info("Trained HMM for %s with %d samples", symbol, n_samples)

    return TrainingArtifacts(
        symbol=symbol,
        model_path=model_path,
        diagnostics_path=diagnostics_path,
        metrics=metrics,
        state_mapping=state_mapping,
    )


__all__ = ["TrainingArtifacts", "train_hmm_for_symbol"]
