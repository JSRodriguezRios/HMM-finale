"""Generate posterior probability time series for trained HMM models."""

from __future__ import annotations

import datetime as dt
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from qcsrc.io import get_data_path, load_state_map
from qcsrc.io.custom_export import export_probability_series
from qcsrc.models import GaussianHMMWrapper, map_states
from qcsrc.util import ensure_utc, get_logger

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ProbabilityArtifacts:
    """Artifacts produced during probability inference."""

    symbol: str
    probabilities: pd.DataFrame
    export_path: Path


def _load_processed_features(symbol: str, *, processed_dir: Optional[Path] = None) -> pd.DataFrame:
    base = Path(processed_dir or get_data_path("processed"))
    path = base / f"{symbol}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed feature matrix missing for {symbol} at {path}"
        )

    frame = pd.read_parquet(path)
    if "timestamp" not in frame.columns:
        raise ValueError("feature parquet must include a timestamp column")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame.sort_values("timestamp", inplace=True)
    frame.set_index("timestamp", inplace=True)
    frame.index.name = "timestamp"
    return frame


def _load_scaler(symbol: str, *, model_dir: Optional[Path] = None) -> StandardScaler:
    base = Path(model_dir or get_data_path("models", "hmm"))
    path = base / f"{symbol}_scaler.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Scaler artifact missing for {symbol} at {path}")

    with path.open("rb") as handle:
        scaler = pickle.load(handle)
    if not isinstance(scaler, StandardScaler):
        raise TypeError("Loaded scaler artifact must be a sklearn StandardScaler")
    return scaler


def _load_model(symbol: str, *, model_dir: Optional[Path] = None) -> GaussianHMMWrapper:
    base = Path(model_dir or get_data_path("models", "hmm"))
    path = base / f"{symbol}_hmm.pkl"
    if not path.exists():
        raise FileNotFoundError(f"HMM model missing for {symbol} at {path}")
    return GaussianHMMWrapper.load(path)


def _aggregate_probabilities(
    posterior: np.ndarray,
    mapping: Dict[int, str],
    *,
    index: pd.DatetimeIndex,
    state_map_override: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    labels = ["bullish", "bearish", "consolidation"]
    if state_map_override:
        labels = sorted(state_map_override.keys(), key=state_map_override.get)

    data = {f"prob_{label}": np.zeros(posterior.shape[0]) for label in labels}
    for state, label in mapping.items():
        key = f"prob_{label}"
        if key not in data:
            data[key] = np.zeros(posterior.shape[0])
        data[key] += posterior[:, state]

    frame = pd.DataFrame(data, index=index)
    frame.index.name = "timestamp"
    return frame


def infer_probabilities_for_symbol(
    symbol: str,
    *,
    start: Optional[dt.datetime] = None,
    end: Optional[dt.datetime] = None,
    processed_dir: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> ProbabilityArtifacts:
    """Run posterior inference for ``symbol`` and export QuantConnect CSV."""

    frame = _load_processed_features(symbol, processed_dir=processed_dir)

    if start is not None:
        frame = frame.loc[frame.index >= ensure_utc(start)]
    if end is not None:
        frame = frame.loc[frame.index <= ensure_utc(end)]

    if frame.empty:
        raise ValueError("No feature rows available for probability inference")

    scaler = _load_scaler(symbol, model_dir=model_dir)
    wrapper = _load_model(symbol, model_dir=model_dir)

    posterior = wrapper.predict_proba(frame.values)

    mapping = map_states(wrapper.model, list(frame.columns), scaler=scaler)
    state_map_override = load_state_map()
    probabilities = _aggregate_probabilities(
        posterior,
        mapping,
        index=frame.index,
        state_map_override=state_map_override,
    )

    export_path = export_probability_series(
        symbol,
        probabilities,
        output_dir=output_dir,
    )
    _LOGGER.info(
        "Inferred probabilities for %s covering %d observations", symbol, len(probabilities)
    )

    return ProbabilityArtifacts(symbol=symbol, probabilities=probabilities, export_path=export_path)


__all__ = ["ProbabilityArtifacts", "infer_probabilities_for_symbol"]
