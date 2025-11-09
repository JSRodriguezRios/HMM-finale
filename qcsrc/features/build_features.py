"""Build standardized feature matrices for the HMM pipeline."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from qcsrc.features.feature_defs import FeatureConfig, load_feature_config
from qcsrc.io import ensure_directory, get_data_path
from qcsrc.util import ensure_utc_index, get_logger

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class FeatureArtifacts:
    """Container for feature matrix outputs."""

    features: pd.DataFrame
    scaler: StandardScaler


def _log_return(series: pd.Series, window: int) -> pd.Series:
    shifted = series.shift(window)
    return np.log(series / shifted)


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(2, window // 2)).std()


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=max(2, window // 2)).mean()
    rolling_std = series.rolling(window=window, min_periods=max(2, window // 2)).std()
    zscore = (series - rolling_mean) / rolling_std
    return zscore.replace({np.inf: np.nan, -np.inf: np.nan}).fillna(0.0)


def _apply_lags(series: pd.Series, lags: list[int], prefix: str) -> pd.DataFrame:
    payload = {}
    for lag in lags:
        payload[f"{prefix}_lag_{lag}h"] = series.shift(lag)
    return pd.DataFrame(payload)


def _validate_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("interim frame must not be empty")
    required_columns = {
        "timestamp",
        "close",
        "volume",
        "best_bid",
        "best_ask",
        "bid_ask_spread",
        "order_imbalance",
        "fear_greed_score",
        "confidence",
    }

    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"interim frame missing required columns: {sorted(missing)}")

    return frame.copy()


def _prepare_index(frame: pd.DataFrame) -> pd.DataFrame:
    frame["timestamp"] = ensure_utc_index(frame["timestamp"])
    frame.sort_values("timestamp", inplace=True)
    frame.set_index("timestamp", inplace=True)
    return frame


def _build_feature_frame(frame: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    hourly_close = frame["close"].astype(float)
    hourly_returns = _log_return(hourly_close, 1)

    features = pd.DataFrame(index=frame.index)
    features["return_log_1h"] = hourly_returns

    for window in config.returns.windows:
        if window == 1:
            continue
        features[f"return_log_{window}h"] = _log_return(hourly_close, window)

    for window in config.volatility.windows:
        features[f"vol_{window}h"] = _rolling_std(hourly_returns, window)

    volume_z = _rolling_zscore(frame["volume"].astype(float), config.liquidity.zscore_window)
    features["volume_zscore"] = volume_z

    for level in config.liquidity.depth_pct_levels:
        column = f"depth_pct_{level}"
        if column not in frame:
            continue
        zscore = _rolling_zscore(frame[column].astype(float), config.liquidity.zscore_window)
        features[f"liquidity_depth_pct_{level}_z"] = zscore

    order_imbalance_z = _rolling_zscore(
        frame["order_imbalance"].astype(float), config.liquidity.zscore_window
    )
    features["liquidity_order_imbalance_z"] = order_imbalance_z

    if config.liquidity.include_bid_ask_spread and "bid_ask_spread" in frame:
        spread_z = _rolling_zscore(
            frame["bid_ask_spread"].astype(float), config.liquidity.zscore_window
        )
        features["liquidity_bid_ask_spread_z"] = spread_z

    fg_series = frame["fear_greed_score"].astype(float)
    conf_series = frame["confidence"].astype(float)

    features["sentiment_fear_greed_z"] = _rolling_zscore(
        fg_series, config.sentiment.zscore_window
    )
    features["sentiment_confidence_z"] = _rolling_zscore(
        conf_series, config.sentiment.zscore_window
    )

    fg_lags = _apply_lags(fg_series, config.sentiment.lags, "sentiment_fear_greed")
    conf_lags = _apply_lags(conf_series, config.sentiment.lags, "sentiment_confidence")

    features = pd.concat([features, fg_lags, conf_lags], axis=1)

    return features


def _scale_features(
    features: pd.DataFrame, scaler: Optional[StandardScaler], fit_scaler: bool
) -> Tuple[pd.DataFrame, StandardScaler]:
    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        scaler = scaler.fit(features.values)
        transformed = scaler.transform(features.values)
    else:
        if not hasattr(scaler, "transform"):
            raise ValueError("Provided scaler must implement transform")
        transformed = scaler.transform(features.values)

    scaled = pd.DataFrame(transformed, index=features.index, columns=features.columns)
    return scaled, scaler


def _persist_outputs(
    symbol: str,
    scaled_features: pd.DataFrame,
    scaler: StandardScaler,
    *,
    feature_dir: Optional[Path],
    model_dir: Optional[Path],
) -> None:
    target_feature_dir = ensure_directory(feature_dir or get_data_path("processed"))
    feature_path = Path(target_feature_dir) / f"{symbol}_features.parquet"
    scaled_features.reset_index().to_parquet(feature_path, index=False)
    _LOGGER.info("Persisted features for %s to %s", symbol, feature_path)

    target_model_dir = ensure_directory(model_dir or get_data_path("models", "hmm"))
    scaler_path = Path(target_model_dir) / f"{symbol}_scaler.pkl"
    with scaler_path.open("wb") as handle:
        pickle.dump(scaler, handle)
    _LOGGER.info("Persisted scaler for %s to %s", symbol, scaler_path)


def build_feature_matrix(
    symbol: str,
    interim_frame: pd.DataFrame,
    *,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True,
    persist: bool = True,
    feature_dir: Optional[Path] = None,
    model_dir: Optional[Path] = None,
) -> FeatureArtifacts:
    """Transform ``interim_frame`` into a scaled feature matrix for ``symbol``."""

    interim_copy = _prepare_index(_validate_inputs(interim_frame))
    config = load_feature_config()

    raw_features = _build_feature_frame(interim_copy, config)
    raw_features.dropna(inplace=True)

    if raw_features.empty:
        raise ValueError("Feature matrix is empty after preprocessing")

    scaled_features, fitted_scaler = _scale_features(
        raw_features, scaler=scaler, fit_scaler=fit_scaler
    )

    if persist:
        _persist_outputs(
            symbol,
            scaled_features,
            fitted_scaler,
            feature_dir=feature_dir,
            model_dir=model_dir,
        )

    return FeatureArtifacts(features=scaled_features, scaler=fitted_scaler)


__all__ = ["FeatureArtifacts", "build_feature_matrix"]
