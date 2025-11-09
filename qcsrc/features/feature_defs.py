"""Load and expose structured feature configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from qcsrc.io import get_config_path, load_yaml_config


@dataclass(frozen=True)
class ReturnsSpec:
    """Configuration for log-return features."""

    windows: List[int]


@dataclass(frozen=True)
class VolatilitySpec:
    """Configuration for rolling volatility features."""

    windows: List[int]


@dataclass(frozen=True)
class LiquiditySpec:
    """Configuration for liquidity-derived features."""

    depth_pct_levels: List[int]
    include_bid_ask_spread: bool
    zscore_window: int


@dataclass(frozen=True)
class SentimentSpec:
    """Configuration for sentiment-derived features."""

    lags: List[int]
    zscore_window: int


@dataclass(frozen=True)
class FeatureConfig:
    """Aggregate configuration for all feature groups."""

    returns: ReturnsSpec
    volatility: VolatilitySpec
    liquidity: LiquiditySpec
    sentiment: SentimentSpec


def _coerce_positive(values: List[int], fallback: List[int]) -> List[int]:
    payload = [value for value in values if value > 0]
    return payload or fallback


def load_feature_config() -> FeatureConfig:
    """Load feature configuration from ``config/features.yaml``."""

    payload = load_yaml_config(get_config_path("features.yaml"))

    returns_raw = payload.get("returns", {})
    volatility_raw = payload.get("volatility", {})
    liquidity_raw = payload.get("liquidity", {})
    sentiment_raw = payload.get("sentiment", {})

    returns_spec = ReturnsSpec(
        windows=_coerce_positive(list(returns_raw.get("windows", [1])), [1])
    )

    volatility_spec = VolatilitySpec(
        windows=_coerce_positive(list(volatility_raw.get("windows", [24])), [24])
    )

    liquidity_spec = LiquiditySpec(
        depth_pct_levels=_coerce_positive(
            list(liquidity_raw.get("depth_pct_levels", [1, 5])), [1]
        ),
        include_bid_ask_spread=bool(liquidity_raw.get("include_bid_ask_spread", True)),
        zscore_window=int(liquidity_raw.get("zscore_window", sentiment_raw.get("zscore_window", 24))),
    )

    sentiment_spec = SentimentSpec(
        lags=_coerce_positive(list(sentiment_raw.get("lags", [1])), [1]),
        zscore_window=int(sentiment_raw.get("zscore_window", 24)),
    )

    return FeatureConfig(
        returns=returns_spec,
        volatility=volatility_spec,
        liquidity=liquidity_spec,
        sentiment=sentiment_spec,
    )


__all__ = [
    "FeatureConfig",
    "LiquiditySpec",
    "ReturnsSpec",
    "SentimentSpec",
    "VolatilitySpec",
    "load_feature_config",
]
