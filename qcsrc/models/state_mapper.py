"""Map trained HMM states to semantic regime labels."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


def _restore_means(
    means: np.ndarray,
    *,
    scaler: StandardScaler | None,
) -> np.ndarray:
    if scaler is None:
        return means
    return means * scaler.scale_ + scaler.mean_


def map_states(
    model: GaussianHMM,
    feature_columns: Sequence[str],
    *,
    scaler: StandardScaler | None = None,
) -> Dict[int, str]:
    """Return a mapping from hidden state index to semantic label."""

    if "return_log_1h" not in feature_columns:
        raise ValueError("feature matrix must include 'return_log_1h' column")

    restored_means = _restore_means(model.means_, scaler=scaler)
    return_idx = feature_columns.index("return_log_1h")
    ordered = np.argsort(restored_means[:, return_idx])

    mapping: Dict[int, str] = {}
    mapping[int(ordered[-1])] = "bullish"
    mapping[int(ordered[0])] = "bearish"

    for state in ordered[1:-1]:
        mapping[int(state)] = "consolidation"

    if len(mapping) != model.n_components:
        # In the two-state edge case mark the remaining state as consolidation.
        for state in range(model.n_components):
            mapping.setdefault(state, "consolidation")

    return mapping


__all__ = ["map_states"]
