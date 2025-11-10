"""Signal threshold helpers for HMM regime probabilities."""

from __future__ import annotations

import numpy as np


def _ensure_vector(probabilities: np.ndarray) -> np.ndarray:
    vector = np.asarray(probabilities, dtype=float)
    if vector.ndim != 1:
        raise ValueError("probabilities must be a 1D vector")
    return vector


def should_long(probabilities: np.ndarray, threshold: float) -> bool:
    vector = _ensure_vector(probabilities)
    return float(vector[0]) >= threshold


def should_short(probabilities: np.ndarray, threshold: float) -> bool:
    vector = _ensure_vector(probabilities)
    if vector.size < 2:
        return False
    return float(vector[1]) >= threshold


def should_flat(probabilities: np.ndarray, threshold: float) -> bool:
    vector = _ensure_vector(probabilities)
    if vector.size < 3:
        return False
    return float(vector[2]) >= threshold


__all__ = ["should_flat", "should_long", "should_short"]
