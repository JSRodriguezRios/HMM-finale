"""Gaussian Hidden Markov Model wrapper utilities."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from hmmlearn.hmm import GaussianHMM


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration values required for HMM training."""

    n_components: int
    n_iter: int = 100
    covariance_type: str = "full"
    random_state: int = 42


class GaussianHMMWrapper:
    """Thin wrapper around :class:`hmmlearn.hmm.GaussianHMM`."""

    def __init__(self, config: TrainingConfig):
        self._config = config
        self._model: Optional[GaussianHMM] = None

    @property
    def model(self) -> GaussianHMM:
        if self._model is None:
            raise RuntimeError("HMM model has not been fitted")
        return self._model

    def fit(self, observations: np.ndarray) -> None:
        if observations.ndim != 2:
            raise ValueError("observations must be a 2D array")
        if len(observations) == 0:
            raise ValueError("observations array must not be empty")

        hmm = GaussianHMM(
            n_components=self._config.n_components,
            covariance_type=self._config.covariance_type,
            n_iter=self._config.n_iter,
            random_state=self._config.random_state,
        )
        hmm.fit(observations)
        self._model = hmm

    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        if observations.ndim != 2:
            raise ValueError("observations must be a 2D array")
        return self.model.predict_proba(observations)

    def score(self, observations: np.ndarray) -> float:
        return float(self.model.score(observations))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"config": self._config, "model": self.model}, handle)

    @classmethod
    def load(cls, path: Path) -> "GaussianHMMWrapper":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        wrapper = cls(payload["config"])
        wrapper._model = payload["model"]
        return wrapper


__all__ = ["GaussianHMMWrapper", "TrainingConfig"]
