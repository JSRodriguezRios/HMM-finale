"""Hidden Markov Model utilities for training and inference."""

from .hmm_model import GaussianHMMWrapper, TrainingConfig
from .metrics import ErrorMetrics, align_predictions, compute_error_metrics
from .state_mapper import map_states
from .thresholds import should_flat, should_long, should_short

__all__ = [
    "GaussianHMMWrapper",
    "TrainingConfig",
    "ErrorMetrics",
    "align_predictions",
    "compute_error_metrics",
    "map_states",
    "should_flat",
    "should_long",
    "should_short",
]
