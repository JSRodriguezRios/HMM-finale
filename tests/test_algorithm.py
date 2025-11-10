from __future__ import annotations

import pytest

from qcsrc.HMMCryptoAlgorithm import determine_target
from qcsrc.custom_data import SignalType


@pytest.mark.parametrize(
    "current_signal,next_signal,expected",
    [
        (SignalType.FLAT, SignalType.LONG, 0.5),
        (SignalType.FLAT, SignalType.SHORT, -0.5),
        (SignalType.LONG, SignalType.FLAT, 0.0),
        (SignalType.SHORT, SignalType.FLAT, 0.0),
    ],
)
def test_determine_target_transitions(current_signal, next_signal, expected):
    assert determine_target(current_signal, next_signal, 0.5) == pytest.approx(expected)


def test_determine_target_no_change():
    assert determine_target(SignalType.LONG, SignalType.LONG, 0.4) is None
    assert determine_target(SignalType.SHORT, SignalType.SHORT, 0.4) is None
    assert determine_target(SignalType.FLAT, SignalType.FLAT, 0.4) is None


def test_determine_target_negative_weight():
    with pytest.raises(ValueError):
        determine_target(SignalType.FLAT, SignalType.LONG, -0.1)
