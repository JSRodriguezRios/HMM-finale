from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from qcsrc.custom_data import HMMStateProba, LiquidityBitAsk, MarketSentiment, SignalType


class _StubSymbol:
    def __init__(self, value: str) -> None:
        self.Value = value


class _StubConfig:
    def __init__(self, value: str) -> None:
        self.Symbol = _StubSymbol(value)


@pytest.mark.parametrize(
    "cls, line, expected_attrs",
    [
        (
            LiquidityBitAsk,
            "2023-01-01 00:00:00,1.5,150.0,250.0,0.25",
            {
                "BidAskSpread": 1.5,
                "DepthPct1": 150.0,
                "DepthPct5": 250.0,
                "OrderImbalance": 0.25,
                "Value": 1.5,
            },
        ),
        (
            MarketSentiment,
            "2023-01-01 01:00:00,45.0,0.8",
            {
                "FearGreedScore": 45.0,
                "Confidence": 0.8,
                "Value": 45.0,
            },
        ),
    ],
)
def test_custom_data_reader_parses_rows(cls, line, expected_attrs):
    config = _StubConfig("BTCUSD")
    data = cls()

    # header rows are ignored
    assert data.Reader(config, "time," + ",".join(attr for attr in expected_attrs if attr != "Value"), None, False) is None

    parsed = data.Reader(config, line, None, False)
    assert parsed is not None
    assert parsed.Symbol is config.Symbol

    expected_time = datetime.strptime(line.split(",", 1)[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    assert parsed.Time == expected_time
    assert parsed.EndTime == expected_time + timedelta(hours=1)

    for attr, expected in expected_attrs.items():
        assert getattr(parsed, attr) == pytest.approx(expected)


def test_get_source_returns_expected_path(tmp_path, monkeypatch):
    config = _StubConfig("BTCUSD")
    instance = LiquidityBitAsk()

    monkeypatch.setattr(
        "qcsrc.custom_data._base.get_qc_data_path",
        lambda *parts: tmp_path.joinpath(*parts),
    )

    source = instance.GetSource(config, None, False)
    assert isinstance(source, str)
    assert source.endswith("LiquidityBitAsk/btcusd.csv")


def test_hmm_state_proba_signal_mapping():
    config = _StubConfig("BTCUSD")
    instance = HMMStateProba()

    bullish_line = "2023-01-01 02:00:00,0.75,0.1,0.15"
    bullish = instance.Reader(config, bullish_line, None, False)
    assert bullish is not None
    assert bullish.GetSignal() == SignalType.LONG

    bearish_line = "2023-01-01 03:00:00,0.1,0.8,0.1"
    bearish = instance.Reader(config, bearish_line, None, False)
    assert bearish is not None
    assert bearish.GetSignal() == SignalType.SHORT

    flat_line = "2023-01-01 04:00:00,0.2,0.1,0.75"
    flat = instance.Reader(config, flat_line, None, False)
    assert flat is not None
    assert flat.GetSignal() == SignalType.FLAT

    neutral_line = "2023-01-01 05:00:00,0.4,0.3,0.3"
    neutral = instance.Reader(config, neutral_line, None, False)
    assert neutral is not None
    assert neutral.GetSignal(0.8) == SignalType.FLAT
