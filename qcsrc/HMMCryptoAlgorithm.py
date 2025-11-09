"""QuantConnect algorithm orchestrating HMM-based crypto trading."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from qcsrc.custom_data import HMMStateProba, LiquidityBitAsk, MarketSentiment, SignalType
from qcsrc.io import ensure_directory, get_data_path, load_assets_config, load_settings_config
from qcsrc.models import align_predictions, compute_error_metrics
from qcsrc.models.thresholds import should_flat, should_long, should_short

try:  # pragma: no cover - only available inside QuantConnect
    from AlgorithmImports import (
        AccountType,
        BrokerageName,
        Market,
        QCAlgorithm,
        Resolution,
        Symbol,
        TradeBar,
        TradeBarConsolidator,
    )
except ImportError:  # pragma: no cover - local fallback for tests

    class QCAlgorithm:  # type: ignore[misc]
        """Minimal QCAlgorithm stand-in for offline tests."""

        def __init__(self) -> None:
            self.Time = None
            self.DebugMessages: list[str] = []
            self.TradeBuilder = _FakeTradeBuilder()

        def SetStartDate(self, *args, **kwargs) -> None:
            return None

        def SetEndDate(self, *args, **kwargs) -> None:
            return None

        def SetCash(self, *args, **kwargs) -> None:
            return None

        def SetBrokerageModel(self, *args, **kwargs) -> None:
            return None

        def SetWarmUp(self, *args, **kwargs) -> None:
            return None

        def Debug(self, message: str) -> None:
            self.DebugMessages.append(message)

        def AddCrypto(self, ticker: str, resolution, market: Optional[str] = None):
            return _FakeSecurity(_FakeSymbol(ticker))

        def AddData(self, data_type, ticker: str, resolution):
            return _FakeSymbol(f"{ticker}_{data_type.__name__}")

        def Liquidate(self, symbol):
            return None

        def SetHoldings(self, symbol, target: float) -> None:
            return None

        @property
        def Portfolio(self):  # pragma: no cover - compatibility shim
            return {}

    class _FakeSymbol:  # pragma: no cover - helper for tests
        def __init__(self, value: str) -> None:
            self.Value = value

    class _FakeSecurity:  # pragma: no cover - helper for tests
        def __init__(self, symbol: _FakeSymbol) -> None:
            self.Symbol = symbol

    class Resolution:  # pragma: no cover - helper for tests
        Minute = "Minute"
        Hour = "Hour"

    class Market:  # pragma: no cover - helper for tests
        GDAX = "GDAX"

    class BrokerageName:  # pragma: no cover - helper for tests
        BinanceUS = "BinanceUS"

    class AccountType:  # pragma: no cover - helper for tests
        Cash = "Cash"

    class TradeBar:  # pragma: no cover - helper for tests
        def __init__(self, symbol, close: float = 0.0, end_time=None):
            self.Symbol = symbol
            self.Close = close
            self.EndTime = end_time

    class TradeBarConsolidator:  # pragma: no cover - helper for tests
        def __init__(self, period_minutes: int) -> None:
            self.period_minutes = period_minutes
            self.DataConsolidated = []

        def Update(self, bar: TradeBar) -> None:
            for handler in list(self.DataConsolidated):
                handler(self, bar)

    Symbol = _FakeSymbol  # type: ignore

    class _FakeTrade:
        def __init__(self, profit_loss: float) -> None:
            self.ProfitLoss = profit_loss

    class _FakeTradeBuilder:
        def __init__(self) -> None:
            self.ClosedTrades: list[_FakeTrade] = []


@dataclass
class SymbolState:
    """Maintain the latest trading signal for an asset."""

    symbol: Symbol
    last_signal: SignalType = SignalType.FLAT


def determine_target(
    current_signal: SignalType,
    next_signal: SignalType,
    weight: float,
) -> Optional[float]:
    """Return the desired portfolio target for a signal transition."""

    if weight < 0:
        raise ValueError("target weight must be non-negative")

    if next_signal == current_signal:
        return None
    if next_signal == SignalType.LONG:
        return weight
    if next_signal == SignalType.SHORT:
        return -weight
    return 0.0


class HMMCryptoAlgorithm(QCAlgorithm):
    """Execute trades based on HMM posterior probabilities."""

    def Initialize(self) -> None:  # noqa: N802 - QC API signature
        self.Debug("Initialize: loading configuration")
        settings = load_settings_config()
        assets = load_assets_config()
        if not assets:
            raise ValueError("assets.yaml must define at least one asset")

        self._threshold = float(settings.get("threshold", 0.7))
        self._lookback_hours = int(settings.get("lookback_hours", 720))
        self._retrain_hours = int(settings.get("retrain_hours", 24))
        self._target_weight = 1.0 / max(len(assets), 1)

        self.SetStartDate(2021, 1, 1)
        self.SetCash(100000)
        try:  # pragma: no cover - brokerage not available in tests
            self.SetBrokerageModel(BrokerageName.BinanceUS, AccountType.Cash)
        except Exception:
            self.Debug("Initialize: falling back to default brokerage model")

        self.SetWarmUp(self._lookback_hours, Resolution.Hour)

        self._symbol_states: Dict[Symbol, SymbolState] = {}
        self._probability_map: Dict[Symbol, Symbol] = {}
        self._last_retrain_checkpoint = None
        self._probability_history: Dict[Symbol, list[Tuple[Optional[object], float]]] = defaultdict(list)
        self._hourly_last_close: Dict[Symbol, float] = {}
        self._realized_returns: Dict[Symbol, list[Tuple[Optional[object], float]]] = defaultdict(list)

        for metadata in assets.values():
            ticker = metadata.get("qc_ticker")
            if not ticker:
                raise ValueError("Each asset must include a qc_ticker entry")

            security = self.AddCrypto(ticker, Resolution.Minute, Market.GDAX)
            symbol = security.Symbol
            self._symbol_states[symbol] = SymbolState(symbol)

            probability_symbol = self.AddData(HMMStateProba, ticker, Resolution.Hour)
            self._probability_map[probability_symbol] = symbol

            self.AddData(LiquidityBitAsk, ticker, Resolution.Hour)
            self.AddData(MarketSentiment, ticker, Resolution.Hour)

            consolidator = TradeBarConsolidator(60)
            consolidator.DataConsolidated += self._on_hour_bar
            try:  # pragma: no cover - SubscriptionManager unavailable in tests
                self.SubscriptionManager.AddConsolidator(symbol, consolidator)
            except AttributeError:
                # Offline fallback to keep consolidator callable in tests.
                self._symbol_states[symbol].consolidator = consolidator  # type: ignore[attr-defined]

        self.Debug("Initialize: configuration complete")

    def _on_hour_bar(self, sender, bar: TradeBar) -> None:
        """Log receipt of hourly bars for debugging within QuantConnect."""

        symbol_value = getattr(getattr(bar, "Symbol", None), "Value", "")
        self.Debug(f"OnHourBar: consolidated bar for {symbol_value}")

        symbol = getattr(bar, "Symbol", None)
        close = getattr(bar, "Close", None)
        if symbol is None or close in (None, 0):
            return

        previous_close = self._hourly_last_close.get(symbol)
        self._hourly_last_close[symbol] = float(close)

        if previous_close in (None, 0):
            return

        timestamp = getattr(bar, "EndTime", None)
        if timestamp is None:
            timestamp = getattr(bar, "Time", None)

        try:
            realized = math.log(float(close) / float(previous_close))
        except (ValueError, ZeroDivisionError):
            return

        self._realized_returns[symbol].append((timestamp, realized))

    def OnData(self, slice_data) -> None:  # noqa: N802 - QC API signature
        if getattr(self, "IsWarmingUp", False):  # pragma: no cover - QC runtime only
            return

        probability_slice = getattr(slice_data, "Get", lambda _: {})
        probabilities = probability_slice(HMMStateProba)
        if not probabilities:
            return

        for proba_symbol, data in probabilities.items():
            asset_symbol = self._probability_map.get(proba_symbol)
            if asset_symbol is None:
                continue

            signal = self._resolve_signal(data)
            state = self._symbol_states.get(asset_symbol)
            if state is None:
                continue

            target = determine_target(state.last_signal, signal, self._target_weight)
            if target is None:
                self._record_probability(asset_symbol, data, signal)
                continue

            self.Debug(
                f"OnData: signal {signal.value} for {asset_symbol.Value} -> target {target}"
            )
            if target == 0.0:
                self.Liquidate(asset_symbol)
            else:
                self.SetHoldings(asset_symbol, target)

            state.last_signal = signal
            self._record_probability(asset_symbol, data, signal)

        self._checkpoint_retrain()

    def _resolve_signal(self, data: HMMStateProba) -> SignalType:
        """Derive the trading signal from the probability vector."""

        probabilities = (data.ProbBullish, data.ProbBearish, data.ProbConsolidation)
        if should_long(probabilities, self._threshold):
            return SignalType.LONG
        if should_short(probabilities, self._threshold):
            return SignalType.SHORT
        if should_flat(probabilities, self._threshold):
            return SignalType.FLAT
        return SignalType.FLAT

    def _checkpoint_retrain(self) -> None:
        """Log retraining checkpoints based on configured cadence."""

        current_time = getattr(self, "Time", None)
        if current_time is None:
            return

        if self._last_retrain_checkpoint is None:
            self._last_retrain_checkpoint = current_time
            return

        delta = current_time - self._last_retrain_checkpoint
        if delta >= timedelta(hours=self._retrain_hours):
            self.Debug("Checkpoint: retraining window reached")
            self._last_retrain_checkpoint = current_time

    def _record_probability(
        self, asset_symbol: Symbol, data: HMMStateProba, signal: SignalType
    ) -> None:
        """Track probability-implied expectations for later diagnostics."""

        timestamp = getattr(data, "EndTime", None)
        if timestamp is None:
            timestamp = getattr(data, "Time", None)
        expected_return = float(data.ProbBullish - data.ProbBearish)
        self._probability_history[asset_symbol].append((timestamp, expected_return))

    def _iter_trade_stats(self) -> Sequence[Tuple[int, int, int]]:
        """Return win/loss counts from the TradeBuilder when available."""

        builder = getattr(self, "TradeBuilder", None)
        trades = getattr(builder, "ClosedTrades", None)
        if not trades:
            return ()

        wins = 0
        losses = 0
        for trade in trades:
            profit_loss = getattr(trade, "ProfitLoss", 0.0)
            if profit_loss > 0:
                wins += 1
            elif profit_loss < 0:
                losses += 1
        return ((wins, losses, len(trades)),)

    def _persist_metrics(self, symbol: Symbol, payload: dict) -> None:
        """Persist metrics payload to diagnostics directory when possible."""

        try:
            diagnostics_dir = ensure_directory(get_data_path("models", "diagnostics"))
            symbol_name = getattr(symbol, "Value", str(symbol)).lower()
            target = Path(diagnostics_dir) / f"{symbol_name}_metrics.json"
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - filesystem may be unavailable
            self.Debug(f"OnEndOfAlgorithm: unable to persist metrics for {symbol}: {exc}")

    def OnEndOfAlgorithm(self) -> None:  # noqa: N802 - QC API signature
        """Summarize evaluation metrics at the end of a backtest/run."""

        for symbol, history in self._probability_history.items():
            realized_map = {timestamp: value for timestamp, value in self._realized_returns[symbol]}
            aligned: list[Tuple[float, float]] = []
            for timestamp, expected in history:
                actual = realized_map.get(timestamp)
                if actual is not None:
                    aligned.append((expected, actual))
            pairs = align_predictions(aligned)
            metrics = compute_error_metrics(pairs)

            symbol_name = getattr(symbol, "Value", str(symbol))
            self.Debug(
                "OnEndOfAlgorithm: %s count=%s mae=%.6f rmse=%.6f mape=%.6f dir=%.2f",
                symbol_name,
                metrics.count,
                metrics.mae,
                metrics.rmse,
                metrics.mape,
                metrics.direction_accuracy,
            )

            payload = {
                "symbol": symbol_name,
                "count": metrics.count,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "mape": metrics.mape,
                "direction_accuracy": metrics.direction_accuracy,
            }

            self._persist_metrics(symbol, payload)

        for wins, losses, total in self._iter_trade_stats():
            self.Debug(
                "OnEndOfAlgorithm: trades=%s wins=%s losses=%s win_rate=%.2f",
                total,
                wins,
                losses,
                wins / total if total else 0.0,
            )


__all__ = ["HMMCryptoAlgorithm", "determine_target", "SymbolState"]
