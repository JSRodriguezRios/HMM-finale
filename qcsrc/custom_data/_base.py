"""Shared helpers for QuantConnect custom data classes."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Dict, Optional, Sequence

from qcsrc.io.file_locator import get_qc_data_path

try:  # pragma: no cover - only exercised inside QuantConnect
    from QuantConnect.Data import SubscriptionDataSource
    from QuantConnect.Data.Custom import PythonData
    from QuantConnect import SubscriptionTransportMedium
except ImportError:  # pragma: no cover - fallback for local tests
    SubscriptionDataSource = None  # type: ignore

    class SubscriptionTransportMedium:  # type: ignore
        """Minimal shim matching QuantConnect's transport medium enum."""

        LocalFile = "LocalFile"

    class PythonData:  # type: ignore
        """Lightweight stand-in for QuantConnect's ``PythonData`` base class."""

        def __init__(self) -> None:
            self.Symbol = None
            self.Time: Optional[datetime] = None
            self.EndTime: Optional[datetime] = None
            self.Value: float = 0.0


_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class CustomDataColumn:
    """Mapping between a CSV column and a model attribute."""

    column_name: str
    attribute_name: str


def _build_subscription_source(path: Path):
    """Return a SubscriptionDataSource or plain path for QC ingestion."""

    if SubscriptionDataSource is None:
        return str(path)
    return SubscriptionDataSource(str(path), SubscriptionTransportMedium.LocalFile)


def _parse_csv_line(line: str, headers: Sequence[str]) -> Optional[Dict[str, str]]:
    """Parse a CSV ``line`` into a dictionary keyed by ``headers``."""

    if not line:
        return None

    # Skip header rows produced by QuantConnect export helpers.
    if line.strip().startswith("time"):
        return None

    reader = csv.DictReader(StringIO(line), fieldnames=headers)
    try:
        values = next(reader)
    except StopIteration:  # pragma: no cover - defensive
        return None

    if values is None:
        return None

    return {key: (value or "") for key, value in values.items()}


class BaseCustomSeries(PythonData):
    """Base helper implementing QuantConnect custom-data parsing contract."""

    csv_columns: Sequence[CustomDataColumn] = ()
    csv_headers: Sequence[str] = ()
    subdirectory: Sequence[str] = ()
    primary_value_column: str = ""
    period: timedelta = timedelta(hours=1)

    def __init__(self) -> None:
        super().__init__()
        for column in self.csv_columns:
            setattr(self, column.attribute_name, 0.0)

    def GetSource(self, config, date, isLiveMode: bool):  # noqa: N802 - QC naming
        symbol = getattr(getattr(config, "Symbol", None), "Value", "")
        if not symbol:
            raise ValueError("configuration missing symbol value")

        path = get_qc_data_path("custom", *self.subdirectory, f"{symbol.lower()}.csv")
        return _build_subscription_source(path)

    def Reader(self, config, line: str, date, isLiveMode: bool):  # noqa: N802 - QC naming
        parsed = _parse_csv_line(line, self.csv_headers)
        if not parsed:
            return None

        timestamp_raw = parsed.get("time")
        if not timestamp_raw:
            return None

        try:
            timestamp = datetime.strptime(timestamp_raw, _TIME_FORMAT).replace(tzinfo=timezone.utc)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"invalid timestamp '{timestamp_raw}' for custom data") from exc

        instance = self.__class__()
        instance.Symbol = getattr(config, "Symbol", None)
        instance.Time = timestamp
        instance.EndTime = timestamp + self.period

        for column in self.csv_columns:
            raw_value = parsed.get(column.column_name, "")
            instance_value = float(raw_value) if raw_value not in ("", None) else 0.0
            setattr(instance, column.attribute_name, instance_value)

        if self.primary_value_column:
            instance.Value = getattr(instance, self.primary_value_column)

        return instance


__all__ = [
    "BaseCustomSeries",
    "CustomDataColumn",
]
