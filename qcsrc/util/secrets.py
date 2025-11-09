"""Credential loading helpers for offline pipelines and QC runtime."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SECRETS_PATH = _PROJECT_ROOT / "secrets" / "credentials.json"

_ENV_KEY_MAP = {
    "cryptoquant": "CRYPTOQUANT_API_KEY",
    "coinstats": "COINSTATS_API_KEY",
    "binance_api_key": "BINANCE_API_KEY",
    "binance_api_secret": "BINANCE_API_SECRET",
}


class MissingCredentialError(KeyError):
    """Raised when a required credential could not be resolved."""


@dataclass(frozen=True)
class CredentialSet:
    """Container for third-party API credentials."""

    cryptoquant_api_key: str
    coinstats_api_key: str
    binance_api_key: str
    binance_api_secret: str

    def as_dict(self) -> Dict[str, str]:
        """Represent the credential set as a plain dictionary."""

        return {
            "CRYPTOQUANT_API_KEY": self.cryptoquant_api_key,
            "COINSTATS_API_KEY": self.coinstats_api_key,
            "BINANCE_API_KEY": self.binance_api_key,
            "BINANCE_API_SECRET": self.binance_api_secret,
        }


def _read_secrets_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    cryptoquant = payload.get("cryptoquant", {})
    coinstats = payload.get("coinstats", {})
    binance = payload.get("binance", {})

    secrets: Dict[str, str] = {}
    if cryptoquant_key := cryptoquant.get("api_key"):
        secrets["CRYPTOQUANT_API_KEY"] = cryptoquant_key
    if coinstats_key := coinstats.get("api_key"):
        secrets["COINSTATS_API_KEY"] = coinstats_key
    if binance_key := binance.get("api_key"):
        secrets["BINANCE_API_KEY"] = binance_key
    if binance_secret := binance.get("api_secret"):
        secrets["BINANCE_API_SECRET"] = binance_secret
    return secrets


def _merge_sources(
    secrets_path: Path,
    required_keys: Iterable[str],
) -> Dict[str, str]:
    merged = _read_secrets_file(secrets_path)

    for key in required_keys:
        env_value = os.getenv(key)
        if env_value:
            merged[key] = env_value

    return merged


def load_credentials(
    secrets_path: Optional[Path] = None,
    required_keys: Optional[Iterable[str]] = None,
) -> CredentialSet:
    """Load credentials from environment variables or a secrets JSON file.

    Parameters
    ----------
    secrets_path:
        Optional path to a ``credentials.json`` secrets file. Defaults to the
        repository ``secrets/credentials.json`` location.
    required_keys:
        Iterable of required environment variable names. Defaults to the keys
        derived from :class:`CredentialSet` when omitted.
    """

    keys = tuple(required_keys or _ENV_KEY_MAP.values())
    path = secrets_path or _DEFAULT_SECRETS_PATH
    merged = _merge_sources(path, keys)

    missing = [key for key in keys if not merged.get(key)]
    if missing:
        raise MissingCredentialError(
            "Missing required credentials: " + ", ".join(sorted(missing))
        )

    return CredentialSet(
        cryptoquant_api_key=merged["CRYPTOQUANT_API_KEY"],
        coinstats_api_key=merged["COINSTATS_API_KEY"],
        binance_api_key=merged["BINANCE_API_KEY"],
        binance_api_secret=merged["BINANCE_API_SECRET"],
    )


__all__ = ["CredentialSet", "MissingCredentialError", "load_credentials"]
