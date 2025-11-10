import json
from pathlib import Path

import pytest

from qcsrc.util.secrets import (
    CredentialSet,
    MissingCredentialError,
    load_credentials,
)


def _write_secrets(path: Path, cryptoquant: str = "cq", coinstats: str = "cs", binance_key: str = "bk", binance_secret: str = "bs") -> None:
    payload = {
        "cryptoquant": {"api_key": cryptoquant},
        "coinstats": {"api_key": coinstats},
        "binance": {"api_key": binance_key, "api_secret": binance_secret},
    }
    path.write_text(json.dumps(payload))


def test_load_credentials_from_file(tmp_path, monkeypatch):
    secrets_path = tmp_path / "credentials.json"
    _write_secrets(secrets_path, "file_cq", "file_cs", "file_bk", "file_bs")

    creds = load_credentials(secrets_path=secrets_path)

    assert isinstance(creds, CredentialSet)
    assert creds.cryptoquant_api_key == "file_cq"
    assert creds.coinstats_api_key == "file_cs"
    assert creds.binance_api_key == "file_bk"
    assert creds.binance_api_secret == "file_bs"


def test_load_credentials_prefers_environment(monkeypatch, tmp_path):
    secrets_path = tmp_path / "credentials.json"
    _write_secrets(secrets_path, "file_cq", "file_cs", "file_bk", "file_bs")

    monkeypatch.setenv("CRYPTOQUANT_API_KEY", "env_cq")
    monkeypatch.setenv("COINSTATS_API_KEY", "env_cs")
    monkeypatch.setenv("BINANCE_API_KEY", "env_bk")
    monkeypatch.setenv("BINANCE_API_SECRET", "env_bs")

    creds = load_credentials(secrets_path=secrets_path)

    assert creds.cryptoquant_api_key == "env_cq"
    assert creds.coinstats_api_key == "env_cs"
    assert creds.binance_api_key == "env_bk"
    assert creds.binance_api_secret == "env_bs"


def test_missing_credentials_raise(tmp_path, monkeypatch):
    secrets_path = tmp_path / "credentials.json"
    secrets_path.write_text("{}")

    with pytest.raises(MissingCredentialError):
        load_credentials(secrets_path=secrets_path)

    monkeypatch.delenv("CRYPTOQUANT_API_KEY", raising=False)
    monkeypatch.setenv("CRYPTOQUANT_API_KEY", "env_cq")
    with pytest.raises(MissingCredentialError):
        load_credentials(secrets_path=secrets_path)


def test_partial_credentials(monkeypatch):
    monkeypatch.delenv("CRYPTOQUANT_API_KEY", raising=False)
    monkeypatch.setenv("CRYPTOQUANT_API_KEY", "partial_cq")

    creds = load_credentials(required_keys=("CRYPTOQUANT_API_KEY",))

    assert creds.cryptoquant_api_key == "partial_cq"
    assert creds.coinstats_api_key == ""
    assert creds.binance_api_key == ""

