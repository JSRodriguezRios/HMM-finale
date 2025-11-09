"""Helpers for loading YAML configuration used by pipeline modules."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

from .file_locator import get_config_path


@lru_cache(maxsize=None)
def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file with caching."""

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_assets_config() -> Dict[str, Dict[str, str]]:
    """Return the configured asset metadata keyed by QC ticker."""

    payload = load_yaml_config(get_config_path("assets.yaml"))
    return payload.get("assets", {})


def load_state_map() -> Dict[str, int]:
    """Load the hidden state name-to-index mapping."""

    with get_config_path("state_map.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_settings_config() -> Dict[str, Any]:
    """Return the training and inference settings defined in ``settings.yaml``."""

    return load_yaml_config(get_config_path("settings.yaml"))


__all__ = [
    "load_assets_config",
    "load_settings_config",
    "load_state_map",
    "load_yaml_config",
]
