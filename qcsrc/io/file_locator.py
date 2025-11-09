"""Path utilities for locating config and data directories."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    """Return the repository root directory."""

    return _PROJECT_ROOT


def get_config_path(name: str) -> Path:
    """Return the path to a configuration file under ``config/``."""

    return _PROJECT_ROOT / "config" / name


def get_data_path(*parts: Iterable[str]) -> Path:
    """Return a path under the ``data`` directory."""

    # ``Path`` flattens nested sequences automatically when expanded.
    return _PROJECT_ROOT.joinpath("data", *parts)


def ensure_directory(path: Path) -> Path:
    """Ensure ``path`` exists as a directory and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["ensure_directory", "get_config_path", "get_data_path", "get_project_root"]
