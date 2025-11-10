"""Logging utilities for pipeline scripts and QuantConnect runtime."""

from __future__ import annotations

import logging
import os
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _resolve_level(level: Optional[int] = None) -> int:
    """Resolve the log level from explicit value or environment variable."""
    if level is not None:
        return level

    env_level = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, env_level, logging.INFO)


def configure_logging(level: Optional[int] = None) -> None:
    """Ensure the root logger is configured with a consistent formatter.

    Calling :func:`configure_logging` multiple times is safe; the root logger
    will only be configured once. Explicit handlers already present are left
    untouched to avoid interfering with QuantConnect's logging system.
    """

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(_resolve_level(level))
        return

    logging.basicConfig(
        level=_resolve_level(level),
        format=_DEFAULT_FORMAT,
        datefmt=_DEFAULT_DATEFMT,
    )


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a module-level logger with shared formatting.

    Parameters
    ----------
    name:
        The logger namespace, typically ``__name__`` from the caller.
    level:
        Optional log level override. Falls back to ``LOG_LEVEL`` environment
        variable and then ``logging.INFO`` when omitted.
    """

    configure_logging(level)
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


__all__ = ["configure_logging", "get_logger"]
