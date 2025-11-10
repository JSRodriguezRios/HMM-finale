import logging

from qcsrc.util.logging_utils import configure_logging, get_logger


def test_get_logger_configures_root():
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    logger = get_logger("test_logger")
    assert logger is logging.getLogger("test_logger")
    assert logging.getLogger().handlers, "Root logger should have handlers after configuration"


def test_configure_logging_respects_environment(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    configure_logging()
    assert logging.getLogger().level == logging.DEBUG

    logger = get_logger("another")
    assert logger.level == logging.NOTSET or logger.level == logging.DEBUG
