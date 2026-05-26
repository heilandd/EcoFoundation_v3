"""Logging via loguru. One configure call at app start; modules call ``get_logger``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from loguru import logger

_DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

LogLevel = Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]


def configure_logging(
    level: LogLevel = "INFO",
    log_file: Path | str | None = None,
    *,
    capture_warnings: bool = True,
    serialize: bool = False,
) -> None:
    """Configure loguru sinks.

    Parameters
    ----------
    level
        Minimum log level for the console sink.
    log_file
        Optional path to also log into a file. Parent dirs are created.
    capture_warnings
        If True, redirect ``warnings.warn`` into loguru.
    serialize
        If True, file sink writes JSON lines (good for run manifests).
    """
    logger.remove()
    logger.add(sys.stderr, level=level, format=_DEFAULT_FORMAT, enqueue=False)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            level="DEBUG",
            format=_DEFAULT_FORMAT,
            serialize=serialize,
            enqueue=True,
            backtrace=True,
            diagnose=False,
            rotation="100 MB",
        )

    if capture_warnings:
        import logging
        import warnings

        class _InterceptHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
                try:
                    lvl = logger.level(record.levelname).name
                except ValueError:
                    lvl = record.levelno
                logger.opt(depth=6, exception=record.exc_info).log(lvl, record.getMessage())

        # Intercept stdlib logging at WARNING+. Going lower floods the log with
        # numba/h5py/matplotlib DEBUG noise — 100s of MB on a Xenium run.
        logging.basicConfig(handlers=[_InterceptHandler()], level=logging.WARNING, force=True)
        # Silence the noisiest stdlib loggers explicitly. fontTools is set to
        # ERROR because matplotlib's font discovery generates harmless WARNINGS
        # that flood the log on every figure.
        for noisy in ("numba", "h5py", "matplotlib", "PIL"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        logging.getLogger("fontTools").setLevel(logging.ERROR)
        logging.captureWarnings(True)
        warnings.simplefilter("default")


def get_logger(name: str | None = None):
    """Bind a logger to a module name. ``name=None`` returns the root logger."""
    if name is None:
        return logger
    return logger.bind(name=name)
