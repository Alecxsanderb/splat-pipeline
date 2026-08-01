"""Console + file logging setup for pipeline runs."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from pipeline.config import LoggingConfig

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def configure_logging(config: LoggingConfig) -> Path:
    """Attach console and file handlers to the root logger. Returns the log file path."""
    config.log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = config.log_dir / f"run-{timestamp}.log"

    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(config.level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return log_file
