"""Project-wide logging: one timestamped file per run under logs/, mirrored to the console."""

from __future__ import annotations

import logging
import sys
from datetime import datetime

from drug_sentiment.config import PROJECT_ROOT

_ROOT_LOGGER = "drug_sentiment"
_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s:%(lineno)d - %(message)s"


def _configure_root() -> logging.Logger:
    root = logging.getLogger(_ROOT_LOGGER)
    if root.handlers:  # already configured in this process
        return root

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{datetime.now():%Y_%m_%d_%H_%M_%S}.log"

    formatter = logging.Formatter(_LOG_FORMAT)
    for handler in (logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child of the project logger, e.g. get_logger(__name__)."""
    _configure_root()
    if name != _ROOT_LOGGER and not name.startswith(f"{_ROOT_LOGGER}."):
        name = f"{_ROOT_LOGGER}.{name}"
    return logging.getLogger(name)
