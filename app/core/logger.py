from __future__ import annotations

import logging
from logging import Logger


def setup_logging(level: str = "INFO") -> Logger:
    """Configure a simple root logger for the prototype."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("municipal_assistant")
