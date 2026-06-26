"""
Athena Structured Logger Module

Provides a centralized logger instance used throughout the Athena AI platform.
Uses Python's standard logging module with structured formatting.
"""

import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance for the Athena platform.

    If no name is provided, defaults to 'athena' as the logger name.
    The logger is configured with console output at INFO level.

    Args:
        name: Optional name for the logger. Defaults to 'athena'.

    Returns:
        A configured logging.Logger instance.
    """
    logger_name = name or "athena"
    logger = logging.getLogger(logger_name)

    # Avoid adding duplicate handlers if called multiple times
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# Default logger instance for convenience
logger = get_logger()