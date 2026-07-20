"""
Athena Structured Logger Module

Provides a centralized logger instance used throughout the Athena AI platform.
Uses Python's standard logging module with structured formatting.

Can also mirror every EventBus event to the console for debugging, via
subscribe_logger_to_bus() — opt-in, see that function for why.
"""

import logging
from typing import Optional

from athena.events.bus import get_event_bus
from athena.events.models import Event


def _get_logger(name: Optional[str] = None) -> logging.Logger:
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


def _event_callback(event: Event) -> None:
    """Callback that logs every event received from the EventBus."""
    logger = _get_logger()
    logger.info(
        "Event: type=%s, source=%s, payload=%s",
        event.type,
        event.source,
        str(event.payload),
    )


def subscribe_logger_to_bus(bus=None) -> None:
    """Log every pipeline event to the console. Opt-in — call it to enable.

    NOT called on import, deliberately. This subscription existed from the
    start but never fired: the bus registered callbacks in a dict that
    publishing did not read, so the logging silently never happened. Fixing
    that plumbing would have switched this on for the first time, and what it
    does is emit one INFO line per event — eight or so per turn, echoing the
    user's own input back at them, interleaved with the answer as it streams.

    Nothing has missed it in the meantime: the stages worth announcing already
    log directly (see CognitiveEngine). So it stays available for debugging a
    pipeline problem, and off by default.
    """
    if bus is None:
        bus = get_event_bus()

    event_types = [
        "ThoughtCreated",
        "KnowledgeLoaded",
        "ToolPlanned",
        "ToolExecuted",
        "ReasoningStarted",
        "PlanningStarted",
        "ToolsPrepared",
        "ResponseGenerated",
        "ReflectionStarted",
        "ThoughtCompleted",
    ]

    for event_type in event_types:
        bus.subscribe(event_type, _event_callback)


# Default logger instance for convenience
logger = _get_logger()