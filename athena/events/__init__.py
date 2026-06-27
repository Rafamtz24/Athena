"""
Athena Event System

Provides an internal event bus for decoupled communication between subsystems.
Every module can publish events and subscribe to events without tight coupling.
"""

from athena.events.models import Event
from athena.events.bus import EventBus
from athena.events.dispatcher import Dispatcher

__all__ = ["Event", "EventBus", "Dispatcher"]