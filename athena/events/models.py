"""
Athena Event Models

Defines the Event dataclass used throughout the event system.
All events published on the bus are instances of this class.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict


@dataclass(frozen=True)
class Event:
    """
    Represents a domain event in the Athena system.

    Events are immutable data carriers that describe something that happened
    within a subsystem. They are used for decoupled communication between modules.

    Fields:
        id: Unique identifier for this event instance.
        timestamp: When the event occurred (UTC).
        type: The event type/category (e.g., 'ThoughtCreated', 'MemoryLoaded').
        source: Which module/subsystem emitted this event.
        payload: The data associated with this event.
        metadata: Additional contextual information.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    type: str = ""
    source: str = ""
    payload: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure immutable fields are set with defaults if not provided."""
        object.__setattr__(self, "timestamp", self.timestamp or datetime.now(timezone.utc))
        object.__setattr__(self, "payload", self.payload if self.payload is not None else {})
        object.__setattr__(self, "metadata", self.metadata if self.metadata is not None else {})

    def with_payload(self, payload: Any) -> "Event":
        """Return a copy of this event with the given payload."""
        return Event(
            id=self.id,
            timestamp=self.timestamp,
            type=self.type,
            source=self.source,
            payload=payload,
            metadata=dict(self.metadata),
        )

    def __str__(self) -> str:
        """Return a string representation of the event."""
        return f"Event(type={self.type}, source={self.source})"