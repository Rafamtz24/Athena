"""
Athena Memory Models

Defines the foundational data structures for all memory systems.
MemoryEntry is the base unit that every memory system works with.
"""

import uuid
from datetime import datetime, timezone
from typing import Any


class MemoryEntry:
    """
    A single entry in Athena's memory system.

    This is the atomic unit of memory storage. Every piece of memory
    (working, episodic, or semantic) is stored as a MemoryEntry.

    Fields:
        id: Unique identifier for this entry.
        timestamp: When this entry was created (UTC).
        content: The actual data/payload of the entry.
        metadata: Additional contextual information about the entry.
    """

    def __init__(
        self,
        content: Any = None,
        metadata: dict | None = None,
    ):
        """
        Initialize a MemoryEntry.

        Args:
            content: The data payload of this memory entry.
            metadata: Optional contextual information.
        """
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.now(timezone.utc)
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"MemoryEntry(id={self.id}, timestamp={self.timestamp})"