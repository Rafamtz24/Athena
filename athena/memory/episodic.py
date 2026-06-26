"""
Athena Episodic Memory

Stores experiences (past events) for recall later.
"""

from typing import Any

from athena.memory.models import MemoryEntry


class EpisodicMemory:
    """
    Stores past experiences.

    Purpose:
        Remember what happened in previous interactions.

    Methods:
        remember(content, metadata): Store an experience.
        recall(): Get all stored experiences.
    """

    def __init__(self) -> None:
        self._episodes: list[MemoryEntry] = []

    def remember(self, content: Any, metadata: dict | None = None) -> str:
        entry = MemoryEntry(content=content, metadata=metadata or {})
        self._episodes.append(entry)
        return entry.id

    def recall(self) -> list[MemoryEntry]:
        return list(self._episodes)