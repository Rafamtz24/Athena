"""
Athena Working Memory

Short-term session memory for temporary storage during a single interaction cycle.
"""

from typing import Any

from athena.memory.models import MemoryEntry


class WorkingMemory:
    """
    Temporary session memory for active processing.

    Purpose:
        Hold data that is currently being used by the Athena brain.
        This memory is cleared when the session ends.

    Methods:
        store(content, metadata): Add data to working memory.
        retrieve(): Get all data from working memory.
        clear(): Remove all data from working memory.
    """

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []

    def store(self, content: Any, metadata: dict | None = None) -> str:
        entry = MemoryEntry(content=content, metadata=metadata or {})
        self._entries.append(entry)
        return entry.id

    def retrieve(self) -> list[MemoryEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()