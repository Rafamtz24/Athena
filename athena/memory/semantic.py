"""
Athena Semantic Memory

Stores factual knowledge that can be queried later.
"""

from typing import Any

from athena.memory.models import MemoryEntry


class SemanticMemory:
    """
    Stores factual knowledge.

    Purpose:
        Remember facts and general knowledge.

    Methods:
        learn(content, metadata): Store factual data.
        query(): Get all stored factual data.
    """

    def __init__(self) -> None:
        self._knowledge: list[MemoryEntry] = []

    def learn(self, content: Any, metadata: dict | None = None) -> str:
        entry = MemoryEntry(content=content, metadata=metadata or {})
        self._knowledge.append(entry)
        return entry.id

    def query(self) -> list[MemoryEntry]:
        return list(self._knowledge)

    def update(self, entry_id: str, new_content: Any) -> bool:
        """Update an existing entry by ID. Returns True if found and updated."""
        for entry in self._knowledge:
            if entry.id == entry_id:
                entry.content = new_content
                return True
        return False

    def get_entry_by_id(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve a single entry by ID."""
        for entry in self._knowledge:
            if entry.id == entry_id:
                return entry
        return None