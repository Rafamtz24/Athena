"""
Athena Semantic Memory

Stores factual knowledge that can be queried later.
Persists knowledge to disk so it survives application restarts.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from athena.config.settings import get_settings
from athena.memory.models import MemoryEntry

# Storage location for persistent semantic memory
_SEMANTIC_MEMORY_PATH = Path(get_settings().storage.semantic_memory_path)


class SemanticMemory:
    """
    Stores factual knowledge.

    Purpose:
        Remember facts and general knowledge.

    Persistence:
        Automatically loads knowledge from disk on initialization.
        Automatically saves knowledge after each learn() call.

    Methods:
        learn(content, metadata): Store factual data.
        query(): Get all stored factual data.
    """

    def __init__(self) -> None:
        self._knowledge: list[MemoryEntry] = []
        self._ensure_storage_dir()
        self._load()

    def _ensure_storage_dir(self) -> None:
        """Create the data directory if it does not exist."""
        _SEMANTIC_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _serialize_entry(self, entry: MemoryEntry) -> dict:
        """Convert a MemoryEntry to a JSON-serializable dict."""
        return {
            "id": entry.id,
            "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
            "content": entry.content,
            "metadata": entry.metadata,
        }

    def _deserialize_entry(self, data: dict) -> MemoryEntry:
        """Reconstruct a MemoryEntry from a JSON dict."""
        entry = MemoryEntry(content=data.get("content"), metadata=data.get("metadata"))
        entry.id = data.get("id", uuid.uuid4())
        ts = data.get("timestamp")
        if ts:
            try:
                entry.timestamp = datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                entry.timestamp = datetime.now(timezone.utc)
        else:
            entry.timestamp = datetime.now(timezone.utc)
        return entry

    def _load(self) -> None:
        """Load semantic memory from disk."""
        if not _SEMANTIC_MEMORY_PATH.exists():
            return
        try:
            with open(_SEMANTIC_MEMORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries_data = data.get("entries", [])
            self._knowledge = [self._deserialize_entry(e) for e in entries_data]
        except (json.JSONDecodeError, KeyError, TypeError):
            self._knowledge = []

    def _save(self) -> None:
        """Save semantic memory to disk."""
        entries_data = [self._serialize_entry(e) for e in self._knowledge]
        temp_path = _SEMANTIC_MEMORY_PATH.with_suffix(".json.tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump({"entries": entries_data}, f, indent=2, default=str)
        temp_path.replace(_SEMANTIC_MEMORY_PATH)

    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text for deterministic duplicate comparison.

        Normalizes:
            - leading whitespace
            - trailing whitespace
            - repeated internal whitespace (collapsed to single space)
            - trailing punctuation (. ! ?)
            - case (lowercased)

        The original text is preserved for storage; the normalized form
        is used only for comparison.
        """
        s = str(text).strip()
        # Collapse repeated internal whitespace to single space
        s = ' '.join(s.split())
        # Strip trailing punctuation (. ! ?)
        s = s.rstrip('.!?')
        # Lowercase
        return s.lower()

    def learn(self, content: Any, metadata: dict | None = None) -> str:
        """Store a factual entry. Returns the entry ID.
        
        Duplicate prevention: if a normalized equivalent already exists,
        this method returns the existing entry's ID without creating a duplicate.
        Normalization handles: whitespace, trailing punctuation, case.
        """
        content_str = str(content).strip()
        normalized_new = self.normalize(content_str)
        
        # Check for normalized duplicate
        for entry in self._knowledge:
            if self.normalize(str(entry.content)) == normalized_new:
                return entry.id
        
        entry = MemoryEntry(content=content_str, metadata=metadata or {})
        self._knowledge.append(entry)
        self._save()
        return entry.id

    def query(self) -> list[MemoryEntry]:
        return list(self._knowledge)

    def update(self, entry_id: str, new_content: Any) -> bool:
        """Update an existing entry by ID. Returns True if found and updated."""
        for entry in self._knowledge:
            if entry.id == entry_id:
                entry.content = new_content
                self._save()
                return True
        return False

    def get_entry_by_id(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve a single entry by ID."""
        for entry in self._knowledge:
            if entry.id == entry_id:
                return entry
        return None

    def remove(self, entry_id: str) -> bool:
        """Remove an entry by ID. Returns True if found and removed."""
        for i, entry in enumerate(self._knowledge):
            if entry.id == entry_id:
                self._knowledge.pop(i)
                self._save()
                return True
        return False