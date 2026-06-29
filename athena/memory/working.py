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

    def store_candidate(self, statement: str, confidence: float = 0.0, category: str = "") -> str:
        """Store a knowledge candidate in working memory."""
        from athena.knowledge.models import KnowledgeCandidate
        candidate = KnowledgeCandidate(statement=statement, confidence=confidence, category=category)
        entry_content = f"CANDIDATE:{candidate.statement}|{candidate.confidence}|{candidate.category}"
        return self.store(entry_content, metadata={"type": "candidate", "candidate": candidate})

    def get_candidates(self) -> list:
        """Retrieve all knowledge candidates from working memory."""
        candidates = []
        for entry in self._entries:
            if entry.metadata.get("type") == "candidate" and entry.content.startswith("CANDIDATE:"):
                # Parse: CANDIDATE:<statement>|<confidence>|<category>
                parts = entry.content[len("CANDIDATE:"):]
                parts_list = parts.split("|", 2)
                if len(parts_list) == 3:
                    from athena.knowledge.models import KnowledgeCandidate
                    candidates.append(KnowledgeCandidate(
                        statement=parts_list[0],
                        confidence=float(parts_list[1]),
                        category=parts_list[2]
                    ))
        return candidates

    def remove_candidate(self, index: int) -> bool:
        """Remove a candidate from working memory by index."""
        candidate_indices = []
        for i, entry in enumerate(self._entries):
            if entry.metadata.get("type") == "candidate":
                candidate_indices.append(i)
        
        if 0 <= index < len(candidate_indices):
            self._entries.pop(candidate_indices[index])
            return True
        return False