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

    def prune(self, max_tokens: int, entries: list = None) -> None:
        """Remove oldest entries until total token count fits within max_tokens.

        Working Memory is responsible for maintaining its own sliding window.
        This method removes entries from the BEGINNING (oldest first) until
        the cumulative token count of the remaining entries fits within the
        specified budget.

        Uses a deterministic token estimate of len(text) // 4 (same heuristic
        as the brain's _prune_to_budget). The Context Budget Manager performs
        exact token counting via the provider when compiling the final package.

        Args:
            max_tokens: Maximum total token count allowed.
            entries: List of string entries to prune in-place. If None,
                     prunes self._entries (MemoryEntry objects).
        """
        if max_tokens <= 0:
            if entries is not None:
                entries.clear()
            else:
                self._entries.clear()
            return

        # Determine which list to prune
        target = entries if entries is not None else self._entries
        if not target:
            return

        def estimate_tokens(text: str) -> int:
            """Estimate token count from text length."""
            if isinstance(text, str):
                return len(text) // 4
            # Handle MemoryEntry or other objects
            content = getattr(text, 'content', str(text))
            return len(str(content)) // 4

        # Walk from newest (end) to oldest (start), accumulating tokens
        total = 0
        for i in range(len(target) - 1, -1, -1):
            total += estimate_tokens(target[i])
            if total > max_tokens:
                # Entries 0..i exceed budget; keep i+1..end
                del target[:i + 1]
                return

        # All entries fit — no change needed