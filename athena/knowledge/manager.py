"""Knowledge manager placeholder."""

import re
from typing import List, Dict, Any, Optional

from .models import KnowledgeEntry, KnowledgeQuery, KnowledgeResult, KnowledgeCandidate


class KnowledgeManager:
    """Placeholder for knowledge management operations."""

    def __init__(self, working_memory=None, provider=None) -> None:
        """Initialize the knowledge manager using WorkingMemory for candidates."""
        self.working_memory = working_memory
        self.provider = provider
        self.knowledge: list[str] = []

    def add_entry(self, entry: KnowledgeEntry) -> str:
        """Add a knowledge entry. Returns empty ID."""
        return ""

    def query(self, query: KnowledgeQuery) -> KnowledgeResult:
        """Search knowledge entries. Returns empty results."""
        return KnowledgeResult()

    def get_entries(self) -> List[KnowledgeEntry]:
        """Get all knowledge entries. Returns empty list."""
        return []

    def delete_entry(self, entry_id: str) -> bool:
        """Delete a knowledge entry. Always returns False."""
        return False

    def update_entry(self, entry_id: str, data: Dict[str, Any]) -> bool:
        """Update a knowledge entry. Returns empty ID."""
        return ""

    def add_candidate(self, candidate: KnowledgeCandidate) -> None:
        """Add a knowledge candidate to WorkingMemory."""
        if self.working_memory is not None:
            self.working_memory.store_candidate(
                statement=candidate.statement,
                confidence=candidate.confidence,
                category=candidate.category
            )

    def get_candidates(self) -> List[KnowledgeCandidate]:
        """Get all knowledge candidates from WorkingMemory."""
        if self.working_memory is not None:
            return self.working_memory.get_candidates()
        return []

    def _build_extraction_prompt(self, conversation: str) -> str:
        """Build extraction prompt for knowledge extraction."""
        return f"You extract durable user knowledge.\n\nExtract only long-term facts worth remembering.\n\nConversation:\n{conversation}"

    def retrieve(self, query: str) -> Optional[str]:
        """Retrieve knowledge based on query.
        
        Uses simple keyword matching to filter relevant entries.
        Only returns entries that contain at least one significant word from the query.
        If self.knowledge is empty, returns None.
        Otherwise, returns only relevant knowledge entries joined by newlines.
        """
        if not self.knowledge:
            return None
        
        # Normalize query: lowercase, split into words, filter short tokens
        query_words = [w.lower() for w in re.findall(r'[a-z]+', query)]
        
        # Filter out common stop words that are too generic to be discriminative
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'can', 'could', 'should', 'would', 'what', 'which', 'who',
                      'how', 'when', 'where', 'why', 'not', 'no', 'and', 'or',
                      'but', 'for', 'with', 'at', 'on', 'in', 'to', 'of', 'it',
                      'this', 'that', 'you', 'i', 'we', 'they', 'he', 'she'}
        
        discriminative_words = [w for w in query_words if len(w) > 2 and w not in stop_words]
        
        # If no discriminative words found, return all knowledge (backward compatible)
        if not discriminative_words:
            return "\n".join(self.knowledge)
        
        # Filter entries: keep only those containing at least one query word
        relevant_entries = []
        for entry in self.knowledge:
            entry_lower = entry.lower()
            if any(word in entry_lower for word in discriminative_words):
                relevant_entries.append(entry)
        
        # If no relevant entries found, return all knowledge (backward compatible)
        if not relevant_entries:
            return "\n".join(self.knowledge)
        
        return "\n".join(relevant_entries)

    def extract_candidates(self, conversation: str) -> List[Any]:
        """Extract knowledge candidates from a conversation using the provider and store in WorkingMemory."""
        if self.provider is None:
            return []
        
        prompt = self._build_extraction_prompt(conversation)
        response = self.provider.call(prompt)
        
        # Parse response into candidate facts (simple newline-separated format)
        extracted_candidates = []
        if response:
            for line in str(response).strip().split("\n"):
                line = line.strip()
                if line and len(line) > 10:
                    from athena.knowledge.models import KnowledgeCandidate
                    candidate = KnowledgeCandidate(
                        statement=line,
                        confidence=0.8,
                        category="extracted"
                    )
                    extracted_candidates.append(candidate)
                    if self.working_memory is not None:
                        self.working_memory.store_candidate(
                            statement=candidate.statement,
                            confidence=candidate.confidence,
                            category=candidate.category
                        )
        return extracted_candidates

    def add(self, knowledge) -> None:
        """Add a knowledge entry to the storage."""
        self.knowledge.append(knowledge)

    def all(self):
        """Return all stored knowledge entries."""
        return self.knowledge

    def promote_candidate(self, candidate) -> bool:
        """Promote a candidate fact to semantic memory. Returns True if successful."""
        from athena.memory.models import MemoryEntry
        entry = MemoryEntry(content=candidate.statement, metadata={
            "type": "knowledge",
            "confidence": candidate.confidence,
            "category": candidate.category
        })
        self.knowledge.append(entry.content)
        return True

    def discard_candidate(self, index: int) -> bool:
        """Remove a candidate from WorkingMemory by index."""
        if self.working_memory is not None:
            return self.working_memory.remove_candidate(index)
        return False
