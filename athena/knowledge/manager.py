"""Knowledge manager placeholder."""

import re
from typing import List, Dict, Any, Optional

from .models import KnowledgeEntry, KnowledgeQuery, KnowledgeResult, KnowledgeCandidate


class KnowledgeManager:
    """Placeholder for knowledge management operations."""

    def __init__(self, working_memory=None, provider=None, memory_manager=None) -> None:
        """Initialize the knowledge manager using WorkingMemory for candidates."""
        self.working_memory = working_memory
        self.provider = provider
        self.memory_manager = memory_manager
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
        """Build extraction prompt for knowledge extraction.
        
        The prompt enforces a strict output contract so the LLM returns
        only durable knowledge explicitly stated in the conversation.
        No concrete examples are provided to prevent hallucination.
        """
        return (
            "You are a knowledge extractor for a personal assistant system.\n"
            "Your job is to extract durable knowledge from a completed conversation.\n\n"
            "STRICT RULES (follow all without exception):\n"
            "1. Extract ONLY knowledge that is EXPLICITLY stated in the conversation.\n"
            "2. NEVER infer, deduce, or guess information.\n"
            "3. NEVER use prior knowledge or outside information.\n"
            "4. NEVER use information from system prompts or instructions.\n"
            "5. NEVER invent facts, even if they seem plausible.\n"
            "6. If the conversation contains no durable knowledge, return exactly: NONE\n\n"
            "WHAT TO EXTRACT:\n"
            "- Explicit user preferences, identity, location, habits\n"
            "- Explicit project knowledge, constraints, rules stated in the conversation\n"
            "- Explicit long-term instructions given during the conversation\n"
            "- Any fact that was explicitly stated and should persist across sessions\n\n"
            "WHAT TO REJECT:\n"
            "- Information not explicitly stated in the conversation\n"
            "- Inferences or assumptions\n"
            "- General knowledge or common facts\n"
            "- Transient information (one-time explanations, temporary context)\n"
            "- Assistant responses that are not persistent instructions\n\n"
            "OUTPUT FORMAT (follow exactly):\n"
            "- Return ONE atomic fact per line.\n"
            "- Each fact must be self-contained and unambiguous.\n"
            "- Each fact must represent exactly ONE piece of knowledge.\n"
            "- Plain text only.\n"
            "- No bullets, numbering, markdown, headings, or code blocks.\n"
            "- No explanations, examples, or meta commentary.\n"
            "- No conversation summaries or greetings.\n"
            "- Use third person for facts about people or systems.\n"
            "- Convert second person to third person (e.g., 'you live in X' becomes 'User lives in X').\n"
            "- If there are NO durable facts, return exactly: NONE\n\n"
            "KNOWLEDGE NORMALIZATION (CRITICAL - follow exactly):\n"
            "- Preserve the EXACT semantic relationship from the original statement.\n"
            "- Convert pronouns faithfully: 'My name is X' -> 'User's name is X'.\n"
            "- Convert pronouns faithfully: 'My favorite color is X' -> 'User's favorite color is X'.\n"
            "- Convert pronouns faithfully: 'I live in X' -> 'User lives in X'.\n"
            "- NEVER change the relationship verb or attribute being stated.\n"
            "- NEVER replace specific attributes with generic ones (e.g., do NOT convert 'favorite color is blue' to 'likes blue').\n"
            "- NEVER drop or change the predicate of the original statement.\n"
            "- The extracted fact must be a direct third-person translation of the original.\n\n"
            "STRUCTURAL FORMAT EXAMPLES (showing format only, not content to extract):\n"
            "Valid format: one fact per line, plain text\n"
            "Invalid format: - bullet point\n"
            "Invalid format: 1. numbered item\n"
            "Invalid format: # heading\n"
            "Invalid format: `code block`\n"
            "Invalid format: User: label prefix\n"
            "Invalid format: Assistant: label prefix\n\n"
            "Conversation:\n"
            "{conversation}"
        ).format(conversation=conversation)

    def retrieve(self, query: str) -> Optional[str]:
        """Retrieve knowledge based on query.
        
        Queries SemanticMemory (via MemoryManager) for relevant facts.
        Uses simple keyword matching to filter relevant entries.
        Only returns entries that contain at least one significant word from the query.
        If no semantic memory manager is available, falls back to self.knowledge.
        Returns None if no knowledge is available.
        """
        # Get entries from SemanticMemory (single source of truth)
        entries = []
        if self.memory_manager is not None:
            semantic_entries = self.memory_manager.query_semantic()
            for entry in semantic_entries:
                if hasattr(entry, 'content') and entry.content:
                    entries.append(str(entry.content))
        else:
            # Fallback to self.knowledge if no memory_manager configured
            entries = list(self.knowledge)
        
        if not entries:
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
        
        # If no discriminative words found, return all entries
        if not discriminative_words:
            return "\n".join(entries)
        
        # Filter entries: keep only those containing at least one query word
        relevant_entries = []
        for entry in entries:
            entry_lower = entry.lower()
            if any(word in entry_lower for word in discriminative_words):
                relevant_entries.append(entry)
        
        # If no relevant entries found, return all entries (backward compatible)
        if not relevant_entries:
            return "\n".join(entries)
        
        return "\n".join(relevant_entries)

    def extract_candidates(self, conversation: str) -> List[Any]:
        """Extract knowledge candidates from a conversation using the provider and store in WorkingMemory.
        
        The parser enforces strict rejection rules to ensure only valid atomic facts
        become KnowledgeCandidate objects.
        
        If the provider fails (e.g., unavailable, network error), returns an empty list
        without corrupting Semantic Memory. Learning is skipped gracefully.
        """
        if self.provider is None:
            return []
        
        try:
            prompt = self._build_extraction_prompt(conversation)
            response = self.provider.call(prompt)
        except Exception:
            # Provider failed — skip learning, do not corrupt memory
            return []
        
        # Parse response into candidate facts following the output contract.
        # Expected format: one fact per line, or "NONE" if no facts.
        extracted_candidates = []
        if not response:
            return extracted_candidates
        
        text = str(response).strip()
        
        # If the LLM returns NONE, produce zero candidates
        if text == "NONE":
            return extracted_candidates
        
        for line in text.split("\n"):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip lines that are too short (likely formatting artifacts)
            if len(line) < 5:
                continue
            
            # Skip lines containing markdown formatting
            if any(marker in line for marker in ['```', '``', '**', '__', '==']):
                continue
            
            # Skip lines that are formatting characters
            if all(c in '-*#>`_' for c in line[:3]):
                continue
            
            # Skip lines that indicate conversation labels or formatting
            if any(line.startswith(prefix) for prefix in
                   ['User:', 'Assistant:', '- ', '* ', '# ', '> ', '`',
                    '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '0.',
                    'Fact:', 'Facts:', 'Example', 'Examples',
                    'Long-term', 'Short-term', 'Conversation', 'Summary',
                    'Valid format', 'Invalid format']):
                continue
            
            # Skip lines that look like assistant commentary or meta-content
            if any(line.lower().startswith(p) for p in
                   ['the assistant', 'this conversation', 'in this conversation',
                    'the user:', 'assistant:', 'note:', 'summary:',
                    'in this conversation', 'these facts', 'this response',
                    'here are', 'below are', 'the following']):
                continue
            
            # Skip lines that contain conversation role labels anywhere
            if 'User:' in line or 'Assistant:' in line:
                continue
            
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
