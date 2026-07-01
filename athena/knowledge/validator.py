"""
Athena Knowledge Validator

Classifies knowledge candidates as:
- Low Quality: contains placeholder, incomplete, or unresolved values (rejected)
- Duplicate: already exists in Semantic Memory (semantic similarity) (rejected)
- New Fact: unique knowledge worth storing (promoted)
- Possible Conflict: contradicts existing Semantic Memory entry (queued for reconciliation)

This is the foundation for Capability 2: Memory Reconciliation.
It does NOT resolve conflicts; it only detects and records them.
"""

from typing import List, Optional, Tuple
from athena.memory.semantic import SemanticMemory


class KnowledgeValidator:
    """
    Validates knowledge candidates against existing Semantic Memory.

    Classification logic (deterministic, string-based):
        - Low Quality: placeholder/incomplete values deterministically rejected
        - Duplicate: candidate statement is already present in semantic memory
        - Possible Conflict: candidate contradicts an existing entry
        - New Fact: unique, valid knowledge worth storing

    Conflicts are stored in a list for future reconciliation by the Memory Reconciler.
    """

    # Words that are never valid as knowledge values (placeholders)
    _PLACEHOLDER_VALUES = frozenset({
        'x', 'unknown', 'unspecified', 'n/a', 'null', 'none',
        'someone', 'something', 'somebody', 'somewhere',
    })

    # Pronouns that indicate unresolved references when used as a value
    _PRONOUN_VALUES = frozenset({
        'her', 'him', 'it', 'they', 'she', 'he', 'them', 'its', 'his',
    })

    def __init__(self, semantic_memory: SemanticMemory) -> None:
        self.semantic_memory = semantic_memory
        self.conflicts: List[dict] = []  # Stores detected conflicts for future reconciliation

    @staticmethod
    def _is_low_quality(statement: str) -> bool:
        """Deterministically detect placeholder, incomplete, or unresolved values.

        No LLM calls. Pure string/pattern matching.
        """
        text = statement.strip().lower()
        if not text:
            return True

        words = text.split()
        if not words:
            return True

        # -- Check the last word (most common position for the knowledge value) --
        last_word = words[-1]
        if last_word in KnowledgeValidator._PLACEHOLDER_VALUES | KnowledgeValidator._PRONOUN_VALUES:
            return True

        # -- Check for any occurrence of "something", "someone", "somebody", "somewhere" --
        # These are never valid values in extracted knowledge
        for w in words:
            if w in {'something', 'someone', 'somebody', 'somewhere'}:
                return True

        # -- Check for standalone "x" as a word (not part of a valid name like "Xena") --
        if ' x ' in f' {text} ':
            return True

        # -- Check for "unspecified" or "n/a" appearing anywhere --
        if 'unspecified' in words or 'n/a' in words:
            return True

        # -- Check for patterns ending with "a place" or "a person" --
        # e.g., "User lives in a place", "User is a person"
        if text.endswith(' a place') or text.endswith(' a person'):
            return True

        return False

    def classify(self, statement: str, confidence: float, category: str) -> Tuple[str, Optional[str]]:
        """
        Classify a knowledge candidate.

        Returns:
            (classification, conflict_id_or_none)

            classification is one of: 'low_quality', 'duplicate', 'new_fact', 'possible_conflict'
            conflict_id is the ID of the conflicting entry if applicable, else None

        If 'possible_conflict', Semantic Memory is NOT updated and the conflict is recorded.
        If 'low_quality', the candidate is rejected — it contains placeholder/incomplete values.
        """
        # First check: reject low-quality / placeholder facts deterministically
        if self._is_low_quality(statement):
            return ('low_quality', None)

        # Get all existing semantic memory entries as plain text strings
        existing_entries = []
        for entry in self.semantic_memory.query():
            if hasattr(entry, 'content'):
                existing_entries.append((entry.content, entry))
            elif isinstance(entry, str):
                existing_entries.append((entry, None))

        # Check for duplicate: normalized match (whitespace, trailing punctuation, case)
        from athena.memory.semantic import SemanticMemory
        normalized_new = SemanticMemory.normalize(statement)
        for content, _ in existing_entries:
            normalized_existing = SemanticMemory.normalize(content)
            if normalized_new == normalized_existing:
                return ('duplicate', None)
            # Near-duplicate: one is substring of the other and both are meaningful length
            if len(normalized_new) > 5 and len(normalized_existing) > 5:
                if normalized_new in normalized_existing or normalized_existing in normalized_new:
                    return ('duplicate', None)

        # Check for conflict: candidate implies opposite meaning of existing entry
        # Simple heuristic: look for negation patterns and contradictory statements
        conflicting_entry = self._find_conflicting_entry(statement, existing_entries)
        if conflicting_entry is not None:
            conflict_id = conflicting_entry.id if hasattr(conflicting_entry, 'id') else str(id(conflicting_entry))
            conflict_record = {
                'candidate_statement': statement,
                'existing_content': conflicting_entry.content if hasattr(conflicting_entry, 'content') else str(conflicting_entry),
                'existing_id': conflict_id,
                'confidence': confidence,
                'category': category
            }
            self.conflicts.append(conflict_record)
            return ('possible_conflict', conflict_id)

        # New fact
        return ('new_fact', None)

    def _find_conflicting_entry(self, statement: str, existing_entries: list) -> Optional[object]:
        """
        Find an existing entry that conflicts with the given statement.
        
        Simple heuristic: look for negation patterns (not, no, never, n't) and
        contradictory numeric values or direct opposites.
        """
        import re
        
        statement_lower = statement.lower()
        
        # Check for negation patterns
        negations = [' not ', " n't ", ' is no ', ' are no ', ' was no ', ' were no ']
        has_negation = any(neg in statement_lower for neg in negations)
        
        if has_negation:
            # Find matching positive entry (one that shares key verbs/nouns but lacks negation)
            for content, _ in existing_entries:
                content_lower = content.lower()
                # Must share meaningful words (at least 3 common words > 3 chars)
                statement_words = set(statement_lower.split())
                content_words = set(content_lower.split())
                common_words = statement_words & content_words
                meaningful_common = [w for w in common_words if len(w) > 3]
                
                if len(meaningful_common) >= 2:
                    # Content should be positive version (no negation, has shared topic words)
                    if "n't" not in content_lower and ' not ' not in content_lower:
                        return self._get_entry_from_list(content, existing_entries)

        # Check for contradictory numeric values (e.g., "5 years" vs "10 years")
        number_pattern = r'\b\d+\s*\w*'
        statement_numbers = re.findall(number_pattern, statement_lower)
        if statement_numbers:
            for content, _ in existing_entries:
                content_numbers = re.findall(number_pattern, content.lower())
                if statement_numbers and content_numbers:
                    # Same topic but different values
                    common_words = set(statement_lower.split()) & set(content.lower().split())
                    if len(common_words) > 2:  # Same general topic
                        return self._get_entry_from_list(content, existing_entries)

        return None

    def _get_entry_from_list(self, content: str, existing_entries: list):
        """Helper to retrieve the original entry object from a content string."""
        content_lower = content.lower()
        for entry_content, entry_obj in existing_entries:
            if entry_content.lower() == content_lower:
                return entry_obj
        return None

    def get_conflicts(self) -> List[dict]:
        """Return all detected conflicts for future reconciliation."""
        return self.conflicts

    def clear_conflicts(self) -> None:
        """Clear the conflict list after processing."""
        self.conflicts = []
