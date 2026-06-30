"""
Athena Knowledge Validator

Classifies knowledge candidates as:
- Duplicate: already exists in Semantic Memory (semantic similarity)
- New Fact: unique knowledge worth storing
- Possible Conflict: conflicts with existing Semantic Memory entry

This is the foundation for Capability 2: Memory Reconciliation.
It does NOT resolve conflicts; it only detects and records them.
"""

from typing import List, Optional, Tuple
from athena.memory.semantic import SemanticMemory


class KnowledgeValidator:
    """
    Validates knowledge candidates against existing Semantic Memory.

    Classification logic (simple string-based):
        - Duplicate: candidate statement is already present in semantic memory
        - Possible Conflict: candidate contradicts an existing entry
        - New Fact: neither duplicate nor conflict

    Conflicts are stored in a list for future reconciliation by the Memory Reconciler.
    """

    def __init__(self, semantic_memory: SemanticMemory) -> None:
        self.semantic_memory = semantic_memory
        self.conflicts: List[dict] = []  # Stores detected conflicts for future reconciliation

    def classify(self, statement: str, confidence: float, category: str) -> Tuple[str, Optional[str]]:
        """
        Classify a knowledge candidate.

        Returns:
            (classification, conflict_id_or_none)
            
            classification is one of: 'duplicate', 'new_fact', 'possible_conflict'
            conflict_id is the ID of the conflicting entry if applicable, else None
            
        If 'possible_conflict', Semantic Memory is NOT updated and the conflict is recorded.
        """
        # Get all existing semantic memory entries as plain text strings
        existing_entries = []
        for entry in self.semantic_memory.query():
            if hasattr(entry, 'content'):
                existing_entries.append((entry.content, entry))
            elif isinstance(entry, str):
                existing_entries.append((entry, None))

        # Check for duplicate: exact or near-exact match (case-insensitive)
        statement_lower = statement.strip().lower()
        for content, _ in existing_entries:
            content_lower = content.strip().lower()
            if statement_lower == content_lower:
                return ('duplicate', None)
            # Near-duplicate: one is substring of the other and both are meaningful length
            if len(statement_lower) > 5 and len(content_lower) > 5:
                if statement_lower in content_lower or content_lower in statement_lower:
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
