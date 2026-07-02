"""
Athena Knowledge Validator

Classifies knowledge candidates as:
- Low Quality: contains placeholder, incomplete, or unresolved values (rejected)
- Duplicate: already exists in Semantic Memory (semantic similarity) (rejected)
- New Fact: unique knowledge worth storing (promoted)
- Possible Conflict: contradicts existing Semantic Memory entry (queued for reconciliation)

This is the foundation for Capability 2: Memory Reconciliation.
It does NOT resolve conflicts; it only detects and records them.

Quality gates are deterministic (no LLM calls). The validator is the primary
quality gate for the Learning Pipeline (Extraction -> Validation).
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
        'nobody', 'nothing', 'nowhere',
    })

    # Expanded multi-word placeholder phrases that indicate missing/empty values
    _PLACEHOLDER_PHRASES = frozenset({
        'none specified', 'not provided', 'no value', 'value unknown',
        'no name', 'no answer', 'no info', 'no information',
        'nothing specified', 'nothing provided',
        'missing', 'empty', 'blank', 'unresolved', 'to be determined',
    })

    # Pronouns that indicate unresolved references when used as a value
    _PRONOUN_VALUES = frozenset({
        'her', 'him', 'it', 'they', 'she', 'he', 'them', 'its', 'his',
    })

    # Imperative command prefixes — statements that instruct the assistant to do
    # something are never durable user knowledge
    _IMPERATIVE_PREFIXES = frozenset({
        'respond with', 'respond', 'say', 'repeat', 'write', 'translate',
        'count', 'tell me', 'print', 'open', 'create', 'generate',
        'list', 'show', 'give', 'provide', 'calculate', 'solve',
        'find', 'search', 'look up', 'call', 'run', 'execute',
        'send', 'read', 'play', 'start', 'stop', 'delete',
    })

    # Patterns that describe the conversation itself rather than durable user facts
    _CONVERSATIONAL_PATTERNS = frozenset({
        'user says', 'user responds', 'user asks', 'user writes',
        'user requested', 'user greeted', 'user told', 'user said',
        'user replied', 'user mentioned', 'user answered',
        'user stated', 'user declared', 'user exclaimed',
        'user typed', 'user input', 'user entered',
        'user gave the', 'user provided the',
        'user responded with', 'user asks about', 'user asked about',
        'user says:', 'user responds:', 'user asks:', 'user writes:',
        'user said:', 'user replied:', 'user mentioned:',
    })

    # Short phrases that indicate the user is testing the assistant
    _TESTING_PHRASES = frozenset({
        'testing', 'just testing', 'test message', 'test input',
        'repeat after me', 'only answer', 'answer with',
        'nothing else', 'output only', 'output exactly',
        'say exactly', 'say nothing else', 'just output',
        'respond exactly', 'reply exactly',
    })

    # Last-word endings that indicate an incomplete fact:
    # the statement trails off without providing the actual value.
    # These are linking verbs, prepositions, and relational words that
    # grammatically require a complement to form a complete fact.
    _INCOMPLETE_LAST_WORDS = frozenset({
        # Linking verbs without complement
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        # Possession without object
        'has', 'have', 'had', 'having',
        # Prepositions requiring an object
        'in', 'at', 'on', 'for', 'with', 'as', 'about', 'from', 'to',
        'by', 'of', 'into', 'onto', 'upon', 'within', 'without',
        # Relational verbs without complement
        'named', 'called', 'known', 'referred', 'considered',
        # Action verbs without object
        'studies', 'studied', 'studying',
        'works', 'worked', 'working',
        'lives', 'lived', 'living',
        'likes', 'liked', 'loving', 'loves',
        'prefers', 'preferred', 'preferring',
        'enjoys', 'enjoyed', 'enjoying',
        'hates', 'hated', 'hating',
        # Communication verbs (describe conversation acts)
        'says', 'said', 'saying',
        'responds', 'responded', 'responding',
        'asks', 'asked', 'asking',
        'replies', 'replied', 'replying',
        'writes', 'wrote', 'writing',
    })

    # Common conversational tokens that should never be stored as facts
    # when they appear as the entire statement (isolated user echo)
    _ECHO_TOKENS = frozenset({
        'hello', 'world', 'ok', 'okay', 'hi', 'hey', 'yes', 'no',
        'testing', 'test', 'thanks', 'thank you', 'thanks!', 'bye',
        'goodbye', 'goodnight', 'good morning', 'good afternoon',
        'lol', 'lmao', 'haha', 'hmm', 'huh', 'ah', 'oh', 'um', 'uh',
    })

    def __init__(self, semantic_memory: SemanticMemory) -> None:
        self.semantic_memory = semantic_memory
        self.conflicts: List[dict] = []  # Stores detected conflicts for future reconciliation

    @staticmethod
    def _is_low_quality(statement: str) -> bool:
        """Deterministically detect ANY knowledge that should NOT be stored.

        Quality gates (checked in order):
          1. Empty / blank
          2. Imperative commands (e.g. "Respond with Hello")
          3. Conversational behavior (e.g. "User says Hello")
          4. Testing interactions (e.g. "repeat after me")
          5. Echoed user text (single-word tokens)
          6. Placeholder values / phrases
          7. Incomplete facts (trailing verb or preposition)
          8. Vague / unresolved values

        No LLM calls. Pure string/pattern matching.
        Returns True if the statement is LOW QUALITY and should be REJECTED.
        """
        text = statement.strip().lower()
        if not text:
            return True

        words = text.split()
        if not words:
            return True

        # ──────────────────────────────────────────────────
        # GATE 1 — Imperative commands
        # Statements that command the assistant to do something
        # are never durable user knowledge.
        # ──────────────────────────────────────────────────
        first_two = ' '.join(words[:2]) if len(words) >= 2 else words[0]
        if first_two in KnowledgeValidator._IMPERATIVE_PREFIXES:
            return True
        if words[0] in KnowledgeValidator._IMPERATIVE_PREFIXES:
            return True

        # ──────────────────────────────────────────────────
        # GATE 2 — Conversational behavior
        # Statements describing the conversation, not the user.
        # e.g. "User says Hello", "User responds OK"
        # ──────────────────────────────────────────────────
        for pattern in KnowledgeValidator._CONVERSATIONAL_PATTERNS:
            if pattern in text:
                return True

        # ──────────────────────────────────────────────────
        # GATE 3 — Testing interactions
        # Short test phrases that should never create memory.
        # ──────────────────────────────────────────────────
        if text in KnowledgeValidator._TESTING_PHRASES:
            return True
        # Also check if the statement contains a testing phrase as its core
        # (e.g. "respond with" embedded in a longer utterance)
        for phrase in KnowledgeValidator._TESTING_PHRASES:
            if phrase in text:
                # Only reject if the testing phrase is the main content,
                # not part of a valid fact description
                if len(text) < 20:
                    return True

        # ──────────────────────────────────────────────────
        # GATE 4 — Echoed user text
        # Single words or short tokens that are just the user's
        # conversational text, not facts about the user.
        # ──────────────────────────────────────────────────
        if len(words) == 1 and words[0] in KnowledgeValidator._ECHO_TOKENS:
            return True

        # ──────────────────────────────────────────────────
        # GATE 5 — Placeholder values in final position
        # The last word is most commonly where the knowledge value sits.
        # ──────────────────────────────────────────────────
        last_word = words[-1]
        if last_word in KnowledgeValidator._PLACEHOLDER_VALUES | KnowledgeValidator._PRONOUN_VALUES:
            return True

        # ──────────────────────────────────────────────────
        # GATE 6 — Multi-word placeholder phrases
        # ──────────────────────────────────────────────────
        for phrase in KnowledgeValidator._PLACEHOLDER_PHRASES:
            if phrase in text:
                return True

        # ──────────────────────────────────────────────────
        # GATE 7 — Vague/unresolved value words anywhere in text
        # ──────────────────────────────────────────────────
        for w in words:
            if w in {'something', 'someone', 'somebody', 'somewhere',
                      'anything', 'anyone', 'anybody', 'anywhere',
                      'everything', 'everyone', 'everybody', 'everywhere'}:
                return True

        # ──────────────────────────────────────────────────
        # GATE 8 — Standalone "x" as a word
        # ──────────────────────────────────────────────────
        if ' x ' in f' {text} ':
            return True

        # ──────────────────────────────────────────────────
        # GATE 9 — "unspecified" or "n/a" appearing anywhere
        # ──────────────────────────────────────────────────
        if 'unspecified' in words or 'n/a' in words:
            return True

        # ──────────────────────────────────────────────────
        # GATE 10 — Patterns ending with placeholder phrasing
        # e.g. "User lives in a place", "User is a person"
        # ──────────────────────────────────────────────────
        if text.endswith(' a place') or text.endswith(' a person'):
            return True

        # ──────────────────────────────────────────────────
        # GATE 11 — Incomplete facts (trailing verb/preposition)
        # When the last word indicates more information should follow
        # (linking verb, preposition, or relational word without complement).
        # e.g. "User lives in" → missing value after "in"
        # e.g. "User's name is" → missing value after "is"
        # e.g. "User studies" → missing value after "studies"
        # ──────────────────────────────────────────────────
        if last_word in KnowledgeValidator._INCOMPLETE_LAST_WORDS:
            return True

        # ──────────────────────────────────────────────────
        # GATE 12 — Statements that are just single-word or
        # very short tokens without a "User" subject prefix.
        # These are never valid semantic facts.
        # ──────────────────────────────────────────────────
        if len(words) <= 2 and not text.startswith('user'):
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
