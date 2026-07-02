"""
Athena Knowledge Validator

Validates knowledge candidates against low-quality rules.

The validator performs ONLY deterministic quality gating:
- Low Quality: contains placeholder, incomplete, or unresolved values (rejected)
- Duplicate: exact normalized match in Semantic Memory (fast-path rejection)

All other candidates pass through as 'valid' for the Memory Reconciler
to determine (via LLM) if they are duplicate, conflict, or different.

This is a QUALITY GATE ONLY. Conflict detection has moved to the Memory Reconciler.
"""

from typing import List, Optional, Tuple

from athena.memory.semantic import SemanticMemory


class KnowledgeValidator:
    """
    Validates knowledge candidates deterministically.

    Classification logic:
        - Low Quality: placeholder/incomplete values deterministically rejected
        - Duplicate: candidate statement has exact normalized match in SM
        - Valid: passes quality gates, ready for memory reconciliation

    This validator does NOT detect conflicts. That is the reconciler's job.
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

    # Imperative command prefixes
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

    # Last-word endings that indicate an incomplete fact
    _INCOMPLETE_LAST_WORDS = frozenset({
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'has', 'have', 'had', 'having',
        'in', 'at', 'on', 'for', 'with', 'as', 'about', 'from', 'to',
        'by', 'of', 'into', 'onto', 'upon', 'within', 'without',
        'named', 'called', 'known', 'referred', 'considered',
        'studies', 'studied', 'studying',
        'works', 'worked', 'working',
        'lives', 'lived', 'living',
        'likes', 'liked', 'loving', 'loves',
        'prefers', 'preferred', 'preferring',
        'enjoys', 'enjoyed', 'enjoying',
        'hates', 'hated', 'hating',
        'says', 'said', 'saying',
        'responds', 'responded', 'responding',
        'asks', 'asked', 'asking',
        'replies', 'replied', 'replying',
        'writes', 'wrote', 'writing',
    })

    # Common conversational tokens that should never be stored as facts
    _ECHO_TOKENS = frozenset({
        'hello', 'world', 'ok', 'okay', 'hi', 'hey', 'yes', 'no',
        'testing', 'test', 'thanks', 'thank you', 'thanks!', 'bye',
        'goodbye', 'goodnight', 'good morning', 'good afternoon',
        'lol', 'lmao', 'haha', 'hmm', 'huh', 'ah', 'oh', 'um', 'uh',
    })

    def __init__(self, semantic_memory: SemanticMemory) -> None:
        self.semantic_memory = semantic_memory

    @staticmethod
    def _is_low_quality(statement: str) -> bool:
        """Deterministically detect ANY knowledge that should NOT be stored.

        Quality gates (checked in order):
          1. Empty / blank
          2. Imperative commands
          3. Conversational behavior
          4. Testing interactions
          5. Echoed user text
          6. Placeholder values / phrases
          7. Incomplete facts
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

        # GATE 1 — Imperative commands
        first_two = ' '.join(words[:2]) if len(words) >= 2 else words[0]
        if first_two in KnowledgeValidator._IMPERATIVE_PREFIXES:
            return True
        if words[0] in KnowledgeValidator._IMPERATIVE_PREFIXES:
            return True

        # GATE 2 — Conversational behavior
        for pattern in KnowledgeValidator._CONVERSATIONAL_PATTERNS:
            if pattern in text:
                return True

        # GATE 3 — Testing interactions
        if text in KnowledgeValidator._TESTING_PHRASES:
            return True
        for phrase in KnowledgeValidator._TESTING_PHRASES:
            if phrase in text and len(text) < 20:
                return True

        # GATE 4 — Echoed user text
        if len(words) == 1 and words[0] in KnowledgeValidator._ECHO_TOKENS:
            return True

        # GATE 5 — Placeholder values in final position
        last_word = words[-1]
        if last_word in KnowledgeValidator._PLACEHOLDER_VALUES | KnowledgeValidator._PRONOUN_VALUES:
            return True

        # GATE 6 — Multi-word placeholder phrases
        for phrase in KnowledgeValidator._PLACEHOLDER_PHRASES:
            if phrase in text:
                return True

        # GATE 7 — Vague/unresolved value words anywhere
        for w in words:
            if w in {'something', 'someone', 'somebody', 'somewhere',
                      'anything', 'anyone', 'anybody', 'anywhere',
                      'everything', 'everyone', 'everybody', 'everywhere'}:
                return True

        # GATE 8 — Standalone "x" as a word
        if ' x ' in f' {text} ':
            return True

        # GATE 9 — "unspecified" or "n/a" appearing anywhere
        if 'unspecified' in words or 'n/a' in words:
            return True

        # GATE 10 — Patterns ending with placeholder phrasing
        if text.endswith(' a place') or text.endswith(' a person'):
            return True

        # GATE 11 — Incomplete facts (trailing verb/preposition)
        if last_word in KnowledgeValidator._INCOMPLETE_LAST_WORDS:
            return True

        # GATE 12 — Very short statements without "User" subject
        if len(words) <= 2 and not text.startswith('user'):
            return True

        return False

    def classify(self, statement: str, confidence: float, category: str) -> Tuple[str, Optional[str]]:
        """
        Classify a knowledge candidate.

        Returns:
            (classification, None)

            classification is one of:
                'low_quality' — rejected (placeholder/incomplete)
                'duplicate'   — exact normalized match exists in SM (fast-path)
                'valid'       — passes quality gates, ready for reconciliation

            This method does NOT detect conflicts. Conflict detection
            is handled by the Memory Reconciler via LLM.
        """
        # First check: reject low-quality / placeholder facts deterministically
        if self._is_low_quality(statement):
            return ('low_quality', None)

        # Check for exact normalized duplicate (fast-path, deterministic)
        normalized_new = SemanticMemory.normalize(statement)
        for entry in self.semantic_memory.query():
            existing_content = str(getattr(entry, 'content', entry))
            normalized_existing = SemanticMemory.normalize(existing_content)
            if normalized_new == normalized_existing:
                return ('duplicate', None)
            # Near-duplicate: one is substring of the other at meaningful length
            if len(normalized_new) > 5 and len(normalized_existing) > 5:
                if normalized_new in normalized_existing or normalized_existing in normalized_new:
                    return ('duplicate', None)

        # Pass through to reconciler for LLM-based classification
        return ('valid', None)
