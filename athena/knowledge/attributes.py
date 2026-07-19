"""
Athena Attribute Facts

Deterministic parsing of knowledge statements into structured
(subject, attribute, value) facts for a small set of common,
single-valued attributes (name, location, color, operating system...).

Why this exists:
    Conflict detection was previously delegated entirely to the LLM via the
    MemoryReconciler. On small / simple reasoning models that classification
    is unreliable — contradictory facts ("User's name is TestUser" vs
    "User's name is Alex") were both kept. For single-valued attributes the
    answer is deterministic and does not need a model: if a new fact states a
    different value for the SAME attribute of the SAME subject, it CONFLICTS
    and the newer value wins.

    This module recognizes only clear, unambiguous single-valued attributes.
    Anything it does not recognize returns None, and the caller falls back to
    the LLM-based reconciler. That keeps the deterministic layer conservative
    (no false conflicts) while making the common cases model-independent.

A single-valued attribute is one a subject can only have one of at a time:
    - a person's name
    - where a person lives
    - a pet's color
    - the operating system

Multi-valued relationships (e.g. "User has a dog named Rex" — a user may
have several pets) are intentionally NOT modeled here; they go to the LLM.
"""

import re
from dataclasses import dataclass
from typing import Optional

from athena.memory.semantic import SemanticMemory


# Subjects that all refer to the user, normalized to a single canonical form.
_SUBJECT_ALIASES = {
    "my": "user",
    "i": "user",
    "me": "user",
    "the user": "user",
    "users": "user",
}


# Attribute patterns: (compiled regex, attribute key, implicit subject or None).
#
# Each regex is anchored and must expose a 'value' group. Patterns that name a
# subject expose a 'subject' group; patterns for a fixed subject (e.g. the
# operating system belongs to the system) declare an implicit subject instead.
#
# Order matters only for specificity — more specific patterns (favorite color)
# are listed before more general ones (color).
_ATTRIBUTE_PATTERNS = [
    # ── favorite color (more specific than plain color) ──
    (re.compile(
        r"^(?P<subject>\w+)(?:'s|s')?\s+favou?rite\s+colou?r\s+is\s+(?P<value>.+)$",
        re.IGNORECASE), "favorite_color", None),

    # ── name ──
    (re.compile(
        r"^(?P<subject>\w+)(?:'s|s')?\s+name\s+is\s+(?P<value>.+)$",
        re.IGNORECASE), "name", None),
    (re.compile(
        r"^(?P<subject>\w+)\s+is\s+named\s+(?P<value>.+)$",
        re.IGNORECASE), "name", None),

    # ── location ──
    (re.compile(
        r"^(?P<subject>\w+)\s+lives?\s+in\s+(?P<value>.+)$",
        re.IGNORECASE), "location", None),
    (re.compile(
        r"^(?P<subject>\w+)\s+(?:is\s+)?located\s+in\s+(?P<value>.+)$",
        re.IGNORECASE), "location", None),
    (re.compile(
        r"^(?P<subject>\w+)\s+resides?\s+in\s+(?P<value>.+)$",
        re.IGNORECASE), "location", None),

    # ── color ──
    (re.compile(
        r"^(?P<subject>\w+)(?:'s|s')?\s+colou?r\s+is\s+(?P<value>.+)$",
        re.IGNORECASE), "color", None),

    # ── operating system (fixed subject: the system) ──
    (re.compile(
        r"^(?:the\s+)?operating\s+system\s+is\s+(?P<value>.+)$",
        re.IGNORECASE), "operating_system", "system"),
]


@dataclass(frozen=True)
class AttributeFact:
    """A statement parsed into a structured single-valued attribute fact.

    Attributes:
        subject: Canonical, lowercased subject (e.g. "user", "rex", "system").
        attribute: Canonical attribute key (e.g. "name", "location").
        value: The value exactly as extracted (original case preserved for
               storage; use ``value_norm`` for comparison).
    """

    subject: str
    attribute: str
    value: str

    @property
    def key(self) -> tuple[str, str]:
        """The (subject, attribute) identity used to detect same-attribute facts."""
        return (self.subject, self.attribute)

    @property
    def value_norm(self) -> str:
        """Normalized value for deterministic equality comparison."""
        return SemanticMemory.normalize(self.value)


def parse_fact(statement: str) -> Optional[AttributeFact]:
    """Parse a statement into an AttributeFact, or None if unrecognized.

    Returns None (rather than guessing) whenever the statement does not match
    one of the known single-valued attribute patterns. Callers treat None as
    "not deterministically classifiable — defer to the LLM reconciler".

    Args:
        statement: A plain-text knowledge statement.

    Returns:
        An AttributeFact for a recognized single-valued attribute, else None.
    """
    if not statement:
        return None

    text = statement.strip().rstrip(".!?").strip()
    if not text:
        return None

    for pattern, attribute, implicit_subject in _ATTRIBUTE_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue

        value = match.group("value").strip()
        if not value:
            return None

        if implicit_subject is not None:
            subject = implicit_subject
        else:
            subject = match.group("subject").strip().lower()
            subject = _SUBJECT_ALIASES.get(subject, subject)

        return AttributeFact(subject=subject, attribute=attribute, value=value)

    return None
