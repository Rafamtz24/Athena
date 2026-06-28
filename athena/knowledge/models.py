"""Knowledge module data models."""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class KnowledgeEntry:
    """Placeholder for a knowledge entry."""

    id: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeQuery:
    """Placeholder for a knowledge query."""

    text: str = ""
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeResult:
    """Placeholder for a knowledge result."""

    entries: List[KnowledgeEntry] = field(default_factory=list)


@dataclass
class KnowledgeCandidate:
    """Placeholder for a knowledge candidate under consideration."""

    statement: str = ""
    confidence: float = 0.0
    category: str = ""