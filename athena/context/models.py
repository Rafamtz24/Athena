"""Context package data models.

Defines the data structures used by the Context Budget Manager to compile
context sources into packages for the reasoning and learning pipelines.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ContextSource:
    """A single source of context with metadata for budgeting.

    Every context source (User Input, Working Memory, Semantic Memory,
    Chat History, Tool Context) exposes metadata instead of relying on
    hardcoded rules. This allows the Context Budget Manager to make
    deterministic budgeting decisions without tool-specific knowledge.

    Attributes:
        name: Human-readable identifier for this context source.
        content: The raw text content of this source.
        priority: Priority level (higher = more important).
            100 = User Input (never trimmed)
            95  = System Prompt (never trimmed)
            90  = Working Memory (not trimmed in this milestone)
            80  = Semantic Memory (not trimmed in this milestone)
            70  = Tool Context (may be trimmed)
            60  = Chat History (may be trimmed)
        learning_visible: Whether this source is visible to the
            Knowledge Extractor for durable knowledge extraction.
        truncatable: Whether this source may be truncated (vs. only
            kept or removed entirely).
    """
    name: str = ""
    content: str = ""
    priority: int = 0
    learning_visible: bool = True
    truncatable: bool = False


@dataclass
class ReasoningContextPackage:
    """A compiled package of context for the reasoning pipeline.

    The Context Budget Manager produces this package after budgeting.
    It is consumed by PromptBuilder, which renders it into the final prompt.

    The package is guaranteed to fit within the active provider's context
    window (after reserving generation budget).

    Attributes:
        sources: Ordered list of ContextSource objects that fit within
            the budget. Order reflects priority (highest first).
        total_tokens: Total token count of all included sources.
        generation_budget: Tokens reserved for model generation.
        context_window: The provider's total context window size.
        trimmed_sources: Names of sources that were partially or fully
            removed during budgeting.
    """
    sources: list = field(default_factory=list)
    total_tokens: int = 0
    generation_budget: int = 0
    context_window: int = 0
    trimmed_sources: list = field(default_factory=list)


@dataclass
class LearningContextPackage:
    """A compiled package of context for the learning pipeline.

    The Context Budget Manager produces this package after budgeting.
    It is consumed by KnowledgeExtractor for durable knowledge extraction.

    Only context sources with learning_visible=True are included.

    Attributes:
        sources: Ordered list of ContextSource objects visible for learning.
        conversation: The full conversation text for knowledge extraction.
        tool_context_content: Tool context content (if learning_visible).
    """
    sources: list = field(default_factory=list)
    conversation: str = ""
    tool_context_content: str = ""
