"""
Athena Memory Manager

Central coordinator for all memory systems.
Provides single public interface for AthenaBrain to interact with memory.
"""

from athena.memory.models import MemoryEntry
from athena.memory.semantic import SemanticMemory
from athena.memory.working import WorkingMemory


class MemoryManager:
    """
    Owns all memory system instances.

    Provides one public interface to all memory operations.
    AthenaBrain communicates ONLY with this class.
    Memory systems never communicate directly with AthenaBrain.

    Two stores, not three. An EpisodicMemory once sat between these, holding
    raw "User: …/Assistant: …" transcripts of the session. It was never
    persisted, so it could not serve long-term recall, and what it held was
    the conversation that Working Memory already carries — so it reached the
    prompt as a second copy of the same turns. Storing transcripts verbatim
    is the known dead end for agent memory: the useful part of an exchange is
    the fact it established, which is what Semantic Memory keeps.

    Methods:
        init_working_memory(): Initialize working memory instance.
        init_semantic_memory(): Initialize semantic memory instance.
        store_working(content, metadata): Store in working memory.
        get_working(): Retrieve from working memory.
        clear_working(): Clear working memory.
        learn(content, metadata): Store in semantic memory.
        query_semantic(): Retrieve from semantic memory.
    """

    def __init__(self) -> None:
        self.working_memory = WorkingMemory()
        self.semantic_memory = SemanticMemory()

    # -- Working Memory Interface --

    def store_working(self, content, metadata=None):
        return self.working_memory.store(content, metadata)

    def get_working(self):
        return self.working_memory.retrieve()

    def clear_working(self):
        self.working_memory.clear()

    # -- Semantic Memory Interface --

    def learn(self, content, metadata=None):
        return self.semantic_memory.learn(content, metadata)

    def query_semantic(self):
        return self.semantic_memory.query()

    # -- Candidate Memory Interface --

    def store_candidate(self, content, metadata=None):
        """Store a knowledge candidate in working memory."""
        return self.working_memory.store_candidate(content, metadata)

    def get_candidates(self):
        """Retrieve all knowledge candidates from working memory."""
        return self.working_memory.get_candidates()

    def remove_candidate(self, index):
        """Remove a candidate from working memory by index."""
        return self.working_memory.remove_candidate(index)