"""
Athena Memory Manager

Central coordinator for all memory systems.
Provides single public interface for AthenaBrain to interact with memory.
"""

from athena.memory.models import MemoryEntry
from athena.memory.episodic import EpisodicMemory
from athena.memory.semantic import SemanticMemory
from athena.memory.working import WorkingMemory


class MemoryManager:
    """
    Owns all memory system instances.

    Provides one public interface to all memory operations.
    AthenaBrain communicates ONLY with this class.
    Memory systems never communicate directly with AthenaBrain.

    Methods:
        init_working_memory(): Initialize working memory instance.
        init_episodic_memory(): Initialize episodic memory instance.
        init_semantic_memory(): Initialize semantic memory instance.
        store_working(content, metadata): Store in working memory.
        get_working(): Retrieve from working memory.
        clear_working(): Clear working memory.
        remember(content, metadata): Store in episodic memory.
        recall(): Retrieve from episodic memory.
        learn(content, metadata): Store in semantic memory.
        query_semantic(): Retrieve from semantic memory.
    """

    def __init__(self) -> None:
        self.working_memory = WorkingMemory()
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()

    # -- Working Memory Interface --

    def store_working(self, content, metadata=None):
        return self.working_memory.store(content, metadata)

    def get_working(self):
        return self.working_memory.retrieve()

    def clear_working(self):
        self.working_memory.clear()

    # -- Episodic Memory Interface --

    def remember(self, content, metadata=None):
        return self.episodic_memory.remember(content, metadata)

    def recall(self):
        return self.episodic_memory.recall()

    # -- Semantic Memory Interface --

    def learn(self, content, metadata=None):
        return self.semantic_memory.learn(content, metadata)

    def query_semantic(self):
        return self.semantic_memory.query()