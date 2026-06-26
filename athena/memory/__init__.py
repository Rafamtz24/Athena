"""
Athena Memory Package

Architectural foundation for Athena's memory system.
Provides abstractions for working, episodic, and semantic memory.
All memory systems are managed through MemoryManager.
"""

from athena.memory.manager import MemoryManager
from athena.memory.models import MemoryEntry

__all__ = ["MemoryManager", "MemoryEntry"]