"""
Athena Brain Module

Core orchestration component that receives requests, reasons, delegates to providers,
and returns responses.
"""

import json
from pathlib import Path

from athena.config.settings import get_settings
from athena.debug.manager import DebugManager
from athena.knowledge.manager import KnowledgeManager
from athena.memory.manager import MemoryManager
from athena.thought.models import Thought
from athena.thought.pipeline import ThoughtPipeline

# Storage location for persistent conversation history
_HISTORY_PATH = Path(get_settings().storage.conversation_history_path)


class AthenaBrain:
    """
    The central brain of the Athena AI platform.

    Responsibilities:
        - Receive incoming requests
        - Create Thought objects
        - Pass Thoughts through ThoughtPipeline
        - Return thought.response
        - Manage memory through MemoryManager
        - Manage knowledge through KnowledgeManager

    Methods:
        process(message): Process a message and return thought.response.
    """

    def __init__(self) -> None:
        from athena.providers.lmstudio import LMStudioProvider

        self.debug_manager = DebugManager()
        self.provider = LMStudioProvider()
        self.memory_manager = MemoryManager()
        self.knowledge_manager = KnowledgeManager(
            working_memory=self.memory_manager.working_memory,
            provider=self.provider,
            memory_manager=self.memory_manager
        )
        self.pipeline = ThoughtPipeline(
            self.memory_manager,
            self.knowledge_manager,
            self.provider,
        )
        self.history: list[str] = []  # Conversation history stored by the brain
        self._ensure_storage_dir()
        self._load_history()

    def _ensure_storage_dir(self) -> None:
        """Create the data directory if it does not exist."""
        _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _load_history(self) -> None:
        """Load conversation history from disk."""
        if not _HISTORY_PATH.exists():
            return
        try:
            with open(_HISTORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.history = data.get("history", [])
        except (json.JSONDecodeError, KeyError, TypeError):
            self.history = []

    def _save_history(self) -> None:
        """Save conversation history to disk."""
        temp_path = _HISTORY_PATH.with_suffix(".json.tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump({"history": self.history}, f, indent=2)
        temp_path.replace(_HISTORY_PATH)

    async def process(self, message: str) -> str:
        """
        Process a message through the Athena brain pipeline.

        Creates a Thought object, passes it through ThoughtPipeline,
        and returns the response from the thought.

        Args:
            message: The input message to process.

        Returns:
            The response string from the thought.
        """
        thought = Thought(user_input=message)
        
        # Copy current conversation history into the thought before processing
        thought.history = list(self.history)

        await self.pipeline.process(thought)
        response = thought.get_response()

        # Store the completed thought in debug manager
        self.debug_manager.set_last_thought(thought)

        # Append conversation turn to history after obtaining response
        self.history.append(f"User: {message}")
        self.history.append(f"Assistant: {response}")

        # Persist conversation history to disk
        self._save_history()

        # Store in episodic memory
        content = f"User:\n{message}\n\nAssistant:\n{response}"
        self.memory_manager.remember(content)

        return response
