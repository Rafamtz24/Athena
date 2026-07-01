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

# Storage locations
_WORKING_MEM_PATH = Path(get_settings().storage.working_memory_path)
_CHAT_HISTORY_PATH = Path(get_settings().storage.chat_history_path)


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
        self.history: list[str] = []  # Active conversation context (may be pruned)
        self._ensure_storage_dir()
        self._load_working_memory()
        self._load_chat_history()

    def _ensure_storage_dir(self) -> None:
        """Create the data directory if it does not exist."""
        _WORKING_MEM_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _load_working_memory(self) -> None:
        """Load active conversation context from working_memory.json."""
        if not _WORKING_MEM_PATH.exists():
            return
        try:
            with open(_WORKING_MEM_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.history = data.get("history", [])
        except (json.JSONDecodeError, KeyError, TypeError):
            self.history = []

    def _load_chat_history(self) -> None:
        """Load chat_history.json or create it if missing."""
        if not _CHAT_HISTORY_PATH.exists():
            self._write_chat_history([])

    def _write_chat_history(self, entries: list) -> None:
        """Write chat_history.json atomically."""
        temp = _CHAT_HISTORY_PATH.with_suffix(".json.tmp")
        with open(temp, "w", encoding="utf-8") as f:
            json.dump({"history": entries}, f, indent=2)
        temp.replace(_CHAT_HISTORY_PATH)

    def _read_chat_history(self) -> list:
        """Read all entries from chat_history.json."""
        if not _CHAT_HISTORY_PATH.exists():
            return []
        try:
            with open(_CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("history", [])
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    def _append_to_chat_history(self, user_msg: str, assistant_msg: str) -> None:
        """Append an interaction to the permanent chat_history.json."""
        entries = self._read_chat_history()
        entries.append(f"User: {user_msg}")
        entries.append(f"Assistant: {assistant_msg}")
        self._write_chat_history(entries)

    def _prune_to_budget(self) -> None:
        """Remove oldest entries from working_memory.json if they exceed csize.

        Uses the same token estimation as PromptBuilder: len(text) // 4.
        Only modifies the in-memory self.history and working_memory.json.
        chat_history.json is NEVER modified.
        """
        csize = get_settings().prompt.csize
        if csize <= 0:
            self.history = []
            self._write_working_memory()
            return

        def estimate_tokens(text: str) -> int:
            return len(text) // 4

        total = 0
        for i in range(len(self.history) - 1, -1, -1):
            total += estimate_tokens(self.history[i])
            if total > csize:
                # Entries 0..i exceed budget; keep i+1..end
                self.history = list(self.history[i + 1:])
                self._write_working_memory()
                return

        # All entries fit — no change needed
        self._write_working_memory()

    def _write_working_memory(self) -> None:
        """Write working_memory.json atomically."""
        temp_path = _WORKING_MEM_PATH.with_suffix(".json.tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump({"history": self.history}, f, indent=2)
        temp_path.replace(_WORKING_MEM_PATH)

    def _save_history(self) -> None:
        """Save working memory to disk (backward-compatible alias)."""
        self._write_working_memory()

    async def process(self, message: str) -> str:
        """
        Process a message through the Athena brain pipeline.

        Creates a Thought object, passes it through ThoughtPipeline,
        and returns the response from the thought.

        After each interaction:
        1. Appends turn to chat_history.json (permanent transcript)
        2. Appends turn to working_memory.json (active context)
        3. Prunes working_memory.json to fit within csize
        4. Stores in episodic memory
        """
        thought = Thought(user_input=message)
        
        # Copy current conversation history into the thought before processing
        thought.history = list(self.history)

        await self.pipeline.process(thought)
        response = thought.get_response()

        # Store the completed thought in debug manager
        self.debug_manager.set_last_thought(thought)

        # Append to permanent chat history (always, never pruned)
        self._append_to_chat_history(message, response)

        # Append to active conversation context
        self.history.append(f"User: {message}")
        self.history.append(f"Assistant: {response}")

        # Prune conversation context to fit within csize, then save
        self._prune_to_budget()

        # Store in episodic memory
        content = f"User:\n{message}\n\nAssistant:\n{response}"
        self.memory_manager.remember(content)

        return response
