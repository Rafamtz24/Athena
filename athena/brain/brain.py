"""
Athena Brain Module

Core orchestration component that receives requests, reasons, delegates to providers,
and returns responses.
"""

import json
import sys
import traceback
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
        from athena.providers import ProviderFactory

        self.debug_manager = DebugManager()
        self.provider = ProviderFactory.create()
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
        self._reset_working_memory()
        self._load_chat_history()

    def answer_from_book(self, chunks: list, question: str) -> str:
        """Answer a question grounded strictly in a selected book (reading mode).

        This is a SEPARATE path from process(): it does not run the Thought
        pipeline, and therefore uses no tools, performs no memory retrieval or
        injection, and extracts no knowledge. Only the book's contents inform
        the answer.
        """
        from athena.books.library import answer_from_book as _answer_from_book

        try:
            return _answer_from_book(self.provider, chunks, question)
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            print(f"\n[BOOK] answer_from_book failed:\n{tb_str}")
            return "I'm sorry, I ran into a problem reading from this book."

    def _ensure_storage_dir(self) -> None:
        """Create the data directory if it does not exist."""
        _WORKING_MEM_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _reset_working_memory(self) -> None:
        """Start each session with empty Working Memory.

        Working Memory is the sliding window of the CURRENT conversation, not
        long-term storage — durable facts live in Semantic Memory, which
        persists and is retrieved during reasoning. Persisting the conversation
        window across sessions let stale turns (e.g. an old "my name is
        TestUser") replay and override confirmed facts, so each new session
        starts fresh. The cleared window is written to disk immediately so the
        persisted file reflects the current session.
        """
        self.history = []
        self._write_working_memory()

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
        """Remove oldest entries from working_memory.json if they exceed budget.

        Delegates to WorkingMemory.prune() which handles the sliding-window
        eviction logic. The Context Budget Manager computes the dynamic budget
        during the pipeline; this method applies it to persist working_memory.json.
        """
        # Use the provider-computed budget when available, else fall back to
        # the static csize setting (backward compatible).
        try:
            context_window = self.provider.get_context_window()
            gen_ratio = getattr(get_settings().budget, 'generation_reserve_ratio', 0.25)
            gen_budget = max(256, int(context_window * gen_ratio))
            prompt_budget = context_window - gen_budget
            # Use a generous default: WM gets 40% of prompt budget as fallback
            csize = max(1024, int(prompt_budget * 0.40) * 4)
        except Exception:
            csize = get_settings().prompt.csize

        if csize <= 0:
            self.history = []
            self._write_working_memory()
            return

        # Delegate to WorkingMemory.prune() for the eviction logic
        self.memory_manager.working_memory.prune(
            max_tokens=csize // 4,
            entries=self.history,
        )

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

        The Tool Planner (within the pipeline) is responsible for detecting
        whether a tool (e.g., /system, web search) is needed. The Tool
        Router is the sole component that executes tools.

        Every request is COMPLETELY INDEPENDENT:
        - A new Thought is created per request (fresh planner_decision, tool_context)
        - The pipeline has comprehensive exception isolation
        - A failure in one request NEVER affects subsequent requests

        After each interaction:
        1. Appends turn to chat_history.json (permanent transcript)
        2. Appends turn to working_memory.json (active context)
        3. Prunes working_memory.json to fit within csize
        4. Stores in episodic memory
        """
        thought = Thought(user_input=message)
        
        # Copy current conversation history into the thought before processing
        thought.history = list(self.history)

        # Process through the pipeline with full exception isolation
        try:
            await self.pipeline.process(thought)
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            print(f"\n[BRAIN TRACE] pipeline.process() threw an unhandled exception:")
            print(f"[BRAIN TRACE] Exception type: {exc_type.__name__}")
            print(f"[BRAIN TRACE] Exception message: {exc_value}")
            print(f"[BRAIN TRACE] Full traceback:\n{tb_str}")
            # Absolute last-resort guard: if pipeline itself throws an
            # unhandled exception, ensure we still produce a valid response.
            if thought.get_response() is None:
                thought.set_response(
                    "I'm sorry, I'm currently unable to process your request."
                )

        response = thought.get_response()

        # Store the completed thought in debug manager
        self.debug_manager.set_last_thought(thought)

        # Always perform post-request bookkeeping so state never leaks.
        # Even if pipeline.process() raised, we still save history and
        # clear working memory to ensure the NEXT request starts fresh.

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

        # Ensure WorkingMemory is always clean for the next request
        try:
            self.memory_manager.clear_working()
        except Exception:
            pass

        return response
