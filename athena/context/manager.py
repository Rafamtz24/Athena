"""Context Budget Manager.

The Context Budget Manager is a core pipeline component responsible for
compiling all available context into context packages that fit within the
active model's context window.

Its responsibility is budgeting context, not modifying or summarizing it.

The architecture supports every future tool without requiring architectural
changes by using metadata on ContextSource objects rather than hardcoded
tool-specific rules.
"""

import sys
import traceback
from typing import Any, List, Optional

from athena.context.models import (
    ContextSource,
    LearningContextPackage,
    ReasoningContextPackage,
)
from athena.config.settings import get_settings


# ── Priority Constants ────────────────────────────────────────────────

# Authority order (high -> low): the user's current message wins, then
# confirmed durable facts (Semantic Memory), then the replayed conversation
# (Working Memory). Semantic ranks ABOVE Working Memory so a stored fact
# ("User's name is Rafael") outranks a stale conversational turn, while the
# current user input still overrides everything (recency preserved).
PRIORITY_USER_INPUT = 100
PRIORITY_SYSTEM_PROMPT = 95
PRIORITY_SEMANTIC_MEMORY = 90
PRIORITY_WORKING_MEMORY = 80
PRIORITY_TOOL_CONTEXT = 70
PRIORITY_CHAT_HISTORY = 60

# Sources with these priorities are NEVER trimmed
_NEVER_TRIM_PRIORITIES = {
    PRIORITY_USER_INPUT,
    PRIORITY_SYSTEM_PROMPT,
    PRIORITY_WORKING_MEMORY,
    PRIORITY_SEMANTIC_MEMORY,  # Not trimmed in this milestone
}


def _iter_tool_contexts(thought: Any) -> list:
    """Return all tool contexts on a thought.

    Prefers the multi-tool `tool_contexts` list (populated when tools are
    chained); falls back to the singular `tool_context` for backward
    compatibility. None entries are filtered out.
    """
    contexts = getattr(thought, 'tool_contexts', None)
    if contexts:
        return [c for c in contexts if c is not None]
    single = getattr(thought, 'tool_context', None)
    return [single] if single is not None else []


class ContextBudgetManager:
    """Compiles context sources into budgeted Reasoning and Learning packages.

    Input:
        - User Input
        - Working Memory
        - Semantic Memory
        - Chat History
        - Tool Context(s)

    Output:
        - Reasoning Context Package (guaranteed to fit within context window)
        - Learning Context Package (learning_visible sources only)

    The manager uses the active provider's token counting and context window
    to make deterministic budgeting decisions. It NEVER:
        - rewrites content
        - summarizes content
        - paraphrases content

    It may ONLY:
        - keep
        - remove
        - truncate
    """

    # ── Cached System Prompt ────────────────────────────────────────────
    # Static system prompt shared by every request.
    # Loaded from athena/prompts/reasoning.json via PromptLoader.
    # Defined at class level so it is loaded once and never recreated.
    _SYSTEM_PROMPT = None  # Loaded lazily on first access

    # ── Cached overhead text for WM budget calculation ──────────────────
    # Static text representing section headers/separators that appear
    # in every prompt. Cached to avoid rebuilding and recounting.
    _OVERHEAD_TEXT = (
        "\n====================\n\nConversation\n\n====================\n\n"
        "\n====================\n\nKnowledge\n\n====================\n\n"
        "\n====================\n\nUser\n\n====================\n\n"
    )

    def __init__(self, provider: Any) -> None:
        """Initialize the Context Budget Manager.

        Args:
            provider: An LLM provider instance implementing count_tokens()
                and get_context_window().
        """
        self.provider = provider
        self.settings = get_settings()
        # Per-compile token count cache: maps text -> token count.
        # Initialized at the start of each compile() and discarded after.
        self._token_cache: dict[str, int] = {}

    def compile(self, thought: Any) -> tuple:
        """Compile all available context into budgeted packages.

        Args:
            thought: The Thought object containing all context sources
                (user_input, history, memories, knowledge, tool_context,
                 candidates).

        Returns:
            A tuple of (ReasoningContextPackage, LearningContextPackage).
        """
        # ── PERFORMANCE: Initialize per-compile token cache ──
        # Clear the cache so previous compile's entries don't pollute
        # the current budget calculation.
        self._token_cache.clear()

        # Collect all context sources
        sources = self._collect_sources(thought)

        # Sort by priority descending
        sources.sort(key=lambda s: s.priority, reverse=True)

        # Get provider context window
        try:
            context_window = self.provider.get_context_window()
        except Exception:
            context_window = 4096  # Fallback default

        # Reserve generation budget (default: 25% of context window)
        gen_ratio = getattr(self.settings.budget, 'generation_reserve_ratio', 0.25)
        generation_budget = max(256, int(context_window * gen_ratio))
        prompt_budget = context_window - generation_budget

        # Budget the sources
        budgeted_sources, trimmed = self._budget_sources(sources, prompt_budget)

        # Build Reasoning Package
        reasoning_package = ReasoningContextPackage(
            sources=budgeted_sources,
            total_tokens=sum(
                self._count_tokens(s.content) for s in budgeted_sources
            ),
            generation_budget=generation_budget,
            context_window=context_window,
            trimmed_sources=trimmed,
        )

        # Build Learning Package
        learning_package = self._build_learning_package(
            thought, budgeted_sources
        )

        return reasoning_package, learning_package

    def compute_wm_budget(self, thought: Any) -> int:
        """Compute the Working Memory token budget for the current interaction.

        Counts ALL non-Working-Memory context sources and subtracts them
        from the prompt budget. The remainder is the maximum Working Memory
        can use without risking context overflow.

        This is called BEFORE compile() so Working Memory can prune itself
        before the final reasoning package is assembled.

        Args:
            thought: The Thought object containing all context sources.

        Returns:
            Maximum token budget for Working Memory (minimum 256).
        """
        # ── PERFORMANCE: Initialize per-compile token cache ──
        # compute_wm_budget runs before compile() in the pipeline,
        # so we initialize the cache here. compile() will reuse it.
        self._token_cache.clear()

        # Get provider context window
        try:
            context_window = self.provider.get_context_window()
        except Exception:
            context_window = 4096

        # Reserve generation budget
        gen_ratio = getattr(self.settings.budget, 'generation_reserve_ratio', 0.25)
        generation_budget = max(256, int(context_window * gen_ratio))
        prompt_budget = context_window - generation_budget

        # Count all non-WM sources
        non_wm_tokens = 0

        # 1. User Input
        non_wm_tokens += self._count_tokens(thought.user_input)

        # 2. System Prompt
        non_wm_tokens += self._count_tokens(self._build_system_prompt())

        # 3. Semantic Memory
        knowledge = getattr(thought, 'knowledge', None)
        if knowledge is not None and knowledge:
            content = str(knowledge) if not isinstance(knowledge, str) else knowledge
            non_wm_tokens += self._count_tokens(content)

        # 4. Tool Context(s)
        for tc in _iter_tool_contexts(thought):
            if tc.content:
                non_wm_tokens += self._count_tokens(tc.content)

        # 5. Chat History (from thought.memories)
        chat_history = getattr(thought, 'memories', []) or []
        if chat_history:
            chat_text = "\n".join(str(m) for m in chat_history)
            non_wm_tokens += self._count_tokens(chat_text)

        # 6. Candidates
        candidates = getattr(thought, 'candidates', None)
        if candidates:
            if isinstance(candidates, list):
                cand_lines = []
                for c in candidates:
                    if hasattr(c, 'statement'):
                        conf = getattr(c, 'confidence', 0.0)
                        cat = getattr(c, 'category', '')
                        cand_lines.append(
                            f"{c.statement} (confidence={conf}, category={cat})"
                        )
                    else:
                        cand_lines.append(str(c))
                cand_text = "\n".join(cand_lines)
            else:
                cand_text = str(candidates)
            non_wm_tokens += self._count_tokens(cand_text)

        # 7. Prompt overhead (section headers, separators, newlines)
        # PERFORMANCE: Static overhead text is cached via the per-compile
        # token cache, so this only calls count_tokens once per compile.
        overhead = self._count_tokens(self._OVERHEAD_TEXT)
        non_wm_tokens += overhead

        wm_budget = prompt_budget - non_wm_tokens
        return max(256, wm_budget)

        # Build Reasoning Package
        reasoning_package = ReasoningContextPackage(
            sources=budgeted_sources,
            total_tokens=sum(
                self._count_tokens(s.content) for s in budgeted_sources
            ),
            generation_budget=generation_budget,
            context_window=context_window,
            trimmed_sources=trimmed,
        )

        # Build Learning Package
        learning_package = self._build_learning_package(
            thought, budgeted_sources
        )

        return reasoning_package, learning_package

    # ── Source Collection ───────────────────────────────────────────

    def _collect_sources(self, thought: Any) -> List[ContextSource]:
        """Collect all context sources from the Thought object.

        Every source is represented as a ContextSource with metadata.
        No tool-specific hardcoding occurs here.
        """
        sources = []

        # 1. System Prompt (priority 95, never trimmed)
        system_prompt = self._build_system_prompt()
        sources.append(ContextSource(
            name="system_prompt",
            content=system_prompt,
            priority=PRIORITY_SYSTEM_PROMPT,
            learning_visible=False,
            truncatable=False,
        ))

        # 2. User Input (priority 100, never trimmed)
        sources.append(ContextSource(
            name="user_input",
            content=thought.user_input,
            priority=PRIORITY_USER_INPUT,
            learning_visible=False,
            truncatable=False,
        ))

        # 3. Working Memory / Conversation History (priority 90, never trimmed)
        history = getattr(thought, 'history', []) or []
        if history:
            if isinstance(history, list):
                history_text = "\n".join(str(h) for h in history)
            else:
                history_text = str(history)
            sources.append(ContextSource(
                name="working_memory",
                content=history_text,
                priority=PRIORITY_WORKING_MEMORY,
                learning_visible=True,
                truncatable=False,
            ))

        # 4. Semantic Memory (priority 80, not trimmed in this milestone)
        knowledge = getattr(thought, 'knowledge', None)
        if knowledge is not None and knowledge:
            content = str(knowledge) if not isinstance(knowledge, str) else knowledge
            sources.append(ContextSource(
                name="semantic_memory",
                content=content,
                priority=PRIORITY_SEMANTIC_MEMORY,
                learning_visible=True,
                truncatable=False,
            ))

        # 5. Tool Context(s) (priority 70, may be trimmed) — one source per tool
        for tc in _iter_tool_contexts(thought):
            if tc.content:
                sources.append(ContextSource(
                    name=f"tool:{tc.tool_name}",
                    content=tc.content,
                    priority=getattr(tc, 'priority', PRIORITY_TOOL_CONTEXT),
                    learning_visible=getattr(tc, 'learning_visible', True),
                    truncatable=True,
                ))

        # 6. Chat History (priority 60, may be trimmed)
        chat_history = getattr(thought, 'memories', []) or []
        if chat_history:
            if isinstance(chat_history, list):
                chat_text = "\n".join(str(m) for m in chat_history)
            else:
                chat_text = str(chat_history)
            sources.append(ContextSource(
                name="chat_history",
                content=chat_text,
                priority=PRIORITY_CHAT_HISTORY,
                learning_visible=True,
                truncatable=True,
            ))

        # 7. Candidates (for prompt, priority between working memory and tool)
        candidates = getattr(thought, 'candidates', None)
        if candidates:
            if isinstance(candidates, list):
                cand_lines = []
                for c in candidates:
                    if hasattr(c, 'statement'):
                        conf = getattr(c, 'confidence', 0.0)
                        cat = getattr(c, 'category', '')
                        cand_lines.append(
                            f"{c.statement} (confidence={conf}, category={cat})"
                        )
                    else:
                        cand_lines.append(str(c))
                cand_text = "\n".join(cand_lines)
            else:
                cand_text = str(candidates)
            sources.append(ContextSource(
                name="candidate_facts",
                content=cand_text,
                priority=75,  # Between working memory (80) and tool (70)
                learning_visible=False,
                truncatable=True,
            ))

        return sources

    # ── Budgeting ──────────────────────────────────────────────────

    def _budget_sources(
        self,
        sources: List[ContextSource],
        prompt_budget: int,
    ) -> tuple:
        """Budget sources to fit within the prompt budget.

        Algorithm:
        1. Calculate total tokens for all sources.
        2. If total fits within budget, return all sources (no trimming).
        3. If total exceeds budget, trim lowest-priority truncatable sources.
        4. Never-trim sources (priority in _NEVER_TRIM_PRIORITIES) are always kept.

        Args:
            sources: List of ContextSource objects sorted by priority desc.
            prompt_budget: Maximum token budget for the prompt.

        Returns:
            Tuple of (budgeted_sources, trimmed_source_names).
        """
        trimmed = []

        # Calculate token counts for each source
        source_tokens = []
        for source in sources:
            tokens = self._count_tokens(source.content)
            source_tokens.append((source, tokens))

        total = sum(t for _, t in source_tokens)

        # If everything fits, return as-is
        if total <= prompt_budget:
            return list(sources), trimmed

        # Budget exceeded — remove lowest-priority truncatable sources
        budgeted = []
        for source, tokens in source_tokens:
            if source.priority in _NEVER_TRIM_PRIORITIES or not source.truncatable:
                # Never trim — always include
                budgeted.append((source, tokens))
            elif total <= prompt_budget:
                # After removals, we now fit — include remaining
                budgeted.append((source, tokens))
            else:
                # Remove this source
                trimmed.append(source.name)
                total -= tokens

        # If still over budget after full removals, try truncation
        if total > prompt_budget:
            budgeted = self._truncate_sources(budgeted, prompt_budget, trimmed)

        return [s for s, _ in budgeted], trimmed

    def _truncate_sources(
        self,
        budgeted: List[tuple],
        prompt_budget: int,
        trimmed: List[str],
    ) -> List[tuple]:
        """Truncate sources from lowest priority upward until within budget.

        Truncation removes the end of each source's content (oldest entries)
        to fit within the remaining budget.
        """
        # Sort by priority ascending (lowest first) for truncation
        budgeted.sort(key=lambda x: x[0].priority)

        total = sum(t for _, t in budgeted)
        if total <= prompt_budget:
            return budgeted

        result = []
        for source, tokens in budgeted:
            if total <= prompt_budget:
                result.append((source, tokens))
                continue

            if source.priority in _NEVER_TRIM_PRIORITIES:
                result.append((source, tokens))
                continue

            # Try to truncate this source
            current_total = total - tokens
            target = prompt_budget - current_total
            target = max(target, 0)
            target = max(target, int(tokens * 0.1))  # Keep at least 10%

            if target < tokens:
                truncated_content = self._truncate_content(
                    source.content, target
                )
                new_tokens = self._count_tokens(truncated_content)
                source.content = truncated_content
                result.append((source, new_tokens))
                total = current_total + new_tokens
                trimmed.append(f"{source.name} (truncated)")
            else:
                result.append((source, tokens))

        return result

    @staticmethod
    def _truncate_content(content: str, target_tokens: int) -> str:
        """Truncate content from the end to fit within target tokens.

        Uses line-based truncation (deterministic, keeps beginning).
        Rough estimate: ~4 chars per token.
        """
        max_chars = target_tokens * 4
        if len(content) <= max_chars:
            return content

        lines = content.split("\n")
        result = []
        char_count = 0
        for line in lines:
            line_len = len(line) + 1  # +1 for newline
            if char_count + line_len > max_chars:
                break
            result.append(line)
            char_count += line_len

        truncated = "\n".join(result)
        if truncated != content:
            truncated += "\n...(context truncated to fit budget)"
        return truncated

    # ── Learning Package ────────────────────────────────────────────

    def _build_learning_package(
        self,
        thought: Any,
        budgeted_sources: List[ContextSource],
    ) -> LearningContextPackage:
        """Build a Learning Package from budgeted sources.

        Only includes sources where learning_visible=True.
        """
        learning_sources = [
            s for s in budgeted_sources if s.learning_visible
        ]

        # Build conversation string from user input + response
        parts = list(getattr(thought, 'history', []) or [])
        parts.append(f"User: {thought.user_input}")
        response = getattr(thought, 'response', None)
        if response:
            parts.append(f"Assistant: {response}")
        conversation = "\n".join(parts) if parts else thought.user_input

        # Extract System Snapshot content for learning. The Knowledge Extractor
        # frames tool_context_content as a System Snapshot (with hardware-fact
        # rules), so ONLY the system tool's content belongs here — web or other
        # tool output must not be mislabeled as durable hardware facts.
        tool_context_content = ""
        for source in learning_sources:
            if source.name == "tool:system":
                tool_context_content = source.content
                break

        return LearningContextPackage(
            sources=learning_sources,
            conversation=conversation,
            tool_context_content=tool_context_content,
        )

    # ── System Prompt ───────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the reasoning package.

        PERFORMANCE: Loaded once via PromptLoader, cached at class level.
        The system prompt never changes between interactions unless the
        cache is explicitly cleared and the prompt file is edited.
        """
        if self.__class__._SYSTEM_PROMPT is None:
            from athena.prompt.loader import PromptLoader
            profile = PromptLoader.load("reasoning")
            self.__class__._SYSTEM_PROMPT = profile.system_prompt
        return self.__class__._SYSTEM_PROMPT

    # ── Token Counting ──────────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the active provider, with per-compile caching.

        Args:
            text: The text to count tokens for.

        Returns:
            Token count, or character-based estimate if provider fails.

        PERFORMANCE: Token counts are cached in a per-compile dict so
        that identical content counted multiple times (e.g., when computing
        WM budget and then again during source budgeting) avoids redundant
        provider calls.
        """
        if not text:
            return 0

        # Check per-compile cache first
        cached = self._token_cache.get(text)
        if cached is not None:
            return cached

        try:
            count = self.provider.count_tokens(text)
        except Exception:
            # Fallback: character-based estimate
            count = len(text) // 4

        self._token_cache[text] = count
        return count