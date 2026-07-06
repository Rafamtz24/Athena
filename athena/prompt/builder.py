class PromptBuilder:
    """Renders a ReasoningContextPackage into a final prompt string.

    The PromptBuilder is a pure renderer. It does NOT perform budgeting.
    It receives a compiled ReasoningContextPackage from the Context Budget
    Manager and renders it deterministically.
    """

    # ── Cached section header templates ──────────────────────────────
    # These are constructed once at class definition time and reused
    # for every prompt build, avoiding repeated string list creation.
    _HEADER_SYSTEM_PROMPT_SUFFIX = ("",)
    _HEADER_USER_INPUT = (
        "====================",
        "",
        "User",
        "",
        "====================",
        "",
    )
    _HEADER_WORKING_MEMORY = (
        "====================",
        "",
        "Conversation",
        "",
        "====================",
        "",
    )
    _HEADER_SEMANTIC_MEMORY = (
        "====================",
        "",
        "Knowledge",
        "",
        "====================",
        "",
    )
    _HEADER_CANDIDATE_FACTS = (
        "====================",
        "",
        "Candidate Facts",
        "",
        "====================",
        "",
    )
    _HEADER_CHAT_HISTORY = (
        "====================",
        "",
        "Memory",
        "",
        "====================",
        "",
    )
    _HEADER_TOOL_PREFIX = (
        "====================",
        "",
    )
    _HEADER_TOOL_SUFFIX = (
        "",
        "====================",
        "",
    )
    _HEADER_GENERIC_PREFIX = (
        "====================",
        "",
    )
    _HEADER_GENERIC_SUFFIX = (
        "",
        "====================",
        "",
    )
    # Trailing newline appended after every content block
    _TRAILING_NEWLINE = ("",)

    def build(self, package) -> str:
        """Build a prompt from a ReasoningContextPackage.

        Args:
            package: A ReasoningContextPackage containing ordered ContextSource
                     objects that fit within the provider's context window.

        Returns:
            The rendered prompt string.
        """
        # Check if we received a ReasoningContextPackage
        if hasattr(package, 'sources'):
            return self._build_from_package(package)

        # Fallback: backward-compatible build from Thought object
        return self._build_from_thought(package)

    def _build_from_package(self, package) -> str:
        """Build prompt from a ReasoningContextPackage.

        Renders each ContextSource in priority order with section headers.
        PERFORMANCE: Section header tuples are cached as class-level
        constants to avoid repeated list creation on every build.
        """
        lines = []

        for source in package.sources:
            name = source.name
            content = source.content

            if not content:
                continue

            # Determine section header based on source name
            if name == "system_prompt":
                lines.append(content)
                lines.append("")
                continue
            elif name == "user_input":
                lines.extend(self._HEADER_USER_INPUT)
                lines.append(content)
                lines.extend(self._TRAILING_NEWLINE)
            elif name == "working_memory":
                lines.extend(self._HEADER_WORKING_MEMORY)
                lines.append(content)
                lines.extend(self._TRAILING_NEWLINE)
            elif name == "semantic_memory":
                lines.extend(self._HEADER_SEMANTIC_MEMORY)
                lines.append(content)
                lines.extend(self._TRAILING_NEWLINE)
            elif name == "candidate_facts":
                lines.extend(self._HEADER_CANDIDATE_FACTS)
                lines.append(content)
                lines.extend(self._TRAILING_NEWLINE)
            elif name.startswith("tool:"):
                tool_label = name[5:]  # Remove "tool:" prefix
                lines.extend(self._HEADER_TOOL_PREFIX)
                lines.append(f"System ({tool_label})")
                lines.extend(self._HEADER_TOOL_SUFFIX)
                lines.append(content)
                lines.extend(self._TRAILING_NEWLINE)
            elif name == "chat_history":
                lines.extend(self._HEADER_CHAT_HISTORY)
                lines.append(content)
                lines.extend(self._TRAILING_NEWLINE)
            else:
                # Generic section for any future source type
                lines.extend(self._HEADER_GENERIC_PREFIX)
                lines.append(name.replace("_", " ").title())
                lines.extend(self._HEADER_GENERIC_SUFFIX)
                lines.append(content)
                lines.extend(self._TRAILING_NEWLINE)

        return "\n".join(lines)

    def _build_from_thought(self, thought) -> str:
        """Backward-compatible build from a raw Thought object.

        Used when no ReasoningContextPackage is available (fallback).
        The system prompt is loaded from athena/prompts/reasoning.json.
        """
        from athena.prompt.loader import PromptLoader
        system_prompt = PromptLoader.get_system_prompt("reasoning")
        lines = [
            system_prompt,
            "",
            "====================",
            "",
            "Conversation",
            "",
            "====================",
        ]

        if not thought.history:
            lines.append("(None)")
        else:
            for item in thought.history:
                lines.append(item)

        lines.extend([
            "",
            "====================",
            "",
            "Memory",
            "",
            "====================",
            "",
        ])

        if not thought.memories:
            lines.append("(None)")
        else:
            for memory in thought.memories:
                lines.append(memory)

        lines.extend([
            "",
            "====================",
            "",
            "Candidate Facts",
            "",
            "====================",
            "",
        ])

        candidates = getattr(thought, 'candidates', None)
        if not candidates:
            lines.append("(None)")
        else:
            for candidate in thought.candidates:
                if hasattr(candidate, 'statement'):
                    conf = getattr(candidate, 'confidence', 0.0)
                    cat = getattr(candidate, 'category', '')
                    lines.append(f"{candidate.statement} (confidence={conf}, category={cat})")
                else:
                    lines.append(str(candidate))

        lines.extend([
            "====================",
            "",
            "Knowledge",
            "",
            "====================",
            "",
        ])

        if thought.knowledge is None:
            lines.append("(None)")
        else:
            lines.append(thought.knowledge)

        lines.extend([
            "====================",
            "",
            "Plan",
            "",
            "====================",
            "",
            "(None)",
            "",
        ])

        tool_context = getattr(thought, 'tool_context', None)
        if tool_context is not None and tool_context.content:
            lines.extend([
                "====================",
                "",
                f"System ({tool_context.tool_name})",
                "",
                "====================",
                "",
                tool_context.content,
                "",
            ])

        lines.extend([
            "====================",
            "",
            "User",
            "",
            "====================",
            "",
            thought.user_input,
            "",
            "====================",
        ])

        return "\n".join(lines)
