class PromptBuilder:
    def build(self, thought) -> str:
        lines = [
            "You are Athena.",
            "",
            "You are a local-first AI cognitive operating system.",
            "",
            "Always answer truthfully.",
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
                # Format: statement (confidence=X, category=Y)
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

        # ── Tool Context (optional, injected by native tools like /system) ──
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
