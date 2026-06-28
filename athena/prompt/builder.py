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
            for item in thought.history[-10:]:
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
