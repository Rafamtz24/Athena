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
            "(None)",
            "",
            "====================",
            "",
            "Knowledge",
            "",
            "====================",
            "",
            "(None)",
            "",
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
