class PromptBuilder:
    def build(self, thought) -> str:
        lines = [
            "You are Athena.",
            "",
            "You are a local-first AI cognitive operating system.",
            "",
            "Always answer truthfully.",
            "",
        ]

        if not thought.history:
            lines.append("Conversation:")
            lines.append("(None)")
        else:
            lines.append("Conversation:")
            for item in thought.history:
                lines.append(item)

        lines.append("")
        lines.append("User:")
        lines.append(thought.user_input)

        return "\n".join(lines)
