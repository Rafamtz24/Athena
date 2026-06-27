class PromptBuilder:
    def build(self, thought) -> str:
        return (
            "You are Athena.\n"
            "\n"
            "You are a local-first AI cognitive operating system.\n"
            "\n"
            "Always answer truthfully.\n"
            "\n"
            "User:\n"
            "<user_input>\n"
            f"{thought.user_input}\n"
        )
