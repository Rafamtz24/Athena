"""
Athena Terminal Chat Interface

Minimal interactive terminal interface for continuous conversation with Athena.

Usage:
    python -m athena.terminal_chat

Commands:
    /context size           Display current conversation context size
    /context size <value>   Update conversation context size immediately
    exit, quit              Leave the chat

Exit by typing: exit or quit
"""

import asyncio

from athena.brain.brain import AthenaBrain
from athena.config.settings import get_settings


def _handle_command(user_input: str) -> str:
    """Handle terminal-only commands. Returns 'consumed' if handled,
    'pass_through' if the command should be processed by the brain,
    or 'unknown' if not recognized.
    """
    parts = user_input.strip().split()
    if not parts:
        return "unknown"

    command = parts[0].lower()

    # /context size — terminal-only command
    if command == "/context" and len(parts) >= 2 and parts[1].lower() == "size":
        if len(parts) == 2:
            # Display current value
            print(f"\nConversation context size: {get_settings().prompt.csize}\n")
            return "consumed"
        elif len(parts) == 3:
            # Update csize
            try:
                new_size = int(parts[2])
                if new_size < 0:
                    print("\nError: Context size must be a non-negative integer.\n")
                    return "consumed"
                get_settings().prompt.csize = new_size
                print(f"\nConversation context size: {get_settings().prompt.csize}\n")
            except ValueError:
                print("\nError: Context size must be a valid integer.\n")
            return "consumed"

    # /system — processed by AthenaBrain (generates tool context)
    if command == "/system":
        return "pass_through"

    return "unknown"


def main() -> None:
    brain = AthenaBrain()

    print("Athena Terminal Chat")
    print("Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        stripped = user_input.strip()
        if stripped.lower() in ("exit", "quit"):
            break

        # Handle terminal commands
        if stripped.startswith("/"):
            result = _handle_command(stripped)
            if result == "consumed":
                continue
            elif result == "pass_through":
                # Pass through to AthenaBrain for processing
                pass
            else:
                # Unknown command — still pass to brain for potential LLM handling
                pass

        response = asyncio.run(brain.process(stripped))
        print(f"\n{response}\n")


if __name__ == "__main__":
    main()
