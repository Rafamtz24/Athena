"""
Athena Terminal Chat Interface

Minimal interactive terminal interface for continuous conversation with Athena.

Usage:
    python -m athena.terminal_chat

Exit by typing: exit or quit
"""

import asyncio

from athena.brain.brain import AthenaBrain


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

        if user_input.strip().lower() in ("exit", "quit"):
            break

        response = asyncio.run(brain.process(user_input))
        print(f"\n{response}\n")


if __name__ == "__main__":
    main()
