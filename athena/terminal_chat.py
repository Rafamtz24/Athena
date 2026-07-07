"""
Athena Terminal Chat Interface

Minimal interactive terminal interface for continuous conversation with Athena.

Usage:
    python -m athena.terminal_chat

Commands:
    /context size           Display current conversation context size
    /context size <value>   Update conversation context size immediately
    /book                   Enter reading mode (answer from a selected PDF)
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


def _reading_mode(brain: AthenaBrain) -> None:
    """Enter book reading mode: select a PDF, then answer questions from it.

    In this mode Athena's normal pipeline is bypassed entirely — no tools, no
    memory retrieval/injection, no knowledge extraction. Answers come only from
    the selected book's contents.
    """
    from athena.books.library import chunk_text, extract_text, list_books

    books = list_books()
    if not books:
        print(
            "\nNo books found. Place one or more PDF files in the 'books' "
            "folder and try again.\n"
        )
        return

    print("\n=== Reading Mode ===")
    print("Available books:")
    for index, book in enumerate(books, start=1):
        print(f"  {index}. {book.stem}")
    print("  0. Cancel")

    # ── Select a book ──
    selected = None
    while selected is None:
        try:
            choice = input("\nSelect a book by number: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if choice == "0" or choice.lower() in ("exit", "quit", "cancel"):
            print("Left reading mode.\n")
            return
        if choice.isdigit() and 1 <= int(choice) <= len(books):
            selected = books[int(choice) - 1]
        else:
            print("Invalid selection. Enter a listed number (or 0 to cancel).")

    # ── Load & chunk the book ──
    print(f"\nLoading '{selected.stem}'...")
    try:
        text = extract_text(selected)
    except Exception as exc:
        print(f"Failed to read the PDF: {exc}\n")
        return
    chunks = chunk_text(text)
    if not chunks:
        print(
            "Could not extract any text from this PDF. It may be a scanned "
            "(image-only) document.\n"
        )
        return

    print(f"Loaded '{selected.stem}' ({len(chunks)} passages).")
    print("Ask questions about this book. Type /exit to leave reading mode.\n")

    # ── Question loop ──
    while True:
        try:
            question = input(f"[{selected.stem}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("/exit", "/close", "/book", "exit", "quit"):
            break
        answer = brain.answer_from_book(chunks, question)
        print(f"\n{answer}\n")

    print("Left reading mode.\n")


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

        # /book — enter reading mode (separate, pipeline-free book QA)
        if stripped.lower() == "/book":
            _reading_mode(brain)
            continue

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
