"""
Athena Terminal Chat Interface

Minimal interactive terminal interface for continuous conversation with Athena.

Usage:
    python -m athena.terminal_chat

Commands:
    /help                   List the available commands
    /context size           Display current conversation context size
    /context size <value>   Update conversation context size immediately
    /think [on|off]         Toggle thinking-model reasoning (default on)
    /learn [on|off]         Toggle post-answer memory learning (default on)
    /system                 Report a system snapshot (CPU, memory, GPU)
    /book                   Enter reading mode (answer from a selected PDF)
    /tarot                  Enter tarot mode (a random draw and reading)
    /serve [port]           Serve the reasoning model over an OpenAI-compatible
                            API for external clients (e.g. Open WebUI)
    exit, quit              Leave the chat

Exit by typing: exit or quit
"""

import asyncio
import itertools
import sys
import threading

from athena.brain.brain import AthenaBrain
from athena.config.settings import get_settings


class _LearningIndicator:
    """Animated 'Learning . .. ...' shown while the post-answer learning phase runs.

    Runs on a daemon thread and rewrites a single terminal line, so it never
    scrolls. ``start()`` is a no-op-safe to pair with ``stop()`` even if the
    phase is skipped.
    """

    def __init__(self, label: str = "Learning") -> None:
        self.label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        for dots in itertools.cycle([".", "..", "..."]):
            if self._stop.is_set():
                break
            # Trailing spaces clear leftovers from the previous (longer) frame.
            sys.stdout.write(f"\r{self.label}{dots}   ")
            sys.stdout.flush()
            if self._stop.wait(0.4):
                break

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1)
        self._thread = None
        # Erase the indicator line so the next prompt starts clean.
        sys.stdout.write("\r" + " " * (len(self.label) + 6) + "\r")
        sys.stdout.flush()


def _persist_pref(key: str, value) -> None:
    """Save a runtime preference so it survives restarts (best-effort)."""
    try:
        from athena.config.persistence import set_pref

        set_pref(key, value)
    except Exception:
        # Persistence is a convenience; never let a write error break the command.
        pass


_HELP_TEXT = """
Athena commands:
  /help                 Show this list of commands
  /think [on|off]       Toggle the model's visible reasoning (default on)
  /learn [on|off]       Toggle memory learning after each answer (default on)
  /context size [n]     Show, or set, the conversation context size
  /system               Report a system snapshot (CPU, memory, GPU)
  /book                 Reading mode - answer from a selected PDF
  /tarot                Tarot mode - draw and interpret a spread
  /serve [port]         Serve the reasoning model over an OpenAI-compatible
                        API for external clients like Open WebUI (default port 8080)
  exit, quit            Leave Athena
"""


def _print_help() -> None:
    """Print the list of available terminal commands."""
    print(_HELP_TEXT)


def _handle_command(user_input: str) -> str:
    """Handle terminal-only commands. Returns 'consumed' if handled,
    'pass_through' if the command should be processed by the brain,
    or 'unknown' if not recognized.
    """
    parts = user_input.strip().split()
    if not parts:
        return "unknown"

    command = parts[0].lower()

    # /help — list the available commands
    if command == "/help":
        _print_help()
        return "consumed"

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

    # /think [on|off] — toggle whether thinking models emit reasoning
    if command == "/think":
        provider_settings = get_settings().provider
        if len(parts) == 1:
            state = "on" if provider_settings.thinking_enabled else "off"
            print(f"\nThinking mode is {state}. Use '/think on' or '/think off'.\n")
            return "consumed"
        arg = parts[1].lower()
        if arg == "on":
            provider_settings.thinking_enabled = True
            _persist_pref("thinking_enabled", True)
            print("\nThinking mode: on.\n")
        elif arg == "off":
            provider_settings.thinking_enabled = False
            _persist_pref("thinking_enabled", False)
            print("\nThinking mode: off (faster; model answers directly).\n")
        else:
            print("\nUsage: /think on | /think off\n")
        return "consumed"

    # /learn [on|off] (aka /learning) — toggle the post-answer learning phase
    if command in ("/learn", "/learning"):
        learning_settings = get_settings().learning
        if len(parts) == 1:
            state = "on" if learning_settings.enabled else "off"
            print(f"\nLearning is {state}. Use '/learn on' or '/learn off'.\n")
            return "consumed"
        arg = parts[1].lower()
        if arg == "on":
            learning_settings.enabled = True
            _persist_pref("learning_enabled", True)
            print("\nLearning: on (Athena updates memory after each answer).\n")
        elif arg == "off":
            learning_settings.enabled = False
            _persist_pref("learning_enabled", False)
            print("\nLearning: off (faster replies; memory is not updated).\n")
        else:
            print("\nUsage: /learn on | /learn off\n")
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


def _tarot_mode(brain: AthenaBrain) -> None:
    """Enter tarot mode: ask for a question and spread, draw cards, and read them.

    Like reading mode, Athena's normal pipeline is bypassed entirely — no tools,
    no web search, no memory retrieval/injection, no knowledge extraction. The
    cards are drawn by a system random generator (OS entropy) BEFORE any
    interpretation, so the model cannot handpick them; it only reads the draw.
    """
    from athena.tarot.reading import SPREADS_REFERENCE, draw_for_spread, format_draws

    # Reversed cards on by default; toggled with /r during spread selection and
    # remembered across readings for this session of tarot mode.
    reversals = True

    def deliver(question: str, reading: dict) -> None:
        """Show the drawn cards, then Athena's interpretation of them."""
        print("\nThe cards drawn for you:\n")
        print(format_draws(reading))
        print("\nReading the cards...\n")
        print(f"{brain.tarot_reading(question, reading)}\n")

    def new_reading() -> bool:
        """Run a full reading: question, spread choice, draw, interpret.

        Returns True if a reading was delivered, or False if the user cancelled
        at the spread menu (which should exit tarot mode).
        """
        nonlocal reversals
        try:
            question = input(
                "Ask a question, or leave blank for a general reading:\n> "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return False

        print("\n" + SPREADS_REFERENCE)
        spread_id = None
        while spread_id is None:
            state = "on" if reversals else "off"
            try:
                choice = input(
                    f"\nChoose a spread (1-6), /r to turn reversed cards "
                    f"{'off' if reversals else 'on'} (now {state}), or 0 to cancel: "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return False
            lowered = choice.lower()
            if lowered in ("/r", "r"):
                reversals = not reversals
                print(f"Reversed cards: {'on' if reversals else 'off'}.")
                continue
            if choice == "0" or lowered in ("exit", "quit", "cancel"):
                return False
            if choice.isdigit() and 1 <= int(choice) <= 6:
                spread_id = int(choice)
            else:
                print("Invalid selection. Enter 1-6, /r, or 0 to cancel.")

        try:
            reading = draw_for_spread(spread_id, allow_reversals=reversals)
        except Exception as exc:
            print(f"\nCould not draw the cards: {exc}\n")
            return False
        deliver(question, reading)
        return True

    print("\n=== Tarot Mode ===")

    if not new_reading():
        print("Left tarot mode.\n")
        return

    # ── Follow-up loop ──
    # Any text is a one-card follow-up question (blank = a general one-card
    # reading); "tarot" starts a new reading; "exit" leaves tarot mode.
    while True:
        try:
            text = input(
                '\nType a follow-up question for a one-card reading (blank for a '
                'general one-card reading), "tarot" for a new reading, or "exit" '
                "to leave tarot mode:\n> "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if text.lower() in ("exit", "quit"):
            break
        if text.lower() == "tarot":
            if not new_reading():
                break
            continue

        # Otherwise, treat the input as a one-card follow-up question.
        try:
            one_card = draw_for_spread(1, allow_reversals=reversals)
        except Exception as exc:
            print(f"\nCould not draw the card: {exc}\n")
            continue
        deliver(text, one_card)

    print("Left tarot mode.\n")


def _serve_mode(brain: AthenaBrain, port: int = 8080) -> None:
    """Serve the reasoning model over an OpenAI-compatible HTTP API.

    Blocks the terminal (like /book and /tarot) and hands the resident
    reasoning model to a local HTTP server that external clients such as Open
    WebUI can connect to. The Thought pipeline, tools, memory, and learning are
    all bypassed — this is pure model inference.

    A dedicated learning model, if one is loaded, is unloaded first to free the
    memory it holds (it is idle while only the reasoning model is served), then
    reloaded when serving stops. When learning simply falls back to the
    reasoning model, there is nothing separate to unload.
    """
    try:
        import uvicorn
    except ImportError:
        print(
            "\nServe mode needs 'uvicorn' and 'fastapi'. Install them with:\n"
            "    pip install fastapi uvicorn\n"
        )
        return

    from athena.serve import build_app

    # Serve mode wraps the in-process GGUF model, which only the llamacpp
    # provider exposes. Other providers (e.g. LM Studio) already run their own
    # server, so there is nothing here for us to wrap.
    if not hasattr(brain.provider, "model"):
        print(
            f"\nServe mode requires the local 'llamacpp' provider. The current "
            f"provider ('{get_settings().provider.provider}') can't be served "
            f"this way.\n"
        )
        return

    # ── Free a dedicated learning model while we serve (reasoning only) ──
    freed_provider = None
    learning = brain.learning_provider
    reasoning = brain.provider
    separate_learning = (
        learning is not reasoning
        and getattr(learning, "model_path", None) != getattr(reasoning, "model_path", None)
    )
    if separate_learning and hasattr(learning, "unload"):
        if learning.unload():
            freed_provider = learning
            print("\nUnloaded the learning model to reserve memory for serving.")

    host = "127.0.0.1"
    app = build_app(reasoning)

    print("\n=== Serve Mode ===")
    print(f"Serving '{reasoning.model_name}' at http://{host}:{port}/v1")
    print("Point Open WebUI (or any OpenAI-compatible client) at that base URL;")
    print("no API key is required. Press Ctrl+C to stop serving.\n")

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        pass
    finally:
        if freed_provider is not None:
            freed_provider.reload()
        print("\nStopped serving. Back to the terminal.\n")


def main() -> None:
    # Tarot card art uses Unicode glyphs; make stdout tolerant so printing it
    # never crashes on a legacy code page (e.g. Windows cp1252) when output is
    # redirected. A real console already prints Unicode via the Windows API.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    # First-run onboarding: if no reasoning model is installed yet, show a
    # friendly setup guide (what a .gguf is, where to get one, what fits this
    # machine) instead of crashing with a FileNotFoundError traceback.
    from athena.onboarding import (
        ensure_model_folders,
        has_reasoning_model,
        print_first_run_guide,
    )

    ensure_model_folders()
    if not has_reasoning_model():
        print_first_run_guide()
        # Keep the window open for double-click launches so the guide is
        # readable; tolerate a closed stdin (piped/automated runs).
        try:
            input("\nPress Enter to close...")
        except (EOFError, KeyboardInterrupt):
            pass
        return

    brain = AthenaBrain()

    print("Welcome to Athena.")
    print("Type /help for a list of commands, or 'exit' to leave.\n")

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

        # /tarot — enter tarot mode (separate, pipeline-free card reading)
        if stripped.lower() == "/tarot":
            _tarot_mode(brain)
            continue

        # /serve [port] — expose the reasoning model over an OpenAI-compatible API
        if stripped.lower() == "/serve" or stripped.lower().startswith("/serve "):
            parts = stripped.split()
            port = 8080
            if len(parts) >= 2:
                try:
                    port = int(parts[1])
                except ValueError:
                    print("\nUsage: /serve [port]  (port must be a number)\n")
                    continue
            _serve_mode(brain, port)
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

        # Show the answer as soon as it's ready, then animate "Learning…" while
        # the slower learning phase (extra model calls) finishes in the pipeline.
        indicator = _LearningIndicator()

        def on_answer(answer: str) -> None:
            print(f"\n{answer}\n")
            if get_settings().learning.enabled:
                indicator.start()

        asyncio.run(brain.process(stripped, on_answer=on_answer))
        indicator.stop()


if __name__ == "__main__":
    main()
