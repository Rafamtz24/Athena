"""Capture of <think>…</think> reasoning traces for `/think show`.

Providers strip reasoning from every answer, so the trace is otherwise lost.
When the user turns on `/think show`, providers hand each trace to ``record()``
here and the terminal drains the buffer just before printing the answer.

A module-level buffer (rather than provider state) is what makes this work
across the several provider calls a single turn makes — planning, answering,
and any tool follow-ups each contribute a trace, in the order they ran.

Capture is off unless ``provider.show_thinking`` is set, so the buffer stays
empty and costs nothing on the default path.
"""

import threading

_lock = threading.Lock()
_traces: list[str] = []

# Paths that generate without draining (serve mode, book and tarot modes) would
# otherwise grow this list for as long as the session runs. A turn contributes
# a handful of traces, so keeping the most recent few loses nothing real.
_MAX_TRACES = 16


def capture_enabled() -> bool:
    """True when the user has asked to see reasoning (`/think show`)."""
    from athena.config.settings import get_settings

    return bool(getattr(get_settings().provider, "show_thinking", False))


def record(trace: str) -> None:
    """Buffer one reasoning trace. No-op when capture is off or trace is empty."""
    if not trace or not trace.strip():
        return
    if not capture_enabled():
        return
    with _lock:
        _traces.append(trace.strip())
        del _traces[:-_MAX_TRACES]


def drain() -> list[str]:
    """Return the buffered traces and empty the buffer."""
    with _lock:
        traces = list(_traces)
        _traces.clear()
    return traces


def clear() -> None:
    """Discard anything buffered. Called at the start of each turn so a turn
    never shows reasoning left over from the previous one's learning phase."""
    with _lock:
        _traces.clear()
