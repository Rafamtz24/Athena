"""Live token output from the model to the terminal.

Without this, an answer appears all at once after the model finishes — on a
35B model split across GPU and CPU that is a twenty-second wait staring at a
spinner. Streaming turns the same wait into visible progress.

The terminal registers a sink; providers call ``emit()`` as tokens arrive.
A module-level sink (rather than a callback threaded through brain, pipeline,
and engine) keeps the many layers between the terminal and the provider out of
the business of relaying output.

Only the answer call streams. The planner and learning-phase calls run through
the same providers but pass ``stream=False``, so their tokens never reach the
screen.

``kind`` is one of:

    "reasoning" — the model's chain-of-thought
    "answer"    — the reply itself
    "progress"  — prompt-evaluation progress, as a whole-number percentage
                  string. No text has been generated yet at this point; the
                  model is still reading the prompt, which on a large model
                  split across GPU and CPU is most of the wait.

The sink decides what each means on screen (the terminal hides reasoning unless
`/think show` is on, and turns progress into a spinner label rather than
output).
"""

import threading

_lock = threading.Lock()
_sink = None


def set_sink(fn) -> None:
    """Register the callable that receives (kind, text) as tokens arrive."""
    global _sink
    with _lock:
        _sink = fn


def clear_sink() -> None:
    """Unregister the sink. Providers fall back to returning whole answers."""
    global _sink
    with _lock:
        _sink = None


def active() -> bool:
    """True when something is listening for live tokens."""
    with _lock:
        return _sink is not None


def emit(kind: str, text: str) -> None:
    """Hand one delta to the sink. No-op when nothing is listening.

    A sink that raises must not take down the generation it is reporting on,
    so failures are swallowed: losing the live echo is recoverable, losing the
    answer is not.
    """
    if not text:
        return
    with _lock:
        sink = _sink
    if sink is None:
        return
    try:
        sink(kind, text)
    except Exception:
        pass
