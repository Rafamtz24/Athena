"""
Tests for `/think show` — surfacing a thinking model's reasoning trace.

Two paths have to work, because the two local providers deliver reasoning
differently:

  Tags in the content — the in-process provider returns the raw completion, so
  <think>…</think> has to be parsed out of it, both in one piece and as a
  stream of deltas where a tag can straddle two chunks.

  A separate field — llama-server parses reasoning out itself and returns it as
  `reasoning_content`, leaving no tags behind. Reading only the content is what
  made `/think show` display nothing on the default provider.

Also covers the answer streaming that carries the reasoning: the answer call
streams, and the planner and learning calls stay silent.
"""
import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from athena.config.settings import get_settings
from athena.providers import reasoning_trace, streaming
from athena.providers.llamacpp import (
    ReasoningStream,
    _split_reasoning,
    _strip_reasoning,
)
from athena.providers.llamaserver import LlamaServerProvider


def _capture():
    """Register a sink and return the list it appends (kind, text) into."""
    emitted = []
    streaming.set_sink(lambda kind, text: emitted.append((kind, text)))
    return emitted


def _run_stream(deltas, starts_inside=False):
    emitted = []
    stream = ReasoningStream(lambda k, t: emitted.append((k, t)), starts_inside)
    for delta in deltas:
        stream.feed(delta)
    return stream.finish(), emitted


# --- Splitting reasoning out of a complete response -------------------------

def test_split_returns_both_halves():
    assert _split_reasoning("<think>weighing it</think>Answer.") == (
        "Answer.",
        "weighing it",
    )


def test_split_handles_template_opened_tag():
    # Qwen templates prefill "<think>" into the prompt, so the completion has
    # only the closing tag and everything before it is reasoning.
    assert _split_reasoning("weighing it</think>Answer.") == ("Answer.", "weighing it")


def test_split_handles_truncated_reasoning():
    # Generation cut off mid-thought: there is reasoning but no answer.
    assert _split_reasoning("<think>weighing i") == ("", "weighing i")


def test_split_leaves_untagged_output_alone():
    assert _split_reasoning("Just an answer.") == ("Just an answer.", "")


def test_strip_records_trace_only_when_showing():
    settings = get_settings().provider
    original = settings.show_thinking
    try:
        settings.show_thinking = False
        reasoning_trace.clear()
        assert _strip_reasoning("<think>hidden</think>Hi.") == "Hi."
        assert reasoning_trace.drain() == []

        settings.show_thinking = True
        assert _strip_reasoning("<think>shown</think>Hi.") == "Hi."
        assert reasoning_trace.drain() == ["shown"]
    finally:
        settings.show_thinking = original
        reasoning_trace.clear()


def test_buffer_is_bounded():
    # Serve, book and tarot modes generate without ever draining, so the
    # buffer must not grow for the life of the session.
    settings = get_settings().provider
    original = settings.show_thinking
    try:
        settings.show_thinking = True
        reasoning_trace.clear()
        for index in range(100):
            reasoning_trace.record(f"trace {index}")
        traces = reasoning_trace.drain()
        assert len(traces) <= 16
        assert traces[-1] == "trace 99"
    finally:
        settings.show_thinking = original
        reasoning_trace.clear()


# --- Classifying a stream of deltas -----------------------------------------

def test_stream_splits_tag_across_deltas():
    (answer, reasoning), emitted = _run_stream(
        ["<thi", "nk>weigh", "ing it</thi", "nk>Ans", "wer."]
    )
    assert (answer, reasoning) == ("Answer.", "weighing it")
    assert emitted == [
        ("reasoning", "weigh"),
        ("reasoning", "ing it"),
        ("answer", "Ans"),
        ("answer", "wer."),
    ]


def test_stream_starting_inside_reasoning():
    (answer, reasoning), emitted = _run_stream(
        ["weighing it", "</think>Answer."], starts_inside=True
    )
    assert (answer, reasoning) == ("Answer.", "weighing it")
    assert emitted[0] == ("reasoning", "weighing it")


def test_stream_passes_through_text_resembling_a_tag():
    (answer, reasoning), _ = _run_stream(["a < b and c ", "< d"])
    assert (answer, reasoning) == ("a < b and c < d", "")


# --- llama-server's separate reasoning field --------------------------------

class _FakeSSEResponse:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)


class _FakeServerProvider(LlamaServerProvider):
    """A provider with the HTTP plumbing stubbed, so no server is needed."""

    base_url = "http://127.0.0.1:0"
    _auth_headers: dict = {}
    model_name = "test-model.gguf"

    def __init__(self):
        self.temperature = 0.7
        self.max_tokens = 128


def test_streaming_reads_reasoning_content_field():
    settings = get_settings().provider
    original = settings.show_thinking
    frames = [
        {"choices": [{"delta": {"reasoning_content": "They said hello. "}}]},
        {"choices": [{"delta": {"reasoning_content": "Be brief."}}]},
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " there!"}}]},
    ]
    lines = [f"data: {json.dumps(f)}" for f in frames]
    # A blank line, a non-data line and a malformed payload must cost at most
    # the frame they appear in.
    lines += ["", "garbage", "data: {oops", "data: [DONE]"]

    try:
        settings.show_thinking = True
        reasoning_trace.clear()
        emitted = _capture()
        with patch(
            "athena.providers.llamaserver.requests.post",
            return_value=_FakeSSEResponse(lines),
        ):
            answer = _FakeServerProvider()._generate_streaming({"messages": []})

        assert answer == "Hello there!"
        assert emitted == [
            ("reasoning", "They said hello. "),
            ("reasoning", "Be brief."),
            ("answer", "Hello"),
            ("answer", " there!"),
        ]
        assert reasoning_trace.drain() == ["They said hello. Be brief."]
    finally:
        streaming.clear_sink()
        settings.show_thinking = original
        reasoning_trace.clear()


def test_non_streaming_reads_reasoning_content_field():
    settings = get_settings().provider
    original = settings.show_thinking
    payload = {
        "choices": [
            {
                "message": {
                    "content": "Hello there!",
                    "reasoning_content": "They said hello.",
                }
            }
        ]
    }

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    try:
        settings.show_thinking = True
        reasoning_trace.clear()
        with patch(
            "athena.providers.llamaserver.requests.post", return_value=_FakeResponse()
        ):
            answer = _FakeServerProvider().generate("hello")

        assert answer == "Hello there!"
        assert reasoning_trace.drain() == ["They said hello."]
    finally:
        settings.show_thinking = original
        reasoning_trace.clear()


# --- Only the answer call streams -------------------------------------------

def test_only_the_answer_call_streams():
    # The planner and learning phases call the same provider. If they streamed
    # too, their prompts would print over the user's answer.
    calls = []

    class _Provider:
        supports_streaming = True

        def generate(self, prompt, system=None, stream=False):
            calls.append(stream)
            return "answer"

    from athena.cognition.engine import CognitiveEngine
    from athena.thought.models import Thought

    provider = _Provider()
    thought = Thought(user_input="hello")
    CognitiveEngine(provider).process(thought)

    assert calls == [True]

    provider.generate("planner prompt")
    assert calls == [True, False]


def test_provider_without_streaming_is_called_the_old_way():
    class _LegacyProvider:
        def generate(self, prompt, system=None):
            return "answer"

    from athena.cognition.engine import CognitiveEngine
    from athena.thought.models import Thought

    thought = Thought(user_input="hello")
    CognitiveEngine(_LegacyProvider()).process(thought)
    assert thought.get_response() == "answer"
