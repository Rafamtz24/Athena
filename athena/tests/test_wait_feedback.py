"""
Tests for what the spinner says while the user waits.

A single label for the whole turn made a 45-second answer indistinguishable
from a hang. The wait has distinct phases and they are reported differently:

  Tool execution — named from the plan ("Searching the web"), driven by the
  pipeline's existing stage events rather than new reporting calls.

  Prompt evaluation — the model reading the prompt before it can emit a single
  token, which on a large model split across GPU and CPU is most of the wait.
  llama-server reports it via `prompt_progress` when asked with
  `return_progress`.

The planner is deliberately absent: it is rule-based and returns in
microseconds, so there is no planning phase long enough to show.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from athena.config.settings import get_settings
from athena.events.bus import get_event_bus
from athena.events.models import Event
from athena.terminal_chat import _ActivityIndicator, _LiveOutput, _PhaseLabels


class _FakeSpinner:
    """Records label changes instead of drawing them."""

    def __init__(self):
        self.labels = []
        self.stopped = False

    def set_label(self, label):
        self.labels.append(label)

    def stop(self):
        self.stopped = True


def _publish(event_type, payload):
    get_event_bus().publish(
        Event(
            type=event_type,
            source="thought_pipeline",
            payload=payload,
            metadata={},
        )
    )


# --- Stage labels ------------------------------------------------------------

def test_tool_plan_names_the_tool():
    spinner = _FakeSpinner()
    phases = _PhaseLabels(spinner)
    phases.subscribe()
    try:
        _publish("ToolPlanned", {"decision_tools": ["web"]})
        _publish("ToolExecuted", {"tool": "web", "executed": True})
    finally:
        phases.unsubscribe()

    assert spinner.labels == ["Searching the web", "Thinking"]


def test_plan_with_no_tool_leaves_the_label_alone():
    spinner = _FakeSpinner()
    phases = _PhaseLabels(spinner)
    phases.subscribe()
    try:
        _publish("ToolPlanned", {"decision_tools": ["none"]})
    finally:
        phases.unsubscribe()

    assert spinner.labels == []


def test_unsubscribed_labeller_ignores_later_turns():
    # Book, tarot and serve modes drive the same pipeline without this
    # spinner; a stale subscriber would relabel an indicator that is not
    # running.
    spinner = _FakeSpinner()
    phases = _PhaseLabels(spinner)
    phases.subscribe()
    phases.unsubscribe()

    _publish("ToolPlanned", {"decision_tools": ["system"]})
    assert spinner.labels == []


# --- Prompt-evaluation progress ---------------------------------------------

def test_progress_updates_the_label_without_printing(capsys):
    spinner = _FakeSpinner()
    live = _LiveOutput(spinner)

    live("progress", "0")
    live("progress", "63")

    assert spinner.labels == ["Reading the prompt 0%", "Reading the prompt 63%"]
    assert not live.printed
    assert not spinner.stopped
    assert capsys.readouterr().out == ""


def test_first_token_ends_the_progress_label():
    settings = get_settings().provider
    original = settings.show_thinking
    try:
        # With reasoning hidden the first token prints nothing, so the label
        # has to be reset explicitly or it would read "Reading the prompt
        # 100%" for the whole generation.
        settings.show_thinking = False
        spinner = _FakeSpinner()
        live = _LiveOutput(spinner)

        live("progress", "100")
        live("reasoning", "a hidden thought")

        assert spinner.labels == ["Reading the prompt 100%", "Thinking"]
        assert not live.printed
    finally:
        settings.show_thinking = original


# --- The indicator itself ----------------------------------------------------

def test_relabelling_pads_to_the_widest_label():
    # Switching to a shorter label must not leave the tail of a longer one on
    # screen, and stop() has to erase the full width it ever drew.
    indicator = _ActivityIndicator("Thinking")
    indicator.set_label("Searching the web")
    indicator.set_label("Done")

    assert indicator.label == "Done"
    assert indicator._width == len("Searching the web")


def test_stop_is_safe_when_never_started():
    _ActivityIndicator("Thinking").stop()


# --- Reasoning budget --------------------------------------------------------

def test_reasoning_budget_is_passed_to_the_server():
    """Reasoning and answer share one token budget, so an unbounded thinking
    model can starve its own reply — one measured turn spent 1830 of 2048
    tokens re-deriving the same sentence. The cap is enforced by the server,
    which closes the thought with a wrap-up message rather than cutting mid
    sentence, so an answer still follows.
    """
    from athena.config.settings import get_settings
    from athena.providers.llamaserver import build_server_command

    settings = get_settings().provider
    original = settings.reasoning_budget

    class _Config:
        n_ctx, n_batch, n_threads = 4096, 512, 6
        flash_attn, gpu_layers = False, 0
        backend, vram_bytes = "CPU", 0

    try:
        settings.reasoning_budget = 1024
        command = build_server_command("srv", "m.gguf", 8080, "k", _Config())

        index = command.index("--reasoning-budget")
        assert command[index + 1] == "1024"
        # A bare cap would truncate mid-thought; the message is what makes the
        # model wrap up and answer.
        assert command[index + 2] == "--reasoning-budget-message"
        assert command[index + 3].strip()

        # The budget must leave room for the answer within max_tokens.
        assert settings.reasoning_budget < settings.max_tokens

        settings.reasoning_budget = -1
        unlimited = build_server_command("srv", "m.gguf", 8080, "k", _Config())
        assert "--reasoning-budget" not in unlimited
    finally:
        settings.reasoning_budget = original
