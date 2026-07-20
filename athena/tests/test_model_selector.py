"""
Tests for GGUF discovery and selection.

The case worth guarding is the one with no stdin. Selection prompts with
input() when several models are installed, and input() raises when there is
nothing to read — piped input that ran out, a launcher started without a
console, or a test runner that captures stdin. Retrying inside the prompt loop
would spin forever on the same EOF, and letting it propagate takes down a
session that only needed a model chosen.

That is not hypothetical: it broke two tests in this suite, which asked
ProviderFactory for a provider and hit the prompt.
"""
import builtins

import pytest

from athena.providers.model_selector import (
    discover_models,
    resolve_model_path,
    resolve_model_path_optional,
)


@pytest.fixture
def model_dir(tmp_path):
    """A directory of fake .gguf files — discovery only looks at names."""

    def _make(*names):
        for name in names:
            (tmp_path / name).write_bytes(b"")
        return tmp_path

    return _make


def _stdin_raising(exception):
    def _raise(*args, **kwargs):
        raise exception

    return _raise


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_models_are_found_and_sorted(model_dir):
    directory = model_dir("beta.gguf", "alpha.gguf")

    assert [p.name for p in discover_models(directory)] == ["alpha.gguf", "beta.gguf"]


def test_search_is_recursive(tmp_path):
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "model.gguf").write_bytes(b"")

    assert [p.name for p in discover_models(tmp_path)] == ["model.gguf"]


def test_missing_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_models(tmp_path / "nope")


def test_empty_directory_is_an_error(model_dir):
    with pytest.raises(FileNotFoundError):
        resolve_model_path(model_dir())


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def test_a_single_model_needs_no_prompt(model_dir, monkeypatch):
    directory = model_dir("only.gguf")
    monkeypatch.setattr(builtins, "input", _stdin_raising(AssertionError("prompted")))

    assert resolve_model_path(directory).endswith("only.gguf")


def test_the_chosen_number_selects_that_model(model_dir, monkeypatch):
    directory = model_dir("alpha.gguf", "beta.gguf")
    monkeypatch.setattr(builtins, "input", lambda *a, **k: "2")

    assert resolve_model_path(directory).endswith("beta.gguf")


def test_an_invalid_answer_asks_again(model_dir, monkeypatch):
    directory = model_dir("alpha.gguf", "beta.gguf")
    answers = iter(["0", "nine", "1"])
    monkeypatch.setattr(builtins, "input", lambda *a, **k: next(answers))

    assert resolve_model_path(directory).endswith("alpha.gguf")


# ---------------------------------------------------------------------------
# No stdin to ask on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("error", [EOFError(), OSError("stdin captured")])
def test_unusable_stdin_falls_back_instead_of_raising(model_dir, monkeypatch, error):
    directory = model_dir("beta.gguf", "alpha.gguf")
    monkeypatch.setattr(builtins, "input", _stdin_raising(error))

    # First alphabetically, so a non-interactive run is at least reproducible.
    assert resolve_model_path(directory).endswith("alpha.gguf")


def test_the_fallback_says_what_it_did(model_dir, monkeypatch, capsys):
    """Silently loading an unpicked model shows up later as an out-of-memory
    abort or an unexplained slow session — the models can differ by tens of
    gigabytes."""
    directory = model_dir("alpha.gguf", "beta.gguf")
    monkeypatch.setattr(builtins, "input", _stdin_raising(EOFError()))

    resolve_model_path(directory)

    assert "alpha.gguf" in capsys.readouterr().out


def test_the_optional_resolver_falls_back_too(model_dir, monkeypatch):
    """The learning-model folder goes through its own resolver, which has the
    same prompt and so needs the same protection."""
    directory = model_dir("beta.gguf", "alpha.gguf")
    monkeypatch.setattr(builtins, "input", _stdin_raising(EOFError()))

    assert resolve_model_path_optional(directory).endswith("alpha.gguf")


def test_the_optional_resolver_returns_none_when_empty(model_dir):
    assert resolve_model_path_optional(model_dir()) is None
