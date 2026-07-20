"""Verification tests for the Thought Pipeline (originally Sprint 3).

Written as a standalone script, so every check ended in `return True` / `return
False` rather than an assertion. pytest treats a returned value as a mistake
(PytestReturnNotNoneWarning) and a `return False` failure passed silently, so
the checks are now plain asserts.

The two end-to-end tests run against a stub provider rather than a real model.
They were previously skipped when no .gguf was present and *failed* when one
was — asking ProviderFactory for a provider prompts for a model when several
are installed, and pytest captures stdin, so the prompt raised OSError. Either
way they never actually verified anything.

What they check is the wiring — that the pipeline stages hand off correctly and
a response comes back — and that needs a provider, not inference. Stubbing it
makes them run everywhere in milliseconds instead of loading gigabytes of
weights. Real inference is exercised by the diagnose_* scripts, which are run
by hand.
"""

import asyncio
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_STUB_ANSWER = "A stubbed response."


class _StubProvider:
    """Stands in for a loaded model, for tests about plumbing rather than output.

    Implements the surface the pipeline actually touches: the engine calls
    generate(), the knowledge pipeline calls generate()/call(), and the Context
    Budget Manager needs count_tokens() and get_context_window().
    """

    model_name = "stub-model.gguf"
    supports_streaming = False

    def generate(self, prompt, system=None, stream=False):
        # "NONE" is the extractor's contract for "no facts here", so the
        # learning phase runs its real code path and settles without inventing
        # candidates from stub text.
        if "knowledge extractor" in str(system or "").lower():
            return "NONE"
        return _STUB_ANSWER

    def call(self, prompt):
        return self.generate(prompt)

    def count_tokens(self, text):
        return max(1, len(str(text)) // 4)

    def get_context_window(self):
        return 4096


@pytest.fixture
def stub_providers(monkeypatch):
    """Make ProviderFactory hand out stubs instead of loading a model."""
    from athena.providers import ProviderFactory

    monkeypatch.setattr(ProviderFactory, "create", staticmethod(lambda: _StubProvider()))
    monkeypatch.setattr(
        ProviderFactory,
        "create_learning",
        staticmethod(lambda reasoning_provider=None: _StubProvider()),
    )
    return _StubProvider


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_imports():
    """All pipeline modules import cleanly."""
    from athena.brain.brain import AthenaBrain
    from athena.thought.models import Thought
    from athena.thought.pipeline import ThoughtPipeline

    assert Thought is not None
    assert ThoughtPipeline is not None
    assert AthenaBrain is not None


def test_thought_creation():
    """A Thought can be instantiated and carries an id and its input."""
    from athena.thought.models import Thought

    thought = Thought(user_input="test message")

    assert thought.id is not None, "Thought should have an ID"
    assert thought.user_input == "test message"
    assert thought.created_at is not None


def test_pipeline_stages():
    """The pipeline exposes its public methods and every internal stage."""
    from athena.thought.pipeline import ThoughtPipeline

    for method in ("create", "process"):
        assert hasattr(ThoughtPipeline, method), f"missing {method}()"

    stages = [
        "_initialize",
        "_load_knowledge",
        "_reason",
        "_plan",
        "_prepare_tools",
        "_build_response",
        "_reflect",
        "_finalize",
    ]

    for stage in stages:
        assert hasattr(ThoughtPipeline, stage), f"missing stage {stage}()"


def test_file_structure():
    """The files the pipeline is made of are all present."""
    required_files = [
        "athena/thought/__init__.py",
        "athena/thought/models.py",
        "athena/thought/pipeline.py",
        "docs/SPRINT3.md",
    ]

    missing = [f for f in required_files if not (PROJECT_ROOT / f).exists()]

    assert not missing, f"missing files: {missing}"


def test_documentation():
    """The pipeline's design document exists and has real content."""
    doc_path = PROJECT_ROOT / "docs" / "SPRINT3.md"

    assert doc_path.exists(), "docs/SPRINT3.md not found"
    assert len(doc_path.read_text(encoding="utf-8")) > 100, (
        "docs/SPRINT3.md should have substantial content"
    )


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def test_memory_manager_integration():
    """MemoryManager is instantiable and its working-memory interface works.

    Only working memory is exercised: episodic and semantic memory persist to
    disk, so writing to them here would pollute the developer's real memory
    files.
    """
    from athena.memory.manager import MemoryManager

    manager = MemoryManager()
    assert manager is not None

    # Two stores: the working window and durable semantic facts. There is no
    # remember()/recall() pair — the episodic store they belonged to held
    # verbatim transcripts of the conversation working memory already carries.
    for method in (
        "store_working",
        "get_working",
        "clear_working",
        "learn",
        "query_semantic",
    ):
        assert hasattr(manager, method), f"MemoryManager missing {method}()"

    assert not hasattr(manager, "remember"), "episodic store is back"

    manager.clear_working()
    manager.store_working("test memory entry", {"category": "test"})

    entries = manager.get_working()
    assert len(entries) == 1
    assert entries[0].content == "test memory entry"

    manager.clear_working()
    assert manager.get_working() == []


def test_pipeline_execution(stub_providers):
    """The pipeline runs end to end and returns a thought.

    ThoughtPipeline takes its collaborators by injection — a bare
    ThoughtPipeline() has provider=None and fails as soon as it reasons — so
    this wires it up the same way AthenaBrain does.
    """
    from athena.knowledge.manager import KnowledgeManager
    from athena.memory.manager import MemoryManager
    from athena.providers import ProviderFactory
    from athena.thought.models import Thought
    from athena.thought.pipeline import ThoughtPipeline

    provider = ProviderFactory.create()
    learning_provider = ProviderFactory.create_learning(provider)
    memory_manager = MemoryManager()
    knowledge_manager = KnowledgeManager(
        working_memory=memory_manager.working_memory,
        provider=learning_provider,
        memory_manager=memory_manager,
    )
    pipeline = ThoughtPipeline(
        memory_manager,
        knowledge_manager,
        provider,
        learning_provider=learning_provider,
    )

    # process() returns the final response string (the Thought it was given is
    # mutated in place), not the Thought itself.
    thought = Thought(user_input="test message")
    response = asyncio.run(pipeline.process(thought))

    assert isinstance(response, str) and response.strip(), (
        f"process() should return a non-empty response, got {response!r}"
    )
    assert thought.response == response


def test_athena_brain_integration(stub_providers):
    """AthenaBrain.process() drives the pipeline and returns a response."""
    from athena.brain.brain import AthenaBrain

    brain = AthenaBrain()

    response = asyncio.run(brain.process("test message via AthenaBrain"))

    assert isinstance(response, str) and response.strip()
    # The turn is recorded for the next one to see.
    assert any("test message via AthenaBrain" in entry for entry in brain.history)
