"""Verification tests for the Thought Pipeline (originally Sprint 3).

Written as a standalone script, so every check ended in `return True` / `return
False` rather than an assertion. pytest treats a returned value as a mistake
(PytestReturnNotNoneWarning) and a `return False` failure passed silently, so
the checks are now plain asserts.

Tests that need a loaded language model are skipped when no .gguf is present,
which keeps the suite green on a fresh clone.
"""

import asyncio
import os
from pathlib import Path

import pytest

from athena.config.settings import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _model_available() -> bool:
    """True when a reasoning model is installed."""
    directory = Path(get_settings().provider.reason_model_directory)
    return directory.is_dir() and any(directory.rglob("*.gguf"))


requires_model = pytest.mark.skipif(
    not _model_available(),
    reason="needs a .gguf model in models/reason",
)


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
        "_load_memory",
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

    for method in (
        "store_working",
        "get_working",
        "clear_working",
        "remember",
        "recall",
        "learn",
        "query_semantic",
    ):
        assert hasattr(manager, method), f"MemoryManager missing {method}()"

    manager.clear_working()
    manager.store_working("test memory entry", {"category": "test"})

    entries = manager.get_working()
    assert len(entries) == 1
    assert entries[0].content == "test memory entry"

    manager.clear_working()
    assert manager.get_working() == []


@requires_model
def test_pipeline_execution():
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


@requires_model
def test_athena_brain_integration():
    """AthenaBrain.process() drives the pipeline and returns a response."""
    from athena.brain.brain import AthenaBrain

    brain = AthenaBrain()

    response = asyncio.run(brain.process("test message via AthenaBrain"))

    assert response is not None
