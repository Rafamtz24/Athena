"""
Tests for the memory consolidator — the only pass that acts on stored facts.

Every other quality rule runs at write time and therefore binds only the next
fact. When a rule is added, whatever it would now reject is already in memory
and stays there: "User performs a system health check" survived in the real
store because the durability gate that rejects it did not exist yet.

Consolidation is deterministic on purpose. Each rule here is one Athena can
apply without a model, so it is reliable whichever model is loaded, and it
cannot silently delete good facts through a bad generation.

The load-bearing tests are the ones asserting what is KEPT.
"""
from datetime import datetime, timedelta, timezone

from athena.knowledge.consolidator import consolidate, describe
from athena.memory.models import MemoryEntry
from athena.memory.semantic import SemanticMemory

_NOW = datetime.now(timezone.utc)


def _memory(*entries) -> SemanticMemory:
    """Build a SemanticMemory from (content, days_old) pairs, or bare strings.

    SemanticMemory loads the real store on construction and saves on every
    change, so both are neutralised: these tests must not read or write the
    developer's actual memory file.
    """
    memory = SemanticMemory()
    memory._knowledge.clear()
    memory._save = lambda: None
    for item in entries:
        content, age = item if isinstance(item, tuple) else (item, 0)
        entry = MemoryEntry(content=content, metadata={})
        entry.timestamp = _NOW - timedelta(days=age)
        memory._knowledge.append(entry)
    return memory


def _contents(memory) -> list:
    return [str(e.content) for e in memory.query()]


# ---------------------------------------------------------------------------
# What consolidation must KEEP
# ---------------------------------------------------------------------------

def test_a_clean_store_is_left_alone():
    memory = _memory(
        "User's name is Alex",
        "User lives in Lisbon",
        "User has a dog named Rex",
        "Operating System is Windows 11",
        "User's favorite color is blue",
    )

    results = consolidate(memory)

    assert results['removed'] == 0
    assert len(_contents(memory)) == 5
    # Nothing removed means nothing to tell the user about.
    assert describe(results) == ''


def test_multi_valued_facts_are_not_treated_as_conflicts():
    """A user may have two pets. Only single-valued attributes conflict.

    This is the failure that would matter most: a consolidator that "resolves"
    every repeated subject deletes real knowledge on every startup.
    """
    memory = _memory(
        "User has a dog named Rex",
        "User has a cat named Cleo",
        "User is learning Rust",
        "User is learning Spanish",
    )

    consolidate(memory)

    assert len(_contents(memory)) == 4


def test_consolidation_is_idempotent():
    memory = _memory(
        ("User lives in Porto", 9),
        ("User lives in Lisbon", 1),
    )

    first = consolidate(memory)
    second = consolidate(memory)

    assert first['removed'] == 1
    assert second['removed'] == 0
    assert _contents(memory) == ["User lives in Lisbon"]


def test_empty_store_is_handled():
    assert consolidate(_memory())['removed'] == 0


# ---------------------------------------------------------------------------
# What consolidation must REMOVE
# ---------------------------------------------------------------------------

def test_entries_the_current_gates_reject_are_dropped():
    memory = _memory(
        "User performs a system health check.",
        "User's name is Alex",
    )

    results = consolidate(memory)

    assert results['stale'] == 1
    assert _contents(memory) == ["User's name is Alex"]


def test_exact_duplicates_collapse_to_the_newest():
    memory = _memory(
        ("User's name is Alex", 3),
        ("user's name is alex.", 1),  # same fact, different case/punctuation
    )

    results = consolidate(memory)

    assert results['duplicates'] == 1
    assert _contents(memory) == ["user's name is alex."]


def test_outdated_single_valued_facts_are_dropped():
    memory = _memory(
        ("User lives in Porto", 9),
        ("User lives in Lisbon", 1),
    )

    results = consolidate(memory)

    assert results['conflicts'] == 1
    assert _contents(memory) == ["User lives in Lisbon"]


def test_each_entry_is_counted_once():
    """An entry can match several rules; it must not be counted twice."""
    memory = _memory(
        ("User performs a system health check.", 5),
        ("User performs a system health check.", 2),
    )

    results = consolidate(memory)

    assert results['removed'] == 2
    assert results['removed'] == (
        results['stale'] + results['duplicates'] + results['conflicts']
    )
    assert _contents(memory) == []


def test_describe_summarises_what_was_removed():
    memory = _memory(
        ("User performs a system health check.", 5),
        ("User lives in Porto", 9),
        ("User lives in Lisbon", 1),
    )

    summary = describe(consolidate(memory))

    assert summary.startswith("Tidied memory: removed 2 entries")
    assert "1 stale" in summary
    assert "1 outdated" in summary
