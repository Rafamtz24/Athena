"""
Memory Reconciliation Diagnostic

Checks that the MemoryReconciler correctly handles:
1. Conflicting facts: newer replaces older (name change)
2. Conflicting facts: newer replaces older (color change)
3. Conflicting facts: newer replaces older (city change)

Run manually:  python athena/tests/diagnose_memory_reconciliation.py

DESTRUCTIVE — this is not part of the test suite, and pytest collects nothing
from it. It loads a real model and CLEARS data/semantic_memory.json to set up
each scenario, so running it discards whatever Athena has learned. Named
diagnose_* rather than test_* for that reason: nothing here should run
unattended alongside the suite.
"""

import json
import sys
import os
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from athena.memory.semantic import SemanticMemory
from athena.knowledge.models import KnowledgeCandidate
from athena.knowledge.reconciler import MemoryReconciler
from athena.config.settings import get_settings

SEMANTIC_MEMORY_PATH = Path(get_settings().storage.semantic_memory_path)


def read_sm_from_disk() -> list:
    """Read semantic memory entries directly from disk."""
    if not SEMANTIC_MEMORY_PATH.exists():
        return []
    with open(SEMANTIC_MEMORY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("entries", [])


def write_sm_to_disk(entries: list) -> None:
    """Write semantic memory entries directly to disk (for test setup)."""
    SEMANTIC_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SEMANTIC_MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump({"entries": entries}, f, indent=2)


def clear_semantic_memory() -> None:
    """Clear semantic memory on disk."""
    write_sm_to_disk([])


def assert_fact_exists(entries: list, fact_text: str, test_name: str) -> None:
    """Assert that a fact exists in the entries list."""
    for entry in entries:
        if entry.get("content") == fact_text:
            return
    print(f"  FAIL [{test_name}] Expected fact not found: '{fact_text}'")
    print(f"  Actual entries: {[e.get('content') for e in entries]}")
    sys.exit(1)


def assert_fact_missing(entries: list, fact_text: str, test_name: str) -> None:
    """Assert that a fact does NOT exist in the entries list."""
    for entry in entries:
        if entry.get("content") == fact_text:
            print(f"  FAIL [{test_name}] Fact should NOT exist: '{fact_text}'")
            print(f"  Actual entries: {[e.get('content') for e in entries]}")
            sys.exit(1)


def print_sm_state(label: str) -> None:
    """Print current semantic memory state for debugging."""
    entries = read_sm_from_disk()
    contents = [e.get("content", "") for e in entries]
    print(f"  [{label}] SM contents: {contents}")


def create_provider():
    """Create an LLM provider for testing."""
    from athena.providers.llamacpp import LlamaCppProvider
    return LlamaCppProvider()


def run_test_scenario(
    test_name: str,
    initial_facts: list,
    candidate_statement: str,
    expected_conflict_count: int,
) -> None:
    """
    Run a single reconciliation test scenario.
    
    Args:
        test_name: Human-readable test name
        initial_facts: List of fact strings to seed SM with
        candidate_statement: The new candidate fact
        expected_conflict_count: Expected number of conflicts resolved
    """
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")

    # Setup: clear SM and seed with initial facts
    clear_semantic_memory()
    sm = SemanticMemory()  # Loads from disk (which is now empty)
    for fact in initial_facts:
        sm.learn(fact, {"type": "knowledge", "confidence": 0.8, "category": "test"})

    # Verify initial state on disk
    disk_entries = read_sm_from_disk()
    for fact in initial_facts:
        assert_fact_exists(disk_entries, fact, f"{test_name} [setup]")
    print(f"  [setup] Initial facts verified on disk [OK]")

    # Create provider and reconciler
    provider = create_provider()
    reconciler = MemoryReconciler(provider)

    # Create candidate
    candidate = KnowledgeCandidate(
        statement=candidate_statement,
        confidence=0.8,
        category="extracted",
    )

    # Reconcile
    print(f"  [reconcile] Sending candidate: '{candidate_statement}'")
    result = reconciler.reconcile([candidate], sm)

    # Read results from disk
    disk_entries = read_sm_from_disk()
    print(f"  [result] Reconciliation counts: {result}")
    print_sm_state("after")

    # Verify: old conflicting facts are gone
    for fact in initial_facts:
        assert_fact_missing(disk_entries, fact, f"{test_name} [old fact removed]")
    print(f"  [verify] Old conflicting facts removed [OK]")

    # Verify: new fact is present
    assert_fact_exists(disk_entries, candidate_statement, f"{test_name} [new fact stored]")
    print(f"  [verify] New fact stored [OK]")

    # Verify: only the expected number of entries (1 new fact)
    assert len(disk_entries) == 1, (
        f"Expected 1 entry, but found {len(disk_entries)}: "
        f"{[e.get('content') for e in disk_entries]}"
    )
    print(f"  [verify] Exactly 1 entry in SM [OK]")
    print(f"  PASS [OK]")


def main():
    print("=" * 60)
    print("MEMORY RECONCILIATION VERIFICATION")
    print("=" * 60)

    # ── Test 1: Name change ──
    run_test_scenario(
        test_name="Name change: Alex -> Pedro",
        initial_facts=["User's name is Alex"],
        candidate_statement="User's name is Pedro",
        expected_conflict_count=1,
    )

    # ── Test 2: Favorite color change ──
    run_test_scenario(
        test_name="Color change: blue -> green",
        initial_facts=["User's favorite color is blue"],
        candidate_statement="User's favorite color is green",
        expected_conflict_count=1,
    )

    # ── Test 3: City change ──
    run_test_scenario(
        test_name="City change: Lisbon -> Porto",
        initial_facts=["User lives in Lisbon"],
        candidate_statement="User lives in Porto",
        expected_conflict_count=1,
    )

    print(f"\n{'='*60}")
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()