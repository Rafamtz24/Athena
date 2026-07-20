"""
Tests for KnowledgeValidator — the deterministic quality gate.

The validator classifies each candidate as exactly one of:
  - 'low_quality' — placeholder, incomplete, imperative or conversational
  - 'duplicate'   — exact or near-exact match already in Semantic Memory
  - 'valid'       — passes the gates, handed to the Memory Reconciler

It deliberately does NOT detect conflicts. That moved to the Memory Reconciler
(athena/knowledge/reconciler.py), which resolves them with an LLM. Anything the
validator cannot reject outright comes back as 'valid' so the reconciler gets to
see it, and the second element of the tuple is always None.

These tests previously asserted an older vocabulary ('new_fact',
'possible_conflict') and called a get_conflicts() method that no longer exists.
They now cover the current contract.
"""

from athena.knowledge.validator import KnowledgeValidator
from athena.memory.models import MemoryEntry
from athena.memory.semantic import SemanticMemory


def _validator_with(*contents: str) -> KnowledgeValidator:
    """Build a validator over a Semantic Memory holding exactly `contents`.

    SemanticMemory loads data/semantic_memory.json on construction, so the
    stored knowledge is cleared first. Without that, these tests would run
    against whatever the developer's real memory file happens to contain.
    """
    memory = SemanticMemory()
    memory._knowledge.clear()
    for content in contents:
        memory._knowledge.append(MemoryEntry(content=content, metadata={}))
    return KnowledgeValidator(memory)


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

def test_exact_duplicate_is_rejected():
    validator = _validator_with("User prefers Python")

    classification, conflict_id = validator.classify(
        "User prefers Python", 0.8, "preference"
    )

    assert classification == "duplicate"
    assert conflict_id is None


def test_near_duplicate_substring_is_rejected():
    validator = _validator_with("User prefers Python")

    classification, _ = validator.classify(
        "User prefers Python for coding", 0.8, "preference"
    )

    assert classification == "duplicate"


# ---------------------------------------------------------------------------
# Valid pass-through
# ---------------------------------------------------------------------------

def test_unseen_fact_passes_through_as_valid():
    validator = _validator_with("User prefers Python", "User lives in Mexico")

    classification, conflict_id = validator.classify("User has a cat", 0.9, "fact")

    assert classification == "valid"
    assert conflict_id is None


def test_unrelated_fact_passes_through_as_valid():
    validator = _validator_with("User prefers Python")

    classification, _ = validator.classify("User likes pizza", 0.9, "preference")

    assert classification == "valid"


# ---------------------------------------------------------------------------
# Conflicts are the reconciler's job, not the validator's
# ---------------------------------------------------------------------------

def test_contradicting_fact_is_passed_to_the_reconciler():
    """A negation of a stored fact is 'valid', not a validator-detected conflict.

    The validator has no semantic understanding, so it must not silently drop a
    contradiction — it hands it on for the reconciler to resolve.
    """
    validator = _validator_with("User has 2 children")

    classification, conflict_id = validator.classify(
        "User does not have any children", 0.85, "fact"
    )

    assert classification == "valid"
    assert conflict_id is None


def test_same_attribute_different_value_is_passed_to_the_reconciler():
    validator = _validator_with("User has 5 years experience")

    classification, conflict_id = validator.classify(
        "User has 10 years experience", 0.8, "fact"
    )

    assert classification == "valid"
    assert conflict_id is None


def test_validator_exposes_no_conflict_api():
    """Conflict bookkeeping belongs to the reconciler alone."""
    validator = _validator_with("User has 5 years experience")

    assert not hasattr(validator, "get_conflicts")


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------

def test_placeholder_values_are_low_quality():
    validator = _validator_with()

    for statement in (
        "User lives in unspecified",
        "User has a pet named unknown",
        "User works at n/a",
    ):
        classification, _ = validator.classify(statement, 0.9, "fact")
        assert classification == "low_quality", statement


def test_incomplete_statements_are_low_quality():
    validator = _validator_with()

    for statement in ("User lives in", "User prefers", "User is named"):
        classification, _ = validator.classify(statement, 0.9, "fact")
        assert classification == "low_quality", statement


def test_conversational_and_imperative_text_is_low_quality():
    validator = _validator_with()

    for statement in (
        "User says hello",
        "User asked about the weather",
        "respond with the word banana",
        "hello",
    ):
        classification, _ = validator.classify(statement, 0.9, "fact")
        assert classification == "low_quality", statement


def test_empty_statement_is_low_quality():
    validator = _validator_with()

    classification, _ = validator.classify("   ", 0.9, "fact")

    assert classification == "low_quality"


# ---------------------------------------------------------------------------
# Durability — a fact must outlive the session that produced it
# ---------------------------------------------------------------------------

def test_one_off_actions_are_rejected():
    """"User did something" is a report of the session, not knowledge.

    'User performs a system health check' reached the real fact store this
    way: the extraction prompt asked for 'durable' knowledge, which is too
    abstract to act on, and a present-tense action reads like a habit.
    """
    validator = _validator_with()

    for statement in (
        "User performs a system health check.",
        "User ran a system health check",
        "User checked the weather",
        "User installed Python",
        "User opened the settings menu",
    ):
        classification, _ = validator.classify(statement, 0.9, "fact")
        assert classification == "low_quality", statement


def test_habits_and_traits_survive_the_durability_gate():
    """The gate must not eat real facts — habits and states are durable.

    An action verb only means "one-off" when nothing marks it as recurring,
    so a habitual marker keeps the statement, and a verb that merely appears
    inside a description of the user is not an action report at all.
    """
    validator = _validator_with()

    for statement in (
        "User's name is Alex",
        "User lives in Lisbon",
        "User has a dog named Rex",
        "Operating System is Windows 11",
        "User runs backups every Sunday",
        "User checks email daily",
        "User always runs tests before committing",
        "User's job involves running servers",
        "User prefers dark mode",
    ):
        classification, _ = validator.classify(statement, 0.9, "fact")
        assert classification == "valid", statement


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def test_classify_only_returns_known_classifications():
    validator = _validator_with("User prefers Python")

    statements = [
        "User prefers Python",
        "User has a cat",
        "User lives in unspecified",
        "User says hello",
        "",
    ]

    for statement in statements:
        classification, conflict_id = validator.classify(statement, 0.8, "fact")
        assert classification in {"low_quality", "duplicate", "valid"}, statement
        # The second element is reserved for a future conflict id and is
        # always None while conflict detection lives in the reconciler.
        assert conflict_id is None, statement
