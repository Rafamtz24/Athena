"""
Tests for batched reconciliation — one provider call per turn, not per fact.

Reconciliation used to make one LLM call per candidate, each re-sending the
whole of Semantic Memory. A turn yielding three facts cost three calls on the
local learning model, and every call judged its fact in isolation.

The risk batching introduces is that one malformed response loses every
candidate instead of one, so the tests that matter most here are the failure
paths: a dropped block, a garbled block, and a dead provider.
"""
from athena.knowledge.models import KnowledgeCandidate
from athena.knowledge.reconciler import (
    MemoryReconciler,
    parse_batch_reconciliation_response,
)
from athena.memory.models import MemoryEntry
from athena.memory.semantic import SemanticMemory


class _Provider:
    """Returns queued responses in order, counting calls."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, prompt, **kwargs):
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return "ACTION: DIFFERENT"


def _memory(*contents) -> SemanticMemory:
    memory = SemanticMemory()
    memory._knowledge.clear()
    memory._save = lambda: None
    for content in contents:
        memory._knowledge.append(MemoryEntry(content=content, metadata={}))
    return memory


def _candidates(*statements):
    return [KnowledgeCandidate(s, 0.9, "fact") for s in statements]


def _contents(memory):
    return [str(e.content) for e in memory.query()]


# ---------------------------------------------------------------------------
# Call count
# ---------------------------------------------------------------------------

def test_many_candidates_cost_one_call():
    memory = _memory("User has a dog named Rex")
    provider = _Provider(
        "FACT: 1\nACTION: DIFFERENT\n\n"
        "FACT: 2\nACTION: DIFFERENT\n\n"
        "FACT: 3\nACTION: DIFFERENT\n"
    )

    results = MemoryReconciler(provider).reconcile(
        _candidates(
            "User has a cat named Cleo",
            "User prefers dark mode",
            "User is learning Rust",
        ),
        memory,
    )

    assert provider.calls == 1
    assert results['new_facts'] == 3
    assert results['processed'] == 3


def test_deterministic_candidates_never_reach_the_model():
    """Single-valued attributes are resolved without a model at all."""
    memory = _memory("User has a dog named Rex")
    provider = _Provider()

    results = MemoryReconciler(provider).reconcile(
        _candidates("User's name is Alex", "User lives in Lisbon"),
        memory,
    )

    assert provider.calls == 0
    assert results['new_facts'] == 2


def test_single_candidate_uses_the_simpler_prompt():
    """One candidate has nothing to batch, and the pair format is easier for
    a small learning model to produce correctly."""
    memory = _memory("User has a dog named Rex")
    provider = _Provider("ACTION: DIFFERENT")

    results = MemoryReconciler(provider).reconcile(
        _candidates("User prefers dark mode"), memory
    )

    assert provider.calls == 1
    assert results['new_facts'] == 1


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------

def test_a_dropped_block_costs_one_retry_not_the_batch():
    memory = _memory("User has a dog named Rex")
    provider = _Provider(
        # Verdict for candidate 2 is missing entirely.
        "FACT: 1\nACTION: DIFFERENT\n\nFACT: 3\nACTION: DIFFERENT\n",
        "ACTION: DIFFERENT",
    )

    results = MemoryReconciler(provider).reconcile(
        _candidates(
            "User has a cat named Cleo",
            "User prefers dark mode",
            "User is learning Rust",
        ),
        memory,
    )

    assert provider.calls == 2  # one batch, one retry
    assert results['new_facts'] == 3
    assert "User prefers dark mode" in _contents(memory)


def test_a_block_without_a_verdict_is_retried_not_assumed():
    """A block with no ACTION line must not be read as DIFFERENT — that would
    store a fact on no evidence."""
    verdicts = parse_batch_reconciliation_response(
        "FACT: 1\nI think this one is probably new?\n\nFACT: 2\nACTION: DIFFERENT\n",
        2,
    )

    assert 0 not in verdicts
    assert verdicts[1] == ('DIFFERENT', [])


def test_provider_failure_leaves_memory_untouched():
    class _Dead:
        def generate(self, prompt, **kwargs):
            raise RuntimeError("server down")

    memory = _memory("User has a dog named Rex")

    results = MemoryReconciler(_Dead()).reconcile(
        _candidates("User has a cat named Cleo", "User prefers dark mode"),
        memory,
    )

    assert results['errors'] == 2
    assert _contents(memory) == ["User has a dog named Rex"]


def test_unparseable_response_does_not_corrupt_memory():
    memory = _memory("User has a dog named Rex")
    # Batch is garbage, and so is every per-candidate retry.
    provider = _Provider("sure thing boss", "also nonsense", "still nonsense")

    MemoryReconciler(provider).reconcile(
        _candidates("User has a cat named Cleo", "User prefers dark mode"), memory
    )

    # Retries fall back to the single-pair parser, which defaults to
    # DIFFERENT; what must not happen is a removal.
    assert "User has a dog named Rex" in _contents(memory)


# ---------------------------------------------------------------------------
# Batch-specific semantics
# ---------------------------------------------------------------------------

def test_candidates_that_duplicate_each_other_are_stored_once():
    """Both are judged against memory as it stood before either was stored,
    so nothing in the model's answer can catch the collision."""
    memory = _memory("User has a dog named Rex")
    provider = _Provider("FACT: 1\nACTION: DIFFERENT\n\nFACT: 2\nACTION: DIFFERENT\n")

    results = MemoryReconciler(provider).reconcile(
        _candidates("User prefers dark mode", "User prefers dark mode"), memory
    )

    assert results['new_facts'] == 1
    assert results['duplicates'] == 1
    assert _contents(memory).count("User prefers dark mode") == 1


def test_conflicts_remove_the_named_entry():
    memory = _memory("User prefers light mode", "User has a dog named Rex")
    provider = _Provider(
        'FACT: 1\nACTION: CONFLICT\nCONFLICTS:\n- "User prefers light mode"\n\n'
        'FACT: 2\nACTION: DIFFERENT\n'
    )

    results = MemoryReconciler(provider).reconcile(
        _candidates("User prefers dark mode", "User is learning Rust"), memory
    )

    assert results['conflicts'] == 1
    assert "User prefers light mode" not in _contents(memory)
    assert "User prefers dark mode" in _contents(memory)


def test_out_of_range_fact_numbers_are_ignored():
    verdicts = parse_batch_reconciliation_response(
        "FACT: 9\nACTION: DIFFERENT\n", 2
    )

    assert verdicts == {}
