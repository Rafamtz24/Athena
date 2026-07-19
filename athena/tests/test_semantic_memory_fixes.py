"""
Regression tests for two Semantic Memory fixes:

  Problem 1 — Unrelated memories injected into conversation.
      KnowledgeManager.retrieve() used substring matching with no relevance
      threshold, so "name" matched "named" (dog fact) and any weak overlap
      was injected. Fixed via stemmed whole-word matching + a threshold.

  Problem 2 — Contradicting single-valued facts were not reconciled because
      conflict detection was fully delegated to the LLM. Added a deterministic
      (subject, attribute, value) layer so name/location/etc. conflicts resolve
      without a model call.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from athena.knowledge.manager import KnowledgeManager
from athena.knowledge.attributes import parse_fact
from athena.knowledge.models import KnowledgeCandidate
from athena.knowledge.reconciler import MemoryReconciler


# ──────────────────────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────────────────────

class _Entry:
    """Minimal stand-in for a MemoryEntry."""
    def __init__(self, entry_id, content):
        self.id = entry_id
        self.content = content


class FakeSemanticMemory:
    """In-memory SemanticMemory that never touches disk."""
    def __init__(self, contents):
        self._entries = [_Entry(f"id-{i}", c) for i, c in enumerate(contents)]
        self._counter = len(self._entries)

    def query(self):
        return list(self._entries)

    def learn(self, content, metadata=None):
        entry = _Entry(f"id-{self._counter}", str(content).strip())
        self._counter += 1
        self._entries.append(entry)
        return entry.id

    def remove(self, entry_id):
        for i, e in enumerate(self._entries):
            if e.id == entry_id:
                self._entries.pop(i)
                return True
        return False

    def contents(self):
        return [e.content for e in self._entries]


class ExplodingProvider:
    """Provider that fails the test if the LLM is called."""
    def generate(self, prompt):
        raise AssertionError("LLM must NOT be called for single-valued attributes")


class _MemMgr:
    def __init__(self, contents):
        self._contents = contents
    def query_semantic(self):
        return [_Entry(f"id-{i}", c) for i, c in enumerate(self._contents)]


# ──────────────────────────────────────────────────────────────
# Problem 1 — retrieval
# ──────────────────────────────────────────────────────────────

def _make_manager(contents):
    km = KnowledgeManager(working_memory=None, provider=None,
                          memory_manager=_MemMgr(contents))
    return km


def test_stem_keeps_name_and_named_distinct():
    assert KnowledgeManager._stem("name") != KnowledgeManager._stem("named")
    # But inflections of the same root collapse:
    assert KnowledgeManager._stem("lives") == KnowledgeManager._stem("live")
    assert KnowledgeManager._stem("dogs") == KnowledgeManager._stem("dog")
    print("  [OK] Stemmer keeps 'name'/'named' apart, collapses inflections")


def test_name_query_does_not_inject_dog():
    km = _make_manager([
        "User has a dog named Rex",
        "User's name is Alex",
        "User lives in Lisbon",
    ])
    result = km.retrieve("my name is actually alex")
    assert result is not None
    assert "Rex" not in result, "Dog fact must not be injected on a name query"
    assert "Lisbon" not in result, "Location fact must not be injected on a name query"
    assert "Alex" in result
    print("  [OK] Name query no longer injects the unrelated dog/location facts")


def test_unrelated_query_returns_nothing():
    km = _make_manager([
        "User has a dog named Rex",
        "User's name is Alex",
    ])
    # A query about the weather shares no discriminative words with any fact.
    assert km.retrieve("what is the weather like today") is None
    print("  [OK] Unrelated query retrieves nothing")


def test_relevant_query_still_retrieves():
    km = _make_manager([
        "User has a dog named Rex",
        "User's name is Alex",
    ])
    result = km.retrieve("what is my dog called")
    assert result is not None and "Rex" in result
    print("  [OK] Directly relevant query still retrieves the fact")


def test_capitalized_query_words_match():
    # Regression: tokenizing a mixed-case query dropped leading capitals
    # ("User" -> "ser"), which only worked before due to lenient substring
    # matching. Capitalized query words must match whole entry words.
    km = _make_manager([
        "User has a dog named Rex",
        "User lives in Lisbon",
    ])
    result = km.retrieve("Tell me about the Dog")
    assert result is not None and "Rex" in result
    print("  [OK] Capitalized query words are matched correctly")


# ──────────────────────────────────────────────────────────────
# Problem 2 — attribute parsing
# ──────────────────────────────────────────────────────────────

def test_parse_single_valued_attributes():
    cases = {
        "User's name is Alex": ("user", "name", "Alex"),
        "User name is Alex": ("user", "name", "Alex"),
        "My name is Alex": ("user", "name", "Alex"),
        "User lives in Lisbon": ("user", "location", "Lisbon"),
        "Rex's color is black": ("rex", "color", "black"),
        "Operating System is Windows 11": ("system", "operating_system", "Windows 11"),
    }
    for statement, (subj, attr, val) in cases.items():
        fact = parse_fact(statement)
        assert fact is not None, f"Expected {statement!r} to parse"
        assert fact.subject == subj
        assert fact.attribute == attr
        assert fact.value == val
    print("  [OK] Single-valued attributes parse to (subject, attribute, value)")


def test_parse_defers_multivalued_and_unknown():
    # A user may have many pets — not a single-valued attribute.
    assert parse_fact("User has a dog named Rex") is None
    # Ambiguous phrasing is deferred rather than guessed.
    assert parse_fact("User uses Windows 11") is None
    assert parse_fact("User enjoys hiking on weekends") is None
    print("  [OK] Multi-valued / ambiguous facts defer to the LLM (parse -> None)")


def test_name_variants_share_a_key():
    a = parse_fact("User's name is Alex")
    b = parse_fact("User name is Alex")
    c = parse_fact("My name is Alex")
    assert a.key == b.key == c.key == ("user", "name")
    assert a.value_norm == b.value_norm == c.value_norm == "alex"
    print("  [OK] Name phrasing variants normalize to one key/value")


# ──────────────────────────────────────────────────────────────
# Problem 2 — deterministic reconciliation (no LLM)
# ──────────────────────────────────────────────────────────────

def test_conflict_resolved_without_llm():
    sm = FakeSemanticMemory(["User's name is TestUser"])
    reconciler = MemoryReconciler(ExplodingProvider())
    candidate = KnowledgeCandidate(statement="User's name is Alex",
                                   confidence=0.8, category="extracted")
    results = reconciler.reconcile([candidate], sm)
    assert results["conflicts"] == 1
    contents = sm.contents()
    assert "User's name is TestUser" not in contents, "Stale name must be removed"
    assert "User's name is Alex" in contents, "New name must be stored"
    print("  [OK] Name conflict resolved deterministically (old removed, new kept)")


def test_duplicate_detected_without_llm():
    sm = FakeSemanticMemory(["User's name is Alex"])
    reconciler = MemoryReconciler(ExplodingProvider())
    candidate = KnowledgeCandidate(statement="User name is Alex",
                                   confidence=0.8, category="extracted")
    results = reconciler.reconcile([candidate], sm)
    assert results["duplicates"] == 1
    assert len(sm.contents()) == 1, "No duplicate entry should be added"
    print("  [OK] Duplicate (phrasing variant) detected deterministically")


def test_new_attribute_stored_without_llm():
    sm = FakeSemanticMemory(["User's name is Alex"])
    reconciler = MemoryReconciler(ExplodingProvider())
    candidate = KnowledgeCandidate(statement="User lives in Lisbon",
                                   confidence=0.8, category="extracted")
    results = reconciler.reconcile([candidate], sm)
    assert results["new_facts"] == 1
    assert "User lives in Lisbon" in sm.contents()
    print("  [OK] New single-valued attribute stored deterministically")


if __name__ == "__main__":
    tests = [
        test_stem_keeps_name_and_named_distinct,
        test_name_query_does_not_inject_dog,
        test_unrelated_query_returns_nothing,
        test_relevant_query_still_retrieves,
        test_capitalized_query_words_match,
        test_parse_single_valued_attributes,
        test_parse_defers_multivalued_and_unknown,
        test_name_variants_share_a_key,
        test_conflict_resolved_without_llm,
        test_duplicate_detected_without_llm,
        test_new_attribute_stored_without_llm,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
    print(f"\nResults: {len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
