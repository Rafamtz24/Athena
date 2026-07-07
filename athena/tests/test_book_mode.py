"""
Tests for book reading mode (/book).

Reading mode answers a question grounded strictly in a selected PDF, via a
separate path from the Thought pipeline: no tools, no memory, no extraction.
These tests exercise chunking, keyword retrieval, prompt assembly, and the
answer path using synthetic text and a mock provider (no PDF/model needed).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from athena.books import library as lib
from athena.prompt.loader import PromptLoader


class MockProvider:
    """Records the last generate() call; no model loaded."""
    def __init__(self):
        self.calls = 0
        self.last_system = None
        self.last_prompt = None

    def count_tokens(self, text):
        return max(1, len(text) // 4)

    def get_context_window(self):
        return 4096

    def generate(self, prompt, system=None):
        self.calls += 1
        self.last_prompt = prompt
        self.last_system = system
        return "[mock answer]"


def _sample_chunks():
    return [
        "Logotherapy is a school of psychotherapy founded by Viktor Frankl "
        "that centers on the human search for meaning.",
        "The dog ran across the sunny park chasing a red ball all afternoon.",
        "Frankl described how meaning can be found even in unavoidable suffering.",
        "A recipe for bread requires flour water yeast and salt mixed together.",
    ]


def test_chunk_text_windows_and_overlap():
    words = " ".join(f"w{i}" for i in range(500))
    chunks = lib.chunk_text(words, target_words=100, overlap_words=20)
    assert len(chunks) > 1
    # Each chunk is at most target_words long.
    assert all(len(c.split()) <= 100 for c in chunks)
    # Consecutive chunks overlap (step = 80, so chunk 2 starts at word 80).
    assert chunks[1].split()[0] == "w80"
    print("  [OK] chunk_text produces overlapping fixed-size windows")


def test_chunk_text_empty():
    assert lib.chunk_text("") == []
    assert lib.chunk_text("   ") == []
    print("  [OK] chunk_text handles empty input")


def test_retrieve_finds_relevant_chunk():
    provider = MockProvider()
    passages = lib.retrieve_relevant(
        _sample_chunks(), "What is logotherapy?", provider.count_tokens, 2000
    )
    assert passages, "Expected at least one passage"
    # The logotherapy chunk must be selected; the bread recipe must not.
    assert any("Logotherapy" in p for p in passages)
    assert not any("bread" in p for p in passages)
    print("  [OK] retrieval selects keyword-relevant chunks, drops irrelevant ones")


def test_retrieve_falls_back_to_opening_when_no_match():
    chunks = _sample_chunks()
    # A question with no lexical overlap -> fall back to the opening chunk.
    passages = lib.retrieve_relevant(
        chunks, "quantum chromodynamics", MockProvider().count_tokens, 2000
    )
    assert passages and passages[0] == chunks[0]
    print("  [OK] retrieval falls back to the book opening on no match")


def test_retrieve_respects_budget():
    chunks = [f"passage {i} " + ("filler " * 50) for i in range(10)]
    # Tiny budget should admit only the first-selected chunk.
    passages = lib.retrieve_relevant(
        chunks, "passage", MockProvider().count_tokens, budget_tokens=1
    )
    assert len(passages) == 1
    print("  [OK] retrieval respects the token budget")


def test_build_book_prompt_contains_passages_and_question():
    prompt = lib.build_book_prompt(["alpha passage", "beta passage"], "why?")
    assert "alpha passage" in prompt and "beta passage" in prompt
    assert "Question: why?" in prompt
    print("  [OK] build_book_prompt includes passages and the question")


def test_answer_from_book_uses_book_prompt_single_call():
    provider = MockProvider()
    answer = lib.answer_from_book(provider, _sample_chunks(), "What is logotherapy?")
    assert answer == "[mock answer]"
    # Exactly one provider call (no tools / no memory / no extra generations).
    assert provider.calls == 1
    assert provider.last_system is not None
    assert "reading mode" in provider.last_system.lower()
    print("  [OK] answer_from_book: one grounded call with the book system prompt")


def test_answer_from_book_no_chunks():
    provider = MockProvider()
    answer = lib.answer_from_book(provider, [], "anything?")
    assert "couldn't find" in answer.lower()
    assert provider.calls == 0
    print("  [OK] answer_from_book returns a graceful message with no content")


def test_book_prompt_profile_loads():
    system_prompt = PromptLoader.get_system_prompt("book")
    assert "ONLY" in system_prompt
    assert "reading mode" in system_prompt.lower()
    print("  [OK] book.json prompt profile loads")


def test_list_books_returns_list():
    # Smoke test — must not raise and must return a list (possibly empty).
    assert isinstance(lib.list_books(), list)
    print("  [OK] list_books returns a list")


if __name__ == "__main__":
    tests = [
        test_chunk_text_windows_and_overlap,
        test_chunk_text_empty,
        test_retrieve_finds_relevant_chunk,
        test_retrieve_falls_back_to_opening_when_no_match,
        test_retrieve_respects_budget,
        test_build_book_prompt_contains_passages_and_question,
        test_answer_from_book_uses_book_prompt_single_call,
        test_answer_from_book_no_chunks,
        test_book_prompt_profile_loads,
        test_list_books_returns_list,
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
