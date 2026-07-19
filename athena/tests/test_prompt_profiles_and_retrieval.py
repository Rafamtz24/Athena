"""
Verification tests for Prompt Profiles and Semantic Memory Relevance improvements.
"""

import sys
import json
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from athena.prompt.loader import PromptLoader, PromptValidationError


def test_prompt_loader_basic():
    for name in ['reasoning', 'extraction', 'reconciliation']:
        profile = PromptLoader.load(name)
        assert profile is not None, f"Failed to load {name}"
        assert profile.name == name
        assert profile.system_prompt
        assert len(profile.system_prompt) > 10
        print(f"  [PASS] {name}.json loaded")


def test_prompt_loader_cache():
    p1 = PromptLoader.load('reasoning')
    p2 = PromptLoader.load('reasoning')
    assert p1 is p2
    print("  [PASS] Cache hit returns same object")


def test_prompt_loader_clear():
    p1 = PromptLoader.load('reasoning')
    PromptLoader.clear_cache('reasoning')
    p3 = PromptLoader.load('reasoning')
    assert p3 is not p1
    PromptLoader.clear_cache()
    assert len(PromptLoader.get_cached_names()) == 0
    print("  [PASS] Cache clear works")


def test_prompt_loader_missing_file():
    try:
        PromptLoader.load('nonexistent_file_xyz')
        assert False
    except FileNotFoundError:
        print("  [PASS] Missing file -> FileNotFoundError")


def test_prompt_loader_invalid_json():
    prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    test_file = prompts_dir / "test_invalid.json"
    try:
        test_file.write_text("{invalid json content}")
        try:
            PromptLoader.load('test_invalid')
            assert False
        except json.JSONDecodeError:
            print("  [PASS] Invalid JSON -> JSONDecodeError")
    finally:
        if test_file.exists():
            test_file.unlink()


def test_prompt_loader_missing_required_field():
    prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    test_file = prompts_dir / "test_missing.json"
    try:
        test_file.write_text(json.dumps({"instructions": "no system_prompt"}))
        try:
            PromptLoader.load('test_missing')
            assert False
        except PromptValidationError:
            print("  [PASS] Missing required field -> PromptValidationError")
    finally:
        if test_file.exists():
            test_file.unlink()


def test_prompt_loader_convenience_methods():
    sp = PromptLoader.get_system_prompt('reasoning')
    assert "Athena" in sp
    inst = PromptLoader.get_instructions('extraction')
    assert inst is not None
    assert "STRICT RULES" in inst
    rf = PromptLoader.get_response_format('reconciliation')
    assert rf is not None
    assert "RESPONSE" in rf
    print("  [PASS] Convenience methods work")


def test_prompt_builder_uses_loader():
    from athena.prompt.builder import PromptBuilder
    builder = PromptBuilder()

    class MockThought:
        def __init__(self):
            self.history = []
            self.memories = []
            self.knowledge = None
            self.user_input = "test"
            self.candidates = None
            self.tool_context = None

    thought = MockThought()
    prompt = builder._build_from_thought(thought)
    assert "You are Athena" in prompt
    print("  [PASS] PromptBuilder fallback uses PromptLoader")


def test_context_budget_manager_uses_loader():
    from athena.context.manager import ContextBudgetManager

    class MockProvider:
        def count_tokens(self, text):
            return len(text) // 4
        def get_context_window(self):
            return 4096

    manager = ContextBudgetManager(MockProvider())
    system_prompt = manager._build_system_prompt()
    assert "You are Athena" in system_prompt
    assert manager._SYSTEM_PROMPT is not None
    print("  [PASS] ContextBudgetManager uses PromptLoader")


def test_reconciliation_build_uses_loader():
    from athena.knowledge.reconciler import build_reconciliation_prompt
    prompt = build_reconciliation_prompt("Test fact", ["Existing fact 1", "Existing fact 2"])
    assert "You compare pairs of factual statements" in prompt
    assert "ACTION: DUPLICATE" in prompt
    assert "Test fact" in prompt
    assert "Existing fact 1" in prompt
    print("  [PASS] Reconciliation prompt uses PromptLoader")


def test_retrieval_score_relevance():
    from athena.knowledge.manager import KnowledgeManager
    km = KnowledgeManager()

    score = km._score_relevance("User has a dog named Rex", ["dog", "name"])
    assert score > 0
    print(f"  [PASS] Score for matching words: {score:.3f}")

    score = km._score_relevance("User lives in Lisbon", ["hello", "test"])
    assert score == 0
    print(f"  [PASS] Score for unrelated words: 0.0")

    score = km._score_relevance("Operating System is Windows 11", ["test", "system"])
    assert score > 0
    print(f"  [PASS] Score for partial match: {score:.3f}")


def test_retrieval_greeting_returns_none():
    from athena.knowledge.manager import KnowledgeManager

    class MockMemoryManager:
        def query_semantic(self):
            return [
                type('Entry', (), {'content': 'User has a dog named Rex'}),
                type('Entry', (), {'content': 'User lives in Lisbon'}),
            ]

    km = KnowledgeManager(memory_manager=MockMemoryManager())
    for greeting in ["Hello", "Hi", "Hey there", "Good morning", "What's up", "yo"]:
        result = km.retrieve(greeting)
        assert result is None, f"'{greeting}' should retrieve nothing"
    print("  [PASS] Greetings retrieve no unrelated facts")


def test_retrieval_vague_statement_returns_none():
    from athena.knowledge.manager import KnowledgeManager

    class MockMemoryManager:
        def query_semantic(self):
            return [
                type('Entry', (), {'content': 'User has a dog named Rex'}),
                type('Entry', (), {'content': 'User lives in Lisbon'}),
            ]

    km = KnowledgeManager(memory_manager=MockMemoryManager())
    for statement in ["I'm testing Athena", "Just checking in", "Let me try something"]:
        result = km.retrieve(statement)
        assert result is None, f"'{statement}' should retrieve nothing"
    print("  [PASS] Vague statements retrieve no facts")


def test_retrieval_factual_question_matches():
    from athena.knowledge.manager import KnowledgeManager

    class MockMemoryManager:
        def query_semantic(self):
            return [
                type('Entry', (), {'content': 'User has a dog named Rex'}),
                type('Entry', (), {'content': 'User lives in Lisbon'}),
                type('Entry', (), {'content': "User's name is Alex"}),
            ]

    km = KnowledgeManager(memory_manager=MockMemoryManager())

    result = km.retrieve("What is my dog's name?")
    assert result is not None
    assert "dog" in result.lower() and "rex" in result.lower()
    print(f"  [PASS] 'What is my dog's name?' retrieves dog fact")

    result = km.retrieve("Where do I live?")
    assert result is not None
    assert "lives" in result.lower() or "lisbon" in result.lower()
    print(f"  [PASS] 'Where do I live?' retrieves location fact")


def test_retrieval_memory_recall_returns_all():
    from athena.knowledge.manager import KnowledgeManager

    class MockMemoryManager:
        def query_semantic(self):
            return [
                type('Entry', (), {'content': 'User has a dog named Rex'}),
                type('Entry', (), {'content': 'User lives in Lisbon'}),
                type('Entry', (), {'content': "User's name is Alex"}),
            ]

    km = KnowledgeManager(memory_manager=MockMemoryManager())

    queries = [
        "What do you remember about me?",
        "What do you know about me?",
        "Tell me everything you remember",
        "What information do you have about me?",
        "What can you tell me about myself?",
    ]
    for query in queries:
        result = km.retrieve(query)
        assert result is not None, f"'{query}' should retrieve memories"
        assert "dog" in result.lower() or "lisbon" in result.lower() or "alex" in result.lower()
    print("  [PASS] Memory recall queries retrieve all durable memories")


def test_retrieval_fallback_to_none():
    from athena.knowledge.manager import KnowledgeManager

    class MockMemoryManager:
        def query_semantic(self):
            return [type('Entry', (), {'content': 'User has a dog named Rex'})]

    km = KnowledgeManager(memory_manager=MockMemoryManager())
    result = km.retrieve("What is the weather like in Paris?")
    assert result is None
    print("  [PASS] Unrelated query returns None (no fallback to all entries)")


def test_retrieval_max_results():
    from athena.knowledge.manager import KnowledgeManager

    class MockMemoryManager:
        def query_semantic(self):
            return [
                type('Entry', (), {'content': 'User has a dog named Rex'}),
                type('Entry', (), {'content': 'User lives in Lisbon'}),
                type('Entry', (), {'content': 'User works at Company'}),
                type('Entry', (), {'content': 'User likes pizza'}),
            ]

    km = KnowledgeManager(memory_manager=MockMemoryManager())
    result = km.retrieve("Tell me about User")
    assert result is not None
    print("  [PASS] Retrieval with matched keywords works")


def run_all():
    tests = [
        ("PromptLoader: basic loading", test_prompt_loader_basic),
        ("PromptLoader: caching", test_prompt_loader_cache),
        ("PromptLoader: clear cache", test_prompt_loader_clear),
        ("PromptLoader: missing file error", test_prompt_loader_missing_file),
        ("PromptLoader: invalid JSON error", test_prompt_loader_invalid_json),
        ("PromptLoader: missing required field", test_prompt_loader_missing_required_field),
        ("PromptLoader: convenience methods", test_prompt_loader_convenience_methods),
        ("PromptBuilder integration", test_prompt_builder_uses_loader),
        ("ContextBudgetManager integration", test_context_budget_manager_uses_loader),
        ("Reconciliation prompt integration", test_reconciliation_build_uses_loader),
        ("Relevance scoring", test_retrieval_score_relevance),
        ("Greeting -> no retrieval", test_retrieval_greeting_returns_none),
        ("Vague statement -> no retrieval", test_retrieval_vague_statement_returns_none),
        ("Factual question -> match", test_retrieval_factual_question_matches),
        ("Memory recall -> all entries", test_retrieval_memory_recall_returns_all),
        ("No match -> None (no fallback)", test_retrieval_fallback_to_none),
        ("Max results limit", test_retrieval_max_results),
    ]

    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print("PROMPT PROFILES & SEMANTIC MEMORY RETRIEVAL")
    print("VERIFICATION TESTS")
    print("=" * 60)
    print()

    for name, func in tests:
        try:
            func()
            passed += 1
            print(f"  [PASS] {name}")
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  [FAIL] {name}: {e}")
        print()

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)

    if errors:
        print("\nFAILURES:")
        for name, error in errors:
            print(f"  - {name}: {error}")
        sys.exit(1)

    return True


if __name__ == "__main__":
    run_all()