"""
Verification tests for performance optimizations in the Athena Core v1 milestone.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from athena.knowledge.manager import _is_extraction_needed, KnowledgeManager
from athena.context.models import LearningContextPackage

def test_skip_greetings():
    assert not _is_extraction_needed("hi")
    assert not _is_extraction_needed("hello")
    assert not _is_extraction_needed("hey there")
    assert not _is_extraction_needed("good morning")
    assert not _is_extraction_needed("Hi")
    assert not _is_extraction_needed("Hello!")
    print("  [OK] Greetings correctly skipped")

def test_skip_thanks():
    assert not _is_extraction_needed("thanks")
    assert not _is_extraction_needed("thank you")
    assert not _is_extraction_needed("ty")
    assert not _is_extraction_needed("thx")
    assert not _is_extraction_needed("appreciate it")
    print("  [OK] Thanks correctly skipped")

def test_skip_goodbyes():
    assert not _is_extraction_needed("goodbye")
    assert not _is_extraction_needed("bye")
    assert not _is_extraction_needed("see you")
    assert not _is_extraction_needed("take care")
    print("  [OK] Goodbyes correctly skipped")

def test_skip_acknowledgements():
    assert not _is_extraction_needed("ok")
    assert not _is_extraction_needed("okay")
    assert not _is_extraction_needed("got it")
    assert not _is_extraction_needed("i see")
    assert not _is_extraction_needed("yeah")
    assert not _is_extraction_needed("no")
    assert not _is_extraction_needed("sure")
    print("  [OK] Acknowledgements correctly skipped")

def test_skip_empty():
    assert not _is_extraction_needed("")
    assert not _is_extraction_needed("   ")
    print("  [OK] Empty input correctly skipped")

def test_run_extraction_for_learnable():
    assert _is_extraction_needed("My name is Rafael.")
    assert _is_extraction_needed("I live in Monterrey.")
    assert _is_extraction_needed("I prefer Linux.")
    assert _is_extraction_needed("My favorite color is blue.")
    assert _is_extraction_needed("I have a dog named Gemma.")
    assert _is_extraction_needed("What is the weather like today?")
    assert _is_extraction_needed("Tell me about quantum computing.")
    print("  [OK] Learnable interactions correctly identified for extraction")

def test_system_prompt_cached():
    from athena.context.manager import ContextBudgetManager
    class MockProvider:
        def count_tokens(self, text): return len(text) // 4
        def get_context_window(self): return 4096
    mgr = ContextBudgetManager(MockProvider())
    p1 = mgr._build_system_prompt()
    p2 = mgr._build_system_prompt()
    assert p1 == p2
    assert p1 == mgr._SYSTEM_PROMPT
    print("  [OK] System prompt correctly cached as class constant")

def test_extraction_prompt_template_cached():
    km = KnowledgeManager(None, None, None)
    assert hasattr(km, '_EXTRACTION_TEMPLATE_PREFIX')
    assert len(km._EXTRACTION_TEMPLATE_PREFIX) > 100
    print("  [OK] Extraction prompt template cached correctly")

def test_prompt_builder_headers_cached():
    from athena.prompt.builder import PromptBuilder
    pb = PromptBuilder()
    assert hasattr(pb, '_HEADER_USER_INPUT')
    assert hasattr(pb, '_HEADER_WORKING_MEMORY')
    assert hasattr(pb, '_HEADER_SEMANTIC_MEMORY')
    assert hasattr(pb, '_HEADER_CANDIDATE_FACTS')
    assert hasattr(pb, '_HEADER_CHAT_HISTORY')
    assert hasattr(pb, '_HEADER_TOOL_PREFIX')
    assert hasattr(pb, '_HEADER_TOOL_SUFFIX')
    print("  [OK] PromptBuilder section headers cached correctly")

def test_reconciliation_prompt_prefix_cached():
    import athena.knowledge.reconciler as r
    assert hasattr(r, '_RECONCILIATION_PROMPT_PREFIX')
    assert hasattr(r, '_RECONCILIATION_PROMPT_SUFFIX')
    assert len(r._RECONCILIATION_PROMPT_PREFIX) > 100
    print("  [OK] Reconciliation prompt prefix cached correctly")

def test_token_count_caching():
    from athena.context.manager import ContextBudgetManager
    class CountingProvider:
        def __init__(self):
            self.count_calls = 0
        def count_tokens(self, text):
            self.count_calls += 1
            return len(text) // 4
        def get_context_window(self): return 4096
    provider = CountingProvider()
    mgr = ContextBudgetManager(provider)
    mgr._token_cache.clear()
    c1 = mgr._count_tokens("hello world")
    assert provider.count_calls == 1
    c2 = mgr._count_tokens("hello world")
    assert provider.count_calls == 1, "Should use cache"
    assert c1 == c2
    c3 = mgr._count_tokens("different text")
    assert provider.count_calls == 2
    print("  [OK] Token count caching works correctly")

def test_extraction_gate_fast_path():
    class NoCallProvider:
        def call(self, prompt):
            raise AssertionError("Provider should NOT be called for greetings")
    km = KnowledgeManager(None, NoCallProvider(), None)
    pkg = LearningContextPackage(sources=[], conversation="User: hi", tool_context_content="")
    result = km.extract_candidates(pkg)
    assert result == []
    print("  [OK] Greeting fast path prevents provider call")

if __name__ == "__main__":
    print("=" * 60)
    print("PERFORMANCE OPTIMIZATION VERIFICATION")
    print("=" * 60)
    tests = [
        ("Opt 1", "Greeting detection", test_skip_greetings),
        ("Opt 1", "Thanks detection", test_skip_thanks),
        ("Opt 1", "Goodbye detection", test_skip_goodbyes),
        ("Opt 1", "Acknowledgement detection", test_skip_acknowledgements),
        ("Opt 1", "Empty input detection", test_skip_empty),
        ("Opt 1", "Learnable extraction", test_run_extraction_for_learnable),
        ("Opt 4", "System prompt cached", test_system_prompt_cached),
        ("Opt 4", "Extraction template cached", test_extraction_prompt_template_cached),
        ("Opt 4", "PromptBuilder headers cached", test_prompt_builder_headers_cached),
        ("Opt 4", "Reconciliation prefix cached", test_reconciliation_prompt_prefix_cached),
        ("Opt 6", "Token count caching", test_token_count_caching),
        ("Opt 7", "Extraction gate fast path", test_extraction_gate_fast_path),
    ]
    failed = 0
    for opt, name, func in tests:
        try:
            func()
        except Exception as e:
            print(f"  [FAIL] [{opt}] {name}: {e}")
            failed += 1
    print()
    total = len(tests)
    passed = total - failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    sys.exit(1 if failed > 0 else 0)