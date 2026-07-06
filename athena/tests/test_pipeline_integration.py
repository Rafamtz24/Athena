"""Integration test: Pipeline with Context Budget Manager."""
import sys
sys.path.insert(0, '.')

from athena.thought.models import Thought
from athena.thought.pipeline import ThoughtPipeline
from athena.tools.models import ToolContext


class MockProvider:
    def count_tokens(self, text):
        return len(text) // 4

    def get_context_window(self):
        return 4096

    def generate(self, prompt, system=None):
        return "This is a mock response."

    def call(self, prompt):
        return "This is a mock response."


class MockMemoryManager:
    class WorkingMemory:
        def retrieve(self):
            return []
        def store_candidate(self, **kwargs):
            pass
        def get_candidates(self):
            return []
        def clear(self):
            pass
        def prune(self, max_tokens, entries=None):
            if entries is not None:
                # Simulate pruning by clearing when budget is very small
                if max_tokens <= 0:
                    entries.clear()
                else:
                    # Basic token estimation pruning
                    total = 0
                    for i in range(len(entries) - 1, -1, -1):
                        total += len(entries[i]) // 4
                        if total > max_tokens:
                            del entries[:i + 1]
                            break
    working_memory = WorkingMemory()

    def recall(self):
        return []
    def query_semantic(self):
        return []
    def get_candidates(self):
        return []
    def clear_working(self):
        pass


class MockKnowledgeManager:
    def __init__(self):
        self.last_package = None

    def retrieve(self, query):
        return None

    def extract_candidates(self, package):
        self.last_package = package
        return []


def test_pipeline_with_budget_manager():
    """Test that the pipeline correctly integrates the Context Budget Manager."""
    provider = MockProvider()
    memory_manager = MockMemoryManager()
    knowledge_manager = MockKnowledgeManager()

    pipeline = ThoughtPipeline(
        memory_manager=memory_manager,
        knowledge_manager=knowledge_manager,
        provider=provider,
    )

    thought = Thought(user_input="Hello, how are you?")
    thought.history = ["User: Hi", "Assistant: Hello!"]
    thought.tool_context = ToolContext(
        tool_name="system",
        content="CPU: Intel i7\nRAM: 32 GB",
        priority=70,
        learning_visible=True,
        metadata={}
    )

    # Run the pipeline
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        response = loop.run_until_complete(pipeline.process(thought))
    finally:
        loop.close()

    # Verify reasoning package was created
    assert thought.reasoning_package is not None, "Reasoning package should be set"
    assert thought.reasoning_package.total_tokens > 0, "Reasoning package should have tokens"
    assert thought.reasoning_package.context_window == 4096, "Context window should be 4096"
    assert thought.reasoning_package.generation_budget == 1024, "Generation budget should be 1024"

    # Verify learning package was created
    assert thought.learning_package is not None, "Learning package should be set"

    # Verify response was generated
    assert thought.get_response() is not None, "Response should be set"
    assert thought.get_response() == "This is a mock response.", \
        f"Unexpected response: {thought.get_response()}"

    # Verify knowledge manager received the Learning Package directly
    assert knowledge_manager.last_package is not None, \
        "Knowledge manager should have received a LearningContextPackage"
    assert hasattr(knowledge_manager.last_package, 'conversation'), \
        "Package should have conversation field"
    assert "Hello, how are you?" in knowledge_manager.last_package.conversation, \
        "Package conversation should contain user input"

    print(f"[PASS] Pipeline integration: reasoning_package.tokens={thought.reasoning_package.total_tokens}")
    print(f"[PASS] Pipeline integration: learning_package.sources={len(thought.learning_package.sources)}")
    print(f"[PASS] Pipeline integration: response='{thought.get_response()}'")
    print(f"[PASS] Pipeline integration: KnowledgeManager received LearningContextPackage directly")


if __name__ == '__main__':
    print("=" * 50)
    print("Pipeline Integration Test")
    print("=" * 50)
    print()
    test_pipeline_with_budget_manager()
    print()
    print("=" * 50)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 50)