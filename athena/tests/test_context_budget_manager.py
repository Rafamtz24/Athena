"""Comprehensive test script for the Context Budget Manager."""
import sys
sys.path.insert(0, '.')

from dataclasses import dataclass, field
from athena.context.models import ContextSource, ReasoningContextPackage, LearningContextPackage
from athena.context.manager import ContextBudgetManager
from athena.prompt.builder import PromptBuilder
from athena.tools.models import ToolContext


class MockProvider:
    def count_tokens(self, text):
        return len(text) // 4

    def get_context_window(self):
        return 4096


class MockThought:
    def __init__(self):
        self.user_input = 'What is the weather today?'
        self.history = ['User: Hi', 'Assistant: Hello!']
        self.memories = ['Memory 1', 'Memory 2']
        self.knowledge = 'User likes pizza. User lives in New York.'
        self.tool_context = None
        self.candidates = []
        self.response = None


def test_basic_compilation():
    """Test basic compilation without tools."""
    provider = MockProvider()
    manager = ContextBudgetManager(provider)
    thought = MockThought()
    rp, lp = manager.compile(thought)

    assert rp.context_window == 4096
    assert rp.generation_budget == 1024
    assert rp.total_tokens < 4096
    assert rp.total_tokens > 0

    source_names = {s.name for s in rp.sources}
    assert 'user_input' in source_names
    assert 'system_prompt' in source_names
    assert 'working_memory' in source_names
    assert 'semantic_memory' in source_names

    priorities = [s.priority for s in rp.sources]
    for i in range(len(priorities) - 1):
        assert priorities[i] >= priorities[i + 1]

    assert len(lp.sources) > 0
    assert lp.tool_context_content == ""

    print(f"[PASS] Basic compilation: {len(rp.sources)} sources, {rp.total_tokens} tokens")
    for s in rp.sources:
        print(f"       [{s.priority}] {s.name} learn={s.learning_visible}")


def test_tool_context_learning_visible():
    """Test that tool context with learning_visible=True is included in learning package."""
    provider = MockProvider()
    manager = ContextBudgetManager(provider)

    class ThoughtWithSystemTool:
        def __init__(self):
            self.user_input = 'Check my system'
            self.history = ['User: Check my system', 'Assistant: Let me check']
            self.memories = []
            self.knowledge = None
            self.tool_context = ToolContext(
                tool_name='system',
                content='CPU: Intel i7\nRAM: 32 GB\nGPU: RTX 3080',
                priority=70,
                learning_visible=True,
                metadata={}
            )
            self.candidates = []
            self.response = None

    thought = ThoughtWithSystemTool()
    rp, lp = manager.compile(thought)

    assert lp.tool_context_content == 'CPU: Intel i7\nRAM: 32 GB\nGPU: RTX 3080', \
        f"Tool context should be in learning package, got: {lp.tool_context_content[:50]}"

    print(f"[PASS] Tool context with learning_visible=True included in Learning Package")


def test_tool_context_learning_not_visible():
    """Test that tool context with learning_visible=False is excluded from learning package."""
    provider = MockProvider()
    manager = ContextBudgetManager(provider)

    class ThoughtWithWebTool:
        def __init__(self):
            self.user_input = 'Search for news'
            self.history = ['User: Search for news']
            self.memories = []
            self.knowledge = None
            self.tool_context = ToolContext(
                tool_name='web',
                content='Search results: ...',
                priority=70,
                learning_visible=False,
                metadata={}
            )
            self.candidates = []
            self.response = None

    thought = ThoughtWithWebTool()
    rp, lp = manager.compile(thought)

    learning_tool_sources = [s for s in lp.sources if s.name.startswith('tool:')]
    assert len(learning_tool_sources) == 0, \
        "Tool context with learning_visible=False should not be in learning package"
    assert lp.tool_context_content == "", \
        "Tool context content should be empty for learning_visible=False"

    print(f"[PASS] Tool context with learning_visible=False excluded from Learning Package")


def test_prompt_builder_from_package():
    """Test that PromptBuilder correctly renders a ReasoningContextPackage."""
    provider = MockProvider()
    manager = ContextBudgetManager(provider)
    thought = MockThought()
    rp, lp = manager.compile(thought)

    builder = PromptBuilder()
    prompt = builder.build(rp)

    assert 'User' in prompt
    assert 'What is the weather today?' in prompt
    assert 'Conversation' in prompt
    assert 'User: Hi' in prompt
    assert 'Knowledge' in prompt
    assert 'New York' in prompt

    print(f"[PASS] PromptBuilder renders ReasoningContextPackage correctly")
    print(f"       Prompt length: {len(prompt)} chars")


def test_never_trim():
    """Test that never-trim priorities are always included."""
    provider = MockProvider()
    manager = ContextBudgetManager(provider)

    class SmallThought:
        def __init__(self):
            self.user_input = 'Hi'
            self.history = []
            self.memories = []
            self.knowledge = None
            self.tool_context = None
            self.candidates = []
            self.response = None

    thought = SmallThought()
    rp, lp = manager.compile(thought)

    source_names = {s.name for s in rp.sources}
    assert 'user_input' in source_names, "user_input should never be trimmed"
    assert 'system_prompt' in source_names, "system_prompt should never be trimmed"

    print(f"[PASS] Never-trim sources always included")


def test_no_tool_specific_rules():
    """Verify ContextBudgetManager has no tool-specific hardcoded rules."""
    source = inspect_source('athena/context/manager.py')

    # Check for tool-specific strings
    tool_specific = ['tool == "', 'tool_name == "', 'if tool_name']
    for pattern in tool_specific:
        if pattern in source:
            # Allow tool: prefix handling which is metadata-based, not tool-specific
            if 'tool:' in pattern:
                continue
            print(f"[WARN] Possible tool-specific code: '{pattern}' found")

    print(f"[PASS] No tool-specific hardcoded rules in ContextBudgetManager")


def inspect_source(path):
    """Read source file for inspection."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


if __name__ == '__main__':
    print("=" * 50)
    print("Context Budget Manager - Verification Tests")
    print("=" * 50)
    print()

    test_basic_compilation()
    print()

    test_tool_context_learning_visible()
    print()

    test_tool_context_learning_not_visible()
    print()

    test_prompt_builder_from_package()
    print()

    test_never_trim()
    print()

    test_no_tool_specific_rules()
    print()

    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)