"""
Deterministic diagnostic — uses a mock provider that ALWAYS works,
to trace the exact pipeline path and identify state corruption.

We need to determine:
  - Which pipeline stage processes "my name is actually alex" differently
    than "hello"
  - Whether any stage corrupts shared state (provider, working memory, 
    semantic memory) that affects subsequent requests

A mock provider gives us 100% reproducible behavior regardless of
the local LLM setup.
"""

import asyncio
import sys
import traceback
from unittest.mock import MagicMock

sys.stdout.reconfigure(line_buffering=True)

from athena.brain.brain import AthenaBrain


def _count_thought_fields(thought, label):
    """Print key thought fields for diagnostic purposes."""
    fields = {
        "user_input": repr(thought.user_input),
        "planner_decision": thought.planner_decision,
        "tool_context": thought.tool_context,
        "response": repr(thought.get_response()),
        "metadata[stage]": thought.metadata.get("stage"),
        "num_history": len(thought.history),
        "knowledge_len": len(str(thought.knowledge)) if thought.knowledge else 0,
        "trace_keys": list(thought.trace.keys()),
        "metadata_keys": list(thought.metadata.keys()),
    }
    print(f"\n  [{label}] Thought state:")
    for k, v in fields.items():
        print(f"    {k}: {v!r}")


async def diagnose():
    brain = AthenaBrain()

    # ── Inject a mock provider that always works ──
    mock = MagicMock()
    mock.generate.return_value = "I understand. Tell me more."
    mock.call.return_value = "NONE"
    brain.provider = mock
    brain.pipeline.provider = mock
    brain.knowledge_manager.provider = mock

    # Clear any pre-existing data
    brain.memory_manager.clear_working()
    brain.history.clear()

    print("=" * 70)
    print("REQUEST 1: 'hello'")
    print("=" * 70)

    try:
        r1 = await brain.process("hello")
        print(f">>> Response 1: {r1!r}")
        print(f">>> brain.history count: {len(brain.history)}")
        print(f">>> brain.memory_manager.get_working() count: {len(brain.memory_manager.get_working())}")
        print(f">>> semantic memory count: {len(brain.memory_manager.query_semantic())}")
    except Exception:
        exc_type, exc_value, exc_tb = sys.exc_info()
        print(f">>> CRASHED: {exc_type.__name__}: {exc_value}")
        traceback.print_exc()

    print()
    print("=" * 70)
    print("REQUEST 2: 'my name is actually alex'")
    print("=" * 70)

    try:
        r2 = await brain.process("my name is actually alex")
        print(f">>> Response 2: {r2!r}")
        print(f">>> brain.history count: {len(brain.history)}")
        print(f">>> brain.memory_manager.get_working() count: {len(brain.memory_manager.get_working())}")
        print(f">>> semantic memory count: {len(brain.memory_manager.query_semantic())}")
        # Show what's in semantic memory
        for i, entry in enumerate(brain.memory_manager.query_semantic()):
            content = entry.content if hasattr(entry, 'content') else str(entry)
            print(f"    semantic[{i}]: {content!r}")
    except Exception:
        exc_type, exc_value, exc_tb = sys.exc_info()
        print(f">>> CRASHED: {exc_type.__name__}: {exc_value}")
        traceback.print_exc()

    print()
    print("=" * 70)
    print("REQUEST 3: 'hello' (post-corruption check)")
    print("=" * 70)

    # Change mock response so we can tell if the same mock is being called
    mock.generate.return_value = "Hello again!"

    try:
        r3 = await brain.process("hello")
        print(f">>> Response 3: {r3!r}")
        if r3 == "Hello again!":
            print(">>> ✅ Provider is still working — no corruption detected")
        elif r3 == "I'm sorry, I'm currently unable to process your request.":
            print(">>> ❌ FAILURE: Provider appears broken even though it's a mock!")
        else:
            print(f">>> ⚠️ Unexpected response: {r3!r}")
    except Exception:
        exc_type, exc_value, exc_tb = sys.exc_info()
        print(f">>> CRASHED: {exc_type.__name__}: {exc_value}")
        traceback.print_exc()

    print()
    print("=" * 70)
    print("DEEP DIAGNOSTIC: trace the last thought through each stage")
    print("=" * 70)

    last = brain.debug_manager.get_last_thought()
    if last:
        _count_thought_fields(last, "LAST_THOUGHT")
        print(f"\n  Full trace: {json.dumps(last.trace, indent=2, default=str)}")
        print(f"\n  Full metadata: {json.dumps(last.metadata, indent=2, default=str)}")
    else:
        print("  No thought in debug manager!")

    print()
    print("=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import json
    asyncio.run(diagnose())
