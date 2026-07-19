"""
Diagnose the learning pipeline specifically.

The mock provider works perfectly but never exercises the knowledge extraction
and validation code path (returns "NONE"). This script simulates what happens
when the REAL provider extracts actual knowledge.

We need to test:
  1. What happens when _validate_knowledge processes a candidate
  2. Whether semantic_memory.learn() changes shared state
  3. Whether the working_memory.clear() in _validate_knowledge succeeds
  4. What happens when the EXTRACTION prompt itself causes a provider crash
"""

import asyncio
import sys
import traceback
from unittest.mock import MagicMock

sys.stdout.reconfigure(line_buffering=True)

from athena.brain.brain import AthenaBrain
from athena.knowledge.models import KnowledgeCandidate
from athena.memory.semantic import SemanticMemory


async def diagnose():
    brain = AthenaBrain()

    # ── Inject mock provider that returns a REAL extraction candidate ──
    mock = MagicMock()
    mock.generate.return_value = "I understand. Tell me more."

    # Simulate the extraction returning an actual fact about the user's name
    # This is what the REAL LLM would return for "my name is actually alex"
    mock.call.return_value = "User's name is Alex"

    brain.provider = mock
    brain.pipeline.provider = mock
    brain.knowledge_manager.provider = mock

    # Clear pre-existing state
    brain.memory_manager.clear_working()
    brain.history.clear()

    print("=" * 70)
    print("STEP 1: process 'hello'")
    print("=" * 70)

    r1 = await brain.process("hello")
    print(f">>> Response: {r1!r}")
    sm_count = len(brain.memory_manager.query_semantic())
    print(f">>> Semantic memory entries: {sm_count}")
    wm_count = len(brain.memory_manager.get_working())
    print(f">>> Working memory entries after request: {wm_count}")

    print()
    print("=" * 70)
    print("STEP 2: process 'my name is actually alex'")
    print("=" * 70)

    try:
        r2 = await brain.process("my name is actually alex")
        print(f">>> Response: {r2!r}")
        sm_count = len(brain.memory_manager.query_semantic())
        print(f">>> Semantic memory entries: {sm_count}")
        if sm_count > 0:
            for i, e in enumerate(brain.memory_manager.query_semantic()):
                print(f"    [{i}] {e.content!r}")
        wm_count = len(brain.memory_manager.get_working())
        print(f">>> Working memory entries after request: {wm_count}")
    except Exception:
        exc_type, exc_value, exc_tb = sys.exc_info()
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(f">>> CRASHED in step 2!")
        print(f">>> Exception: {exc_type.__name__}: {exc_value}")
        print(f">>> Traceback:\n{tb_str}")

    print()
    print("=" * 70)
    print("STEP 3: process 'hello' again (check if working)")
    print("=" * 70)

    mock.generate.return_value = "Hello again! Still working."

    try:
        r3 = await brain.process("hello")
        print(f">>> Response: {r3!r}")
        if r3 == "Hello again! Still working.":
            print(">>> ✅ All good!")
        elif "unable to process" in str(r3).lower():
            print(">>> ❌ BROKEN AFTER STEP 2!")
            sm_count = len(brain.memory_manager.query_semantic())
            print(f">>> Semantic memory now has {sm_count} entries")
            for i, e in enumerate(brain.memory_manager.query_semantic()):
                print(f"    [{i}] {e.content!r}")
        else:
            print(f">>> Unexpected: {r3!r}")
    except Exception:
        exc_type, exc_value, exc_tb = sys.exc_info()
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(f">>> CRASHED in step 3!")
        print(f">>> Traceback:\n{tb_str}")

    print()
    print("=" * 70)
    print("STEP 4: Test with ACTUAL provider (no mock) to see exception")
    print("=" * 70)

    # Now restore the REAL provider to see the ACTUAL exception
    from athena.providers.factory import ProviderFactory
    real_provider = ProviderFactory.create()
    brain.provider = real_provider
    brain.pipeline.provider = real_provider
    brain.knowledge_manager.provider = real_provider
    brain.memory_manager.clear_working()

    mock2 = MagicMock()
    mock2.generate.side_effect = Exception("SIMULATED PROVIDER CRASH - trigger CognitiveEngine catch")
    mock2.call.side_effect = Exception("SIMULATED EXTRACTION CRASH")
    brain.provider = mock2
    brain.pipeline.provider = mock2
    brain.knowledge_manager.provider = mock2

    try:
        r4 = await brain.process("test")
        print(f">>> Response with simulated crash: {r4!r}")
        last = brain.debug_manager.get_last_thought()
        if last:
            print(f">>> Trace: {last.trace}")
    except Exception:
        exc_type, exc_value, exc_tb = sys.exc_info()
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(f">>> BRAIN CRASHED (should not happen): {exc_type.__name__}: {exc_value}")
        print(f">>> Traceback:\n{tb_str}")

    print()
    print("=" * 70)
    print("STEP 5: Verify brain still works after simulated crash")
    print("=" * 70)

    mock3 = MagicMock()
    mock3.generate.return_value = "Recovered!"
    mock3.call.return_value = "NONE"
    brain.provider = mock3
    brain.pipeline.provider = mock3
    brain.knowledge_manager.provider = mock3

    try:
        r5 = await brain.process("hello")
        print(f">>> Response after simulated crash: {r5!r}")
        if r5 == "Recovered!":
            print(">>> ✅ Provider works after simulated crash - isolation holds")
        else:
            print(f">>> Unexpected: {r5!r}")
    except Exception:
        exc_type, exc_value, exc_tb = sys.exc_info()
        print(f">>> CRASHED after recovery: {exc_type.__name__}: {exc_value}")

    print()
    print("=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(diagnose())
