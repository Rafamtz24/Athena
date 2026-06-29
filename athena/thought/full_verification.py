"""Full verification that WorkingMemory candidates are exposed to Thought during processing."""
import asyncio
import sys
sys.path.insert(0, '.')

from athena.memory.working import WorkingMemory
from athena.thought.pipeline import ThoughtPipeline
from athena.thought.models import Thought


async def test_full_pipeline():
    """Test that candidates flow through the full pipeline."""
    # Setup working memory with candidates
    wm = WorkingMemory()
    wm.store_candidate('fact1', 0.8, 'test')
    wm.store_candidate('fact2', 0.9, 'test')
    
    class MockMM:
        def __init__(self, wm):
            self.working_memory = wm
        def get_candidates(self):
            return self.working_memory.get_candidates()
    
    mm = MockMM(wm)
    pipeline = ThoughtPipeline(memory_manager=mm)
    
    # Create thought and run through full process
    thought = Thought(user_input='test')
    
    print("=== BEFORE PROCESS ===")
    print(f"  thought.candidates: {thought.candidates}")
    
    await pipeline.process(thought)
    
    print("\n=== AFTER PROCESS ===")
    print(f"  thought.metadata.get('stage'): {thought.metadata.get('stage')}")
    print(f"  thought.candidates type: {type(thought.candidates)}")
    print(f"  Number of candidates: {len(thought.candidates)}")
    
    for i, c in enumerate(thought.candidates):
        print(f"    [{i}] Statement: '{c.statement}', Confidence: {c.confidence}, Category: '{c.category}'")
    
    # Assertions
    assert len(thought.candidates) == 2, f"Expected 2 candidates, got {len(thought.candidates)}"
    assert thought.candidates[0].statement == 'fact1', f"First candidate statement mismatch"
    assert thought.candidates[1].statement == 'fact2', f"Second candidate statement mismatch"
    
    print("\n✓ VERIFICATION PASSED: WorkingMemory candidates are correctly exposed to Thought.candidates")


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
