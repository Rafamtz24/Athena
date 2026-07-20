"""Verify that WorkingMemory candidates are exposed to Thought."""
import asyncio
import sys
sys.path.insert(0, '.')

from athena.memory.working import WorkingMemory
from athena.thought.pipeline import ThoughtPipeline
from athena.thought.models import Thought


async def test():
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

    thought = Thought(user_input='test')
    print('Before _load_candidates:')
    print(f'  thought.candidates: {thought.candidates}')

    pipeline._load_candidates(thought)

    print('After _load_candidates:')
    print(f'  thought.candidates type: {type(thought.candidates)}')
    print(f'  Number of candidates: {len(thought.candidates)}')
    for c in thought.candidates:
        print(f'    - Statement: {c.statement}, Confidence: {c.confidence}')

    # Verify the facts are accessible from thought
    assert len(thought.candidates) == 2, f"Expected 2 candidates, got {len(thought.candidates)}"
    assert thought.candidates[0].statement == 'fact1', f"Expected 'fact1', got '{thought.candidates[0].statement}'"
    print("\nVERIFICATION PASSED: WorkingMemory candidates are exposed to Thought.candidates")


if __name__ == "__main__":
    asyncio.run(test())
