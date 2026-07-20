"""Full flow verification: WorkingMemory candidates -> Thought.candidates."""
import asyncio
import sys
sys.path.insert(0, '.')


async def test_full_flow():
    from athena.memory.manager import MemoryManager
    from athena.thought.pipeline import ThoughtPipeline

    # Setup memory manager with candidates in working memory
    mm = MemoryManager()
    mm.working_memory.store_candidate('fact1', 0.8, 'test')
    mm.working_memory.store_candidate('fact2', 0.9, 'test')

    provider = type('MockProvider', (), {'generate': lambda s, p: 'Mock response'})()
    
    # Create pipeline with real MemoryManager and mock provider
    from athena.knowledge.manager import KnowledgeManager
    km = KnowledgeManager(working_memory=mm.working_memory, provider=provider)
    
    pipeline = ThoughtPipeline(mm, km, provider)
    
    thought_type = __import__('athena.thought.models', fromlist=['Thought']).Thought
    thought = thought_type(user_input='test')
    
    print("=== BEFORE PROCESS ===")
    print(f"  thought.candidates: {thought.candidates}")
    print(f"  WorkingMemory candidates count: {mm.working_memory.get_candidates().__len__()}")
    
    await pipeline.process(thought)
    
    print("\n=== AFTER PROCESS ===")
    print(f"  thought.metadata.get('stage'): {thought.metadata.get('stage')}")
    print(f"  thought.candidates type: {type(thought.candidates)}")
    print(f"  Number of candidates loaded into Thought: {len(thought.candidates)}")
    
    for i, c in enumerate(thought.candidates):
        print(f"    [{i}] Statement: '{c.statement}', Confidence: {c.confidence}, Category: '{c.category}'")
    
    # Verify the facts are accessible from thought
    assert len(thought.candidates) == 2, f"Expected 2 candidates, got {len(thought.candidates)}"
    assert thought.candidates[0].statement == 'fact1', f"First candidate statement mismatch"
    assert thought.candidates[1].statement == 'fact2', f"Second candidate statement mismatch"
    
    print("\nVERIFICATION PASSED: WorkingMemory candidates are correctly exposed to Thought.candidates")


if __name__ == "__main__":
    asyncio.run(test_full_flow())
