"""Verify that _validate_knowledge can consume extract_candidates output."""
import sys
sys.path.insert(0, '.')

from athena.knowledge.manager import KnowledgeManager
from athena.context.models import ContextSource, LearningContextPackage
from athena.memory.working import WorkingMemory
from athena.memory.manager import MemoryManager


class MockProvider:
    def call(self, prompt):
        return "User prefers Python\nUser lives in Mexico"


def test_validate_knowledge_consumption():
    """Simulate the pipeline flow: extract -> validate."""
    wm = WorkingMemory()
    km = KnowledgeManager(working_memory=wm)
    km.provider = MockProvider()

    # Step 1: Extract candidates via LearningContextPackage
    conversation = "Hello\nI am Rafael"
    package = LearningContextPackage(
        sources=[ContextSource(name="conversation", content=conversation)],
        conversation=conversation,
        tool_context_content="",
    )
    extracted = km.extract_candidates(package)
    
    print(f"Extracted {len(extracted)} candidates from KnowledgeManager")
    
    # Step 2: Simulate what _validate_knowledge does
    # It calls memory_manager.get_candidates() which reads from WorkingMemory
    mm = MemoryManager()
    mm.working_memory = wm
    
    candidates = mm.get_candidates()
    print(f"Retrieved {len(candidates)} candidates via MemoryManager")
    
    # Step 3: Validate each candidate (from _validate_knowledge)
    for idx, candidate in enumerate(candidates):
        # Access .confidence, .statement, .category as done in pipeline
        has_confidence = hasattr(candidate, 'confidence')
        has_statement = hasattr(candidate, 'statement')
        has_category = hasattr(candidate, 'category')
        
        print(f"  Candidate {idx}: confidence={candidate.confidence}, "
              f"statement='{candidate.statement}', category='{candidate.category}'")
        
        assert has_confidence and has_statement and has_category, \
            f"Candidate missing required attributes"
        
        # This is the validation logic from pipeline line 243
        if candidate.confidence >= 0.7:
            print(f"    -> Would be promoted (confidence {candidate.confidence} >= 0.7)")
    
    print("\nVERIFICATION PASSED: _validate_knowledge can consume extract_candidates output")


if __name__ == "__main__":
    test_validate_knowledge_consumption()
