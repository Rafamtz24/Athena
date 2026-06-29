"""Test to verify KnowledgeManager.extract_candidates() returns consistent structure."""
import sys
sys.path.insert(0, '.')

from athena.knowledge.manager import KnowledgeManager
from athena.memory.working import WorkingMemory
from athena.knowledge.models import KnowledgeCandidate


class MockProvider:
    """Mock provider that returns fake extraction results."""
    def call(self, prompt):
        return "User prefers Python\nUser lives in Mexico\nUser works with Athena"


def test_extract_candidates_structure():
    wm = WorkingMemory()
    km = KnowledgeManager(working_memory=wm)
    km.provider = MockProvider()

    conversation = "Hello\nI am Rafael\nI live in Mexico\nI work on AI projects"
    
    result = km.extract_candidates(conversation)
    
    # Verify return type is list
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    
    # Verify each candidate has consistent structure
    for candidate in result:
        assert hasattr(candidate, 'statement'), "Missing 'statement' attribute"
        assert hasattr(candidate, 'confidence'), "Missing 'confidence' attribute"
        assert hasattr(candidate, 'category'), "Missing 'category' attribute"
        assert isinstance(candidate.statement, str), f"statement should be str, got {type(candidate.statement)}"
        assert isinstance(candidate.confidence, (int, float)), f"confidence should be numeric, got {type(candidate.confidence)}"
        assert isinstance(candidate.category, str), f"category should be str, got {type(candidate.category)}"
    
    # Verify count matches expected lines (>10 chars)
    print(f"Extracted {len(result)} candidates:")
    for i, c in enumerate(result):
        print(f"  [{i}] statement='{c.statement}', confidence={c.confidence}, category='{c.category}'")
    
    # Verify WorkingMemory also has the candidates
    stored_candidates = wm.get_candidates()
    assert len(stored_candidates) == len(result), \
        f"WorkingMemory has {len(stored_candidates)} but extract returned {len(result)}"
    
    print("\nVERIFICATION PASSED: All extracted candidates have consistent structure")


if __name__ == "__main__":
    test_extract_candidates_structure()
