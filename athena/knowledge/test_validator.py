"""
Test to verify KnowledgeValidator correctly classifies candidates as:
- Duplicate (rejected)
- New Fact (promoted)
- Possible Conflict (stored for reconciliation, NOT promoted)

This tests the foundation of Capability 2: Memory Reconciliation.
"""
import sys
sys.path.insert(0, '.')

from athena.memory.semantic import SemanticMemory
from athena.knowledge.validator import KnowledgeValidator


def test_duplicate_detection():
    """Test that duplicate candidates are detected and rejected."""
    mem = SemanticMemory()
    # Add an existing fact
    from athena.memory.models import MemoryEntry
    entry = MemoryEntry(content="User prefers Python", metadata={})
    mem._knowledge.append(entry)

    validator = KnowledgeValidator(mem)
    
    # Exact duplicate
    classification, conflict_id = validator.classify("User prefers Python", 0.8, "preference")
    assert classification == 'duplicate', f"Expected 'duplicate' but got '{classification}'"
    
    # Near-duplicate (substring match)
    classification, conflict_id = validator.classify("User prefers Python for coding", 0.8, "preference")
    assert classification == 'duplicate', f"Expected 'duplicate' for substring but got '{classification}'"
    
    print("[PASS] Duplicate detection works correctly")


def test_new_fact_promotion():
    """Test that new facts are correctly identified."""
    mem = SemanticMemory()
    from athena.memory.models import MemoryEntry
    entry1 = MemoryEntry(content="User prefers Python", metadata={})
    entry2 = MemoryEntry(content="User lives in Mexico", metadata={})
    mem._knowledge.append(entry1)
    mem._knowledge.append(entry2)

    validator = KnowledgeValidator(mem)
    
    # New fact (not related to existing entries)
    classification, conflict_id = validator.classify("User has a cat", 0.9, "fact")
    assert classification == 'new_fact', f"Expected 'new_fact' but got '{classification}'"
    assert conflict_id is None
    
    print("[PASS] New fact detection works correctly")


def test_conflict_detection():
    """Test that conflicting facts are detected."""
    mem = SemanticMemory()
    from athena.memory.models import MemoryEntry
    entry1 = MemoryEntry(content="User has 2 children", metadata={})
    mem._knowledge.append(entry1)

    validator = KnowledgeValidator(mem)
    
    # Conflicting fact (negation pattern - "not" vs numeric contradiction)
    classification, conflict_id = validator.classify("User does not have any children", 0.85, "fact")
    assert classification == 'possible_conflict', f"Expected 'possible_conflict' but got '{classification}'"
    assert conflict_id is not None
    
    print("[PASS] Conflict detection works correctly")


def test_no_false_conflicts():
    """Test that unrelated facts are not flagged as conflicts."""
    mem = SemanticMemory()
    from athena.memory.models import MemoryEntry
    entry1 = MemoryEntry(content="User prefers Python", metadata={})
    mem._knowledge.append(entry1)

    validator = KnowledgeValidator(mem)
    
    # Unrelated fact (should be new_fact, not conflict)
    classification, conflict_id = validator.classify("User likes pizza", 0.9, "preference")
    assert classification == 'new_fact', f"Expected 'new_fact' for unrelated fact but got '{classification}'"
    
    print("[PASS] No false conflicts detected")


def test_conflict_stored_for_reconciliation():
    """Test that conflicts are stored in validator.conflicts for future reconciliation."""
    mem = SemanticMemory()
    from athena.memory.models import MemoryEntry
    entry1 = MemoryEntry(content="User has 5 years experience", metadata={})
    mem._knowledge.append(entry1)

    validator = KnowledgeValidator(mem)
    
    # Conflicting fact with different numeric value (same topic)
    classification, conflict_id = validator.classify("User has 10 years experience", 0.8, "fact")
    assert classification == 'possible_conflict', f"Expected 'possible_conflict' but got '{classification}'"
    
    # Verify conflict is stored
    conflicts = validator.get_conflicts()
    assert len(conflicts) == 1, f"Expected 1 conflict but got {len(conflicts)}"
    assert conflicts[0]['candidate_statement'] == "User has 10 years experience"
    assert 'existing_content' in conflicts[0]
    
    print("[PASS] Conflicts are stored for future reconciliation")


if __name__ == "__main__":
    test_duplicate_detection()
    test_new_fact_promotion()
    test_conflict_detection()
    test_no_false_conflicts()
    test_conflict_stored_for_reconciliation()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED: KnowledgeValidator foundation is correct")
    print("=" * 60)
