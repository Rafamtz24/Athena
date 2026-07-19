"""
Verify Semantic Memory quality improvements.

Tests the enhanced KnowledgeValidator to ensure:
  - VALID durable user facts are accepted (classified as 'new_fact')
  - INVALID candidates (imperative, conversational, testing, placeholder,
    incomplete, echoed) are rejected (classified as 'low_quality')

This is the primary quality gate for the Learning Pipeline.
"""

import sys
sys.path.insert(0, '.')

from athena.memory.semantic import SemanticMemory
from athena.knowledge.validator import KnowledgeValidator


# ─────────────────────────────────────────────────────────
# PASS CRITERIA
# ─────────────────────────────────────────────────────────
# VALID facts must be classified as 'new_fact'
# INVALID candidates must be classified as 'low_quality'

VALID_FACTS = [
    # Name
    "User's name is Alex",
    "User's last name is Turner",
    "User's full name is John Michael Smith",

    # Location
    "User lives in Lisbon",
    "User lives in Mexico City",
    "User lives in the United States",
    "User is from Canada",
    "User lives in New York City",

    # Pet
    "User has a dog named Rex",
    "User has a cat named Whiskers",
    "User owns a parrot",

    # Preferences
    "User's favorite color is blue",
    "User's favorite food is pizza",
    "User likes programming in Python",
    "User prefers dark mode",
    "User enjoys reading science fiction",

    # Occupation / Studies
    "User is studying psychology",
    "User works as a software engineer",
    "User is a doctor",
    "User works at Google",
    "User studies computer science at MIT",

    # Other durable facts
    "User has three siblings",
    "User graduated from Stanford in 2020",
    "User speaks Spanish and English",
    "User was born on March 15",
    "User exercises three times per week",
]

INVALID_IMPERATIVE = [
    # "Respond with..." patterns
    "Respond with Hello",
    "Respond with the word OK",
    "Respond with exactly the word OK and nothing else",

    # "Say..." patterns
    "Say Hello",
    "Say World",
    "Say exactly Hello",
    "Say nothing else",

    # "Repeat..." patterns
    "Repeat this",
    "Repeat after me",
    "Repeat the following words",

    # "Write..." patterns
    "Write a poem",
    "Write a story about dragons",
    "Write code for a calculator",

    # "Translate..." patterns
    "Translate Hello to Spanish",
    "Translate this sentence to French",

    # "Count..." patterns
    "Count to ten",
    "Count the number of words",

    # "Tell me..." patterns
    "Tell me a joke",
    "Tell me a story",
    "Tell me about yourself",

    # "Print..." patterns
    "Print Hello World",
    "Print the result",

    # "Open..." patterns
    "Open the door",
    "Open the file README.md",

    # Other command patterns
    "Create a todo list",
    "Generate a response",
    "Calculate 2 plus 2",
    "Search for restaurants near me",
    "List all files in the directory",
    "Show me the weather forecast",
    "Give me a summary",
]

INVALID_CONVERSATIONAL = [
    # "User says..." patterns
    "User says Hello",
    "User says World",
    "User says OK",
    "User says testing",
    "User says: Hello",

    # "User responds..." patterns
    "User responds OK",
    "User responds with exactly the word OK and nothing else",
    "User responds with Hello",
    "User responds: OK",

    # "User asks..." patterns
    "User asks about the weather",
    "User asks for help",
    "User asks: what is the capital of France",
    "User asks about the project status",

    # "User requested..." patterns
    "User requested information about Python",
    "User requested a summary",
    "User requested help with homework",

    # "User greeted..." patterns
    "User greeted the assistant",
    "User greeted the system",

    # "User told..." patterns
    "User told the assistant to write a poem",
    "User told me to repeat the instructions",
    "User said: Hello",

    # "User writes..." patterns
    "User writes Hello",
    "User writes: I need help",

    # Other conversational patterns
    "User mentioned the weather",
    "User answered the question",
    "User typed Hello World",
    "User entered the command",
    "User gave the following input",
    "User provided the information",
]

INVALID_TESTING = [
    # Direct test phrases
    "testing",
    "just testing",
    "repeat after me",
    "only answer",
    "answer with",
    "nothing else",
    "say exactly",
    "respond exactly",

    # Variations
    "respond with",
    "output only",
    "output exactly",
    "test message",
]

INVALID_PLACEHOLDER = [
    # Last-word placeholders
    "User lives in none specified",
    "User lives in unknown",
    "User's name is unknown",
    "User's name is unspecified",
    "User's name is none",
    "User's name is N/A",
    "User's name is null",
    "User's favorite food is none",
    "User lives in unspecified",
    "User works at unknown",

    # Multi-word placeholder phrases
    "User's name is none specified",
    "User's address is not provided",
    "User's email is not provided",
    "User's phone number is no value",
    "User's occupation is value unknown",

    # Pronoun values
    "User's name is it",
    "User lives in her",
    "User works with him",
    "User studies with them",

    # Vague values
    "User lives in a place",
    "User is a person",
    "User lives in someone",
    "User lives in somewhere",
    "User is something",
    "User knows someone",
]

INVALID_INCOMPLETE = [
    # Trailing linking verbs
    "User lives in",
    "User lives at",
    "User's favorite food is",
    "User's name is",
    "User is from",
    "User works at",
    "User works in",
    "User works as",
    "User studies at",
    "User likes",
    "User prefers",
    "User has a dog named",
    "User has a cat called",
    "User graduated from",
    "User is known as",

    # Trailing incomplete verbs
    "User studies",
    "User works",
    "User likes",
    "User lives",
    "User prefers",
    "User has",
    "User enjoys",
]

INVALID_ECHO = [
    # Single-word echoes
    "Hello",
    "World",
    "OK",
    "Testing",
    "Test",
    "Hi",
    "Hey",
    "Bye",
    "Yes",
    "No",
    "Thanks",
]

# Additional edge cases from the task description
INVALID_SPECIFIC = [
    "User responds with exactly the word OK and nothing else",
    "User says: Hello",
    "User says: World",
    "Say Hello",
    "Say World",
    "Respond with OK",
    "Repeat this",
    "Write a poem",
    "Tell me a joke",
    "User says Hello",
    "User responds OK",
    "User requested X",
    "User lives in none specified",
    "User lives in unknown",
    "User's name is unknown",
    "User's favorite food is",
    "User has a dog named",
]

# Facts that would be valid coming from the VALID_EXAMPLES
# as described in the task description
VALID_AUTO_BIOGRAPHICAL = [
    "User's name is Alex",
    "User lives in Lisbon",
    "User has a dog named Rex",
    "User's favorite color is blue",
    "User is studying psychology",
]


def test_valid_facts():
    """Test that valid durable user facts pass the quality gate."""
    mem = SemanticMemory()
    validator = KnowledgeValidator(mem)
    failures = []

    for fact in VALID_FACTS:
        classification, _ = validator.classify(fact, 0.8, "fact")
        if classification == 'low_quality':
            failures.append(f"  VALID fact rejected as low_quality: '{fact}'")

    if failures:
        print(f"[FAIL] {len(failures)} valid facts were incorrectly rejected:")
        for f in failures:
            print(f)
    else:
        print(f"[PASS] All {len(VALID_FACTS)} valid facts correctly accepted")

    return len(failures)


def test_invalid_imperative():
    """Test that imperative commands are rejected."""
    mem = SemanticMemory()
    validator = KnowledgeValidator(mem)
    failures = []

    for fact in INVALID_IMPERATIVE:
        classification, _ = validator.classify(fact, 0.8, "fact")
        if classification != 'low_quality':
            failures.append(f"  Imperative NOT rejected: '{fact}'")

    if failures:
        print(f"[FAIL] {len(failures)} imperative instructions passed the gate:")
        for f in failures:
            print(f)
    else:
        print(f"[PASS] All {len(INVALID_IMPERATIVE)} imperative instructions correctly rejected")

    return len(failures)


def test_invalid_conversational():
    """Test that conversational descriptions are rejected."""
    mem = SemanticMemory()
    validator = KnowledgeValidator(mem)
    failures = []

    for fact in INVALID_CONVERSATIONAL:
        classification, _ = validator.classify(fact, 0.8, "fact")
        if classification != 'low_quality':
            failures.append(f"  Conversational NOT rejected: '{fact}'")

    if failures:
        print(f"[FAIL] {len(failures)} conversational descriptions passed the gate:")
        for f in failures:
            print(f)
    else:
        print(f"[PASS] All {len(INVALID_CONVERSATIONAL)} conversational descriptions correctly rejected")

    return len(failures)


def test_invalid_testing():
    """Test that testing interactions are rejected."""
    mem = SemanticMemory()
    validator = KnowledgeValidator(mem)
    failures = []

    for fact in INVALID_TESTING:
        classification, _ = validator.classify(fact, 0.8, "fact")
        if classification != 'low_quality':
            failures.append(f"  Testing NOT rejected: '{fact}'")

    if failures:
        print(f"[FAIL] {len(failures)} testing interactions passed the gate:")
        for f in failures:
            print(f)
    else:
        print(f"[PASS] All {len(INVALID_TESTING)} testing interactions correctly rejected")

    return len(failures)


def test_invalid_placeholder():
    """Test that placeholder values are rejected."""
    mem = SemanticMemory()
    validator = KnowledgeValidator(mem)
    failures = []

    for fact in INVALID_PLACEHOLDER:
        classification, _ = validator.classify(fact, 0.8, "fact")
        if classification != 'low_quality':
            failures.append(f"  Placeholder NOT rejected: '{fact}'")

    if failures:
        print(f"[FAIL] {len(failures)} placeholder values passed the gate:")
        for f in failures:
            print(f)
    else:
        print(f"[PASS] All {len(INVALID_PLACEHOLDER)} placeholder values correctly rejected")

    return len(failures)


def test_invalid_incomplete():
    """Test that incomplete facts are rejected."""
    mem = SemanticMemory()
    validator = KnowledgeValidator(mem)
    failures = []

    for fact in INVALID_INCOMPLETE:
        classification, _ = validator.classify(fact, 0.8, "fact")
        if classification != 'low_quality':
            failures.append(f"  Incomplete NOT rejected: '{fact}'")

    if failures:
        print(f"[FAIL] {len(failures)} incomplete facts passed the gate:")
        for f in failures:
            print(f)
    else:
        print(f"[PASS] All {len(INVALID_INCOMPLETE)} incomplete facts correctly rejected")

    return len(failures)


def test_invalid_echo():
    """Test that echoed user text is rejected."""
    mem = SemanticMemory()
    validator = KnowledgeValidator(mem)
    failures = []

    for fact in INVALID_ECHO:
        classification, _ = validator.classify(fact, 0.8, "fact")
        if classification != 'low_quality':
            failures.append(f"  Echo NOT rejected: '{fact}'")

    if failures:
        print(f"[FAIL] {len(failures)} echoed tokens passed the gate:")
        for f in failures:
            print(f)
    else:
        print(f"[PASS] All {len(INVALID_ECHO)} echoed tokens correctly rejected")

    return len(failures)


def test_specific_task_examples():
    """Test the exact examples from the task specification."""
    mem = SemanticMemory()
    validator = KnowledgeValidator(mem)
    failures = []

    # VALID examples
    for fact in VALID_AUTO_BIOGRAPHICAL:
        classification, _ = validator.classify(fact, 0.8, "fact")
        if classification == 'low_quality':
            failures.append(f"  TASK VALID fact rejected: '{fact}'")

    # INVALID examples
    for fact in INVALID_SPECIFIC:
        classification, _ = validator.classify(fact, 0.8, "fact")
        if classification != 'low_quality':
            failures.append(f"  TASK INVALID fact NOT rejected: '{fact}'")

    if failures:
        print(f"[FAIL] {len(failures)} task-specific examples failed:")
        for f in failures:
            print(f)
    else:
        total = len(VALID_AUTO_BIOGRAPHICAL) + len(INVALID_SPECIFIC)
        print(f"[PASS] All {total} task-specific examples classified correctly")

    return len(failures)


def test_existing_tests_still_pass():
    """Verify that the existing test_validator.py assertions still hold."""
    from athena.memory.models import MemoryEntry

    mem = SemanticMemory()

    # Pre-populate with existing facts for duplicate detection
    entry = MemoryEntry(content="User prefers Python", metadata={})
    mem._knowledge.append(entry)

    validator = KnowledgeValidator(mem)
    failures = []

    # Duplicate detection
    classification, _ = validator.classify("User prefers Python", 0.8, "preference")
    if classification != 'duplicate':
        failures.append(f"Duplicate detection failed: got '{classification}' expected 'duplicate'")

    # New fact
    classification, _ = validator.classify("User has a cat", 0.9, "fact")
    if classification != 'new_fact':
        failures.append(f"New fact detection failed: got '{classification}' expected 'new_fact'")

    # Conflict detection
    mem2 = SemanticMemory()
    entry2 = MemoryEntry(content="User has 2 children", metadata={})
    mem2._knowledge.append(entry2)
    validator2 = KnowledgeValidator(mem2)
    classification, conflict_id = validator2.classify("User does not have any children", 0.85, "fact")
    if classification != 'possible_conflict':
        failures.append(f"Conflict detection failed: got '{classification}' expected 'possible_conflict'")

    # No false conflicts
    mem3 = SemanticMemory()
    entry3 = MemoryEntry(content="User prefers Python", metadata={})
    mem3._knowledge.append(entry3)
    validator3 = KnowledgeValidator(mem3)
    classification, _ = validator3.classify("User likes pizza", 0.9, "preference")
    if classification != 'new_fact':
        failures.append(f"No false conflicts failed: got '{classification}' expected 'new_fact'")

    if failures:
        print(f"[FAIL] {len(failures)} existing tests broke:")
        for f in failures:
            print(f"  {f}")
    else:
        print("[PASS] Existing test_validator.py assertions still pass")

    return len(failures)


def run_all_tests():
    """Run all quality verification tests."""
    print("=" * 70)
    print("SEMANTIC MEMORY QUALITY VERIFICATION")
    print("=" * 70)
    print()

    total_failures = 0

    print("[1/9] Valid durable user facts...")
    total_failures += test_valid_facts()

    print()
    print("[2/9] Reject imperative instructions...")
    total_failures += test_invalid_imperative()

    print()
    print("[3/9] Reject conversational behavior...")
    total_failures += test_invalid_conversational()

    print()
    print("[4/9] Reject testing interactions...")
    total_failures += test_invalid_testing()

    print()
    print("[5/9] Reject placeholder values...")
    total_failures += test_invalid_placeholder()

    print()
    print("[6/9] Reject incomplete facts...")
    total_failures += test_invalid_incomplete()

    print()
    print("[7/9] Reject echoed user text...")
    total_failures += test_invalid_echo()

    print()
    print("[8/9] Task specification examples...")
    total_failures += test_specific_task_examples()

    print()
    print("[9/9] Existing test compatibility...")
    total_failures += test_existing_tests_still_pass()

    print()
    print("=" * 70)
    if total_failures == 0:
        print("ALL TESTS PASSED: Semantic Memory quality gates are effective")
    else:
        print(f"TESTS FAILED: {total_failures} assertions failed")
    print("=" * 70)

    return total_failures


if __name__ == "__main__":
    sys.exit(run_all_tests())
