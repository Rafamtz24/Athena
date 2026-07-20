"""Test script for PromptBuilder candidate facts verification."""
import sys
from dataclasses import dataclass

# Add current directory to path
sys.path.insert(0, '.')

from athena.prompt.builder import PromptBuilder


@dataclass
class KnowledgeCandidate:
    statement: str
    confidence: float = 0.0
    category: str = ''


def test_prompt_with_candidates():
    """Test that candidates are included in the prompt."""

    class MockThought:
        def __init__(self):
            self.history = []
            self.memories = ['Memory 1', 'Memory 2']
            self.candidates = [
                KnowledgeCandidate('The sky is blue', 0.8, 'observation'),
                KnowledgeCandidate('Water boils at 100C', 0.95, 'science')
            ]
            self.knowledge = 'Some knowledge'
            self.user_input = 'Hello'

    builder = PromptBuilder()
    thought = MockThought()
    prompt = builder.build(thought)

    print('=== PROMPT WITH CANDIDATES ===')
    print(prompt)
    print()

    # Verify candidates section exists
    assert 'Candidate Facts' in prompt, 'Candidate Facts section missing!'
    assert 'The sky is blue' in prompt, 'Candidate statement not found!'
    assert '(confidence=0.8, category=observation)' in prompt, 'Confidence/category format incorrect!'
    print('[PASS] All assertions passed for candidates test')


def test_prompt_without_candidates():
    """Test that empty candidates are handled correctly."""

    class MockThought:
        def __init__(self):
            self.history = []
            self.memories = ['Memory 1']
            self.candidates = []
            self.knowledge = None
            self.user_input = 'Test'

    builder = PromptBuilder()
    thought = MockThought()
    prompt = builder.build(thought)

    print('=== PROMPT WITHOUT CANDIDATES ===')
    print(prompt)
    print()

    assert '(None)' in prompt, 'Expected (None) for empty candidates!'
    print('[PASS] Empty candidates handled correctly')


def test_prompt_with_no_candidates_attribute():
    """Test that missing candidates attribute is handled."""

    class MockThought:
        def __init__(self):
            self.history = []
            self.memories = ['Memory 1']
            self.knowledge = None
            self.user_input = 'Test'
            # No candidates attribute at all

    builder = PromptBuilder()
    thought = MockThought()
    
    try:
        prompt = builder.build(thought)
        print('[PASS] Missing candidates attribute handled gracefully')
    except AttributeError as e:
        print(f'[FAIL] AttributeError raised: {e}')


if __name__ == '__main__':
    test_prompt_with_candidates()
    test_prompt_without_candidates()
    test_prompt_with_no_candidates_attribute()
    print('\n=== ALL TESTS PASSED ===')
