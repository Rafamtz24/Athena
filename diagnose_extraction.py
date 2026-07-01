"""Diagnostic script to instrument the Learning Pipeline and capture real extraction data.

This script instruments the complete Learning Pipeline and prints:
- Completed Interaction
- Extraction Prompt
- Raw LLM Response
- Parsed Candidates
- Validator Decisions
- Semantic Memory Writes

Test conversations:
1. "My name is Rafael."
2. "My favorite color is blue."
3. "I live in Monterrey, Mexico."
"""
import json
import os
import sys
from pathlib import Path

os.chdir(os.path.dirname(__file__))

# Import Athena components
from athena.knowledge.manager import KnowledgeManager
from athena.knowledge.validator import KnowledgeValidator
from athena.memory.semantic import SemanticMemory
from athena.memory.manager import MemoryManager
from athena.providers.lmstudio import LMStudioProvider

# Initialize components
provider = LMStudioProvider()
memory_manager = MemoryManager()
knowledge_manager = KnowledgeManager(
    working_memory=memory_manager.working_memory,
    provider=provider
)

# Test conversations
test_conversations = [
    "User: My name is Rafael.\nAssistant: Hello Rafael! Nice to meet you.",
    "User: My favorite color is blue.\nAssistant: Got it, your favorite color is blue.",
    "User: I live in Monterrey, Mexico.\nAssistant: I'll remember that you live in Monterrey.",
]

# Track semantic memory before
initial_entries = memory_manager.semantic_memory.query()
initial_count = len(initial_entries)

print("=" * 60)
print("LEARNING PIPELINE DIAGNOSTIC")
print("=" * 60)
print(f"Initial Semantic Memory entries: {initial_count}")
print()

for conv_idx, conversation in enumerate(test_conversations, 1):
    print("=" * 60)
    print(f"CONVERSATION {conv_idx}")
    print("=" * 60)
    
    # ==========================
    # Completed Interaction
    # ==========================
    print("==========================")
    print("Completed Interaction")
    print("==========================")
    print(conversation)
    print()
    
    # ==========================
    # Extraction Prompt
    # ==========================
    print("==========================")
    print("Extraction Prompt")
    print("==========================")
    prompt = knowledge_manager._build_extraction_prompt(conversation)
    print(prompt)
    print()
    
    # ==========================
    # Raw LLM Response
    # ==========================
    print("==========================")
    print("Raw LLM Response")
    print("==========================")
    raw_response = provider.call(prompt)
    print(raw_response)
    print()
    
    # ==========================
    # Parsed Candidates
    # ==========================
    print("==========================")
    print("Parsed Candidates")
    print("==========================")
    
    text = str(raw_response).strip()
    parsed_candidates = []
    
    if text == "NONE":
        print("(NONE response - zero candidates)")
    else:
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if len(line) < 5:
                continue
            if any(marker in line for marker in ['```', '``', '**', '__', '==']):
                continue
            if all(c in '-*#>`_' for c in line[:3]):
                continue
            if any(line.startswith(prefix) for prefix in
                   ['User:', 'Assistant:', '- ', '* ', '# ', '> ', '`',
                    '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '0.',
                    'Fact:', 'Facts:', 'Example', 'Examples',
                    'Long-term', 'Short-term', 'Conversation', 'Summary',
                    'Valid format', 'Invalid format']):
                continue
            if any(line.lower().startswith(p) for p in
                   ['the assistant', 'this conversation', 'in this conversation',
                    'the user:', 'assistant:', 'note:', 'summary:',
                    'in this conversation', 'these facts', 'this response',
                    'here are', 'below are', 'the following']):
                continue
            if 'User:' in line or 'Assistant:' in line:
                continue
            
            from athena.knowledge.models import KnowledgeCandidate
            candidate = KnowledgeCandidate(
                statement=line,
                confidence=0.8,
                category="extracted"
            )
            parsed_candidates.append(candidate)
            print(f"  CANDIDATE: {candidate.statement}")
    
    if not parsed_candidates:
        print("  (no candidates)")
    print()
    
    # ==========================
    # Validator Decisions
    # ==========================
    print("==========================")
    print("Validator Decisions")
    print("==========================")
    
    validator = KnowledgeValidator(memory_manager.semantic_memory)
    
    if not parsed_candidates:
        print("  (no candidates to validate)")
    else:
        for candidate in parsed_candidates:
            classification, conflict_id = validator.classify(
                candidate.statement,
                candidate.confidence,
                candidate.category
            )
            print(f"  {candidate.statement} -> {classification}")
    print()
    
    # ==========================
    # Semantic Memory Writes
    # ==========================
    print("==========================")
    print("Semantic Memory Writes")
    print("==========================")
    
    if not parsed_candidates:
        print("  (no writes - no candidates)")
    else:
        for candidate in parsed_candidates:
            classification, conflict_id = validator.classify(
                candidate.statement,
                candidate.confidence,
                candidate.category
            )
            if classification == 'new_fact':
                entry_id = memory_manager.learn(candidate.statement, {
                    "type": "knowledge",
                    "confidence": candidate.confidence,
                    "category": candidate.category
                })
                print(f"  STORED: {candidate.statement} (id: {entry_id})")
            elif classification == 'duplicate':
                print(f"  REJECTED: {candidate.statement} (duplicate)")
            elif classification == 'possible_conflict':
                print(f"  REJECTED: {candidate.statement} (conflict - needs reconciliation)")
    
    print()
    print("-" * 60)
    print()

# Track semantic memory after
final_entries = memory_manager.semantic_memory.query()
final_count = len(final_entries)

print("=" * 60)
print("FINAL STATE")
print("=" * 60)
print(f"Initial Semantic Memory entries: {initial_count}")
print(f"Final Semantic Memory entries: {final_count}")
print(f"New entries created: {final_count - initial_count}")
print()
print("All Semantic Memory entries:")
for entry in final_entries:
    content = entry.content if hasattr(entry, 'content') else str(entry)
    print(f"  - {content[:100]}...")
