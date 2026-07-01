"""
Athena Knowledge Reconciler

Resolves possible conflicts detected by KnowledgeValidator using the LLM.

When a knowledge candidate is classified as 'possible_conflict', this reconciler:
1. Presents the existing semantic memory entry and the new candidate to the LLM
2. Asks the model to choose exactly one outcome: REPLACE, KEEP, or REJECT
3. Applies the decision to Semantic Memory

This is the second stage of Capability 2: Memory Reconciliation.
"""

from typing import Optional

from athena.memory.semantic import SemanticMemory


class MemoryReconciler:
    """
    Resolves knowledge conflicts using LLM-based reconciliation.

    Workflow for each possible_conflict:
        - Call reconcile(conflicts) to process all pending conflicts
        - Each conflict is resolved independently via the LLM
        - The decision (REPLACE/KEEP/REJECT) is applied to Semantic Memory

    Only Possible Conflicts are reconciled.
    Duplicates and New Facts continue using deterministic logic in KnowledgeValidator.
    """

    def __init__(self, llm_provider):
        self.llm_provider = llm_provider

    def reconcile(self, conflicts: list[dict], semantic_memory: SemanticMemory) -> dict:
        """
        Reconcile all pending conflicts using the LLM.

        Args:
            conflicts: List of conflict records from KnowledgeValidator.conflicts
            semantic_memory: Reference to SemanticMemory for applying decisions

        Returns:
            Dict with reconciliation results: {
                'processed': int,
                'replaced': int,
                'kept': int,
                'rejected': int
            }
        """
        results = {'processed': 0, 'replaced': 0, 'kept': 0, 'rejected': 0}

        for conflict in conflicts:
            decision = self._resolve(conflict, semantic_memory)
            results['processed'] += 1
            if decision == 'REPLACE':
                results['replaced'] += 1
            elif decision == 'KEEP':
                results['kept'] += 1
            else:
                results['rejected'] += 1

        return results

    def _resolve(self, conflict: dict, semantic_memory: SemanticMemory) -> str:
        """
        Resolve a single conflict using the LLM.

        The LLM is asked to choose exactly one outcome:
            REPLACE - Replace existing entry with new candidate
            KEEP    - Keep existing entry, discard new candidate
            REJECT  - Reject new candidate (same as KEEP)

        If the provider fails, returns 'REJECT' as a safe default
        (keeps existing memory, discards the new candidate).

        Returns:
            'REPLACE', 'KEEP', or 'REJECT'
        """
        existing_content = conflict['existing_content']
        candidate_statement = conflict['candidate_statement']

        prompt = (
            "You are resolving a knowledge conflict. Choose exactly ONE outcome.\n\n"
            f"Existing memory: \"{existing_content}\"\n"
            f"New candidate:   \"{candidate_statement}\"\n\n"
            "Choose one:\n"
            "- REPLACE if the new candidate should replace the existing entry (e.g., it's more accurate, newer, or corrects an error)\n"
            "- KEEP if the existing entry is better (e.g., the candidate contradicts but existing is more reliable)\n"
            "- REJECT if the candidate is clearly wrong and should be discarded\n\n"
            "Respond with exactly one word: REPLACE, KEEP, or REJECT."
        )

        try:
            response = self.llm_provider.generate(prompt)
        except Exception:
            # Provider failed — safe default: reject new candidate, keep existing memory
            return 'REJECT'

        decision = response.strip().upper()

        # Normalize response to expected outcome
        if 'REPLACE' in decision:
            result = 'REPLACE'
        elif 'KEEP' in decision:
            result = 'KEEP'
        else:
            result = 'REJECT'

        # Apply the decision for REPLACE
        if result == 'REPLACE':
            existing_id = conflict.get('existing_id')
            if existing_id and semantic_memory.update(existing_id, candidate_statement):
                pass  # Successfully updated
            else:
                # Fallback: find by content match
                entries = semantic_memory.query()
                for entry in entries:
                    if entry.content == existing_content:
                        entry.content = candidate_statement
                        break

        return result
