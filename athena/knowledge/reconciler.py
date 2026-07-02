"""
Athena Memory Reconciler

Reconciles validated knowledge candidates against existing Semantic Memory.

For each validated candidate, the reconciler:
1. Builds ONE prompt containing the candidate + ALL existing SM entries
   (via the dedicated prompt builder: build_reconciliation_prompt)
2. Calls the reasoning model ONCE per candidate
3. Parses the structured response (action + conflicting content)
4. Applies deterministic memory modifications

Provider call count: EXACTLY ONE per validated candidate.
The number of existing SM entries is irrelevant (all batched in one prompt).

Fail-safe: if reconciliation fails (LLM error, parse error), Semantic Memory
is NOT modified. The failure is logged and the candidate is discarded.
Preserving SM consistency is more important than learning one fact.

Edge cases:
- Empty Semantic Memory: short-circuited (no LLM call, always DIFFERENT)
- Provider failure: log error, do NOT modify SM
- Parse failure: log error, do NOT modify SM
"""

import logging
import re
from typing import List, Tuple

from athena.knowledge.models import KnowledgeCandidate
from athena.memory.semantic import SemanticMemory

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Dedicated prompt builder (isolated from reconciliation logic)
# ──────────────────────────────────────────────────────────────

def build_reconciliation_prompt(
    candidate_statement: str,
    existing_contents: List[str],
) -> str:
    """
    Build the reconciliation prompt for the reasoning model.

    Args:
        candidate_statement: The new candidate fact (plain text).
        existing_contents: List of existing SM entry contents (plain text).

    Returns:
        A prompt string ready to send to the LLM.

    This function is isolated so the prompt can evolve independently
    from the reconciliation algorithm.
    """
    lines = [
        "You compare pairs of factual statements about a user. ",
        "For each pair, decide if they are IDENTICAL, CONTRADICTORY, or UNRELATED.",
        "",
        "=== RULES ===",
        "",
        "Two facts are IDENTICAL (DUPLICATE) if:",
        "- They describe the exact same attribute of the same subject",
        "- AND they give the same value for that attribute",
        'Example: "User name is Rafael" and "User name is Rafael"',
        "",
        "Two facts are CONTRADICTORY (CONFLICT) if:",
        "- They describe the SAME attribute of the SAME subject",
        "- BUT they give DIFFERENT, incompatible values",
        'Example: "User name is Rafael" vs "User name is Pedro"',
        '  -> BOTH describe the "name" attribute of "User"',
        '  -> Rafael != Pedro, so they CONFLICT',
        'Example: "User favorite color is blue" vs "User favorite color is red"',
        '  -> BOTH describe "favorite color" of "User"',
        '  -> blue != red, so they CONFLICT',
        'Example: "User lives in New York" vs "User lives in Chicago"',
        '  -> BOTH describe "city" of "User"',
        '  -> New York != Chicago, so they CONFLICT',
        "",
        "Two facts are UNRELATED (DIFFERENT) if:",
        "- They describe DIFFERENT attributes (even of the same subject)",
        'Example: "User name is Rafael" vs "User has a dog named Gemma"',
        '  -> "name" and "dog" are different attributes -> UNRELATED',
        'Example: "User favorite color is blue" vs "User lives in Paris"',
        '  -> "favorite color" and "lives in" are different -> UNRELATED',
        "",
        "=== NOW CLASSIFY THIS PAIR ===",
        "",
        f'EXISTING: "{existing_contents[0] if existing_contents else "(none)"}"',
        f'NEW:      "{candidate_statement}"',
        "",
        "Based on the RULES above, which category fits?",
        "",
        "=== RESPONSE ===",
        "Reply with EXACTLY two lines. No extra text. No explanations.",
        "",
        "If IDENTICAL:",
        "ACTION: DUPLICATE",
        "MATCHES:",
        '- "exact text of the matching existing fact"',
        "",
        "If CONTRADICTORY:",
        "ACTION: CONFLICT",
        "CONFLICTS:",
        '- "exact text of the conflicting existing fact"',
        "",
        "If UNRELATED:",
        "ACTION: DIFFERENT",
    ]

    # Add remaining existing entries (if any) as additional context
    if existing_contents and len(existing_contents) > 1:
        lines.append("")
        lines.append("NOTE - Other existing entries (for context, not the main pair):")
        for i, content in enumerate(existing_contents[1:], 2):
            lines.append(f'  {i}. "{content}"')

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Response parser
# ──────────────────────────────────────────────────────────────

def parse_reconciliation_response(response: str) -> Tuple[str, List[str]]:
    """
    Parse the LLM response into (action, matched_contents).

    Expected response formats:

    ACTION: DUPLICATE
    MATCHES:
    - "exact fact text"

    ACTION: CONFLICT
    CONFLICTS:
    - "exact fact text 1"
    - "exact fact text 2"

    ACTION: DIFFERENT
    MATCHES: NONE

    Returns:
        Tuple of (action, list_of_content_strings).
        action is one of: 'DUPLICATE', 'CONFLICT', 'DIFFERENT'
        matched_contents is empty if none found.
    """
    text = str(response).strip()

    # Extract ACTION line
    action_match = re.search(
        r'ACTION\s*:\s*(DUPLICATE|CONFLICT|DIFFERENT)',
        text,
        re.IGNORECASE,
    )
    if not action_match:
        return ('DIFFERENT', [])

    action = action_match.group(1).upper()

    if action == 'DIFFERENT':
        return ('DIFFERENT', [])

    # Extract quoted content lines after MATCHES: or CONFLICTS:
    # Pattern: find MATCHES: or CONFLICTS: header, then collect "- "quoted"" lines
    header = 'MATCHES' if action == 'DUPLICATE' else 'CONFLICTS'
    # Match quoted strings like: - "some text" or "some text"
    # Find section after the header
    header_pattern = rf'{header}\s*:'
    header_match = re.search(header_pattern, text, re.IGNORECASE)
    if not header_match:
        return (action, [])

    # Get everything after the header line
    after_header = text[header_match.end():].strip()

    # Check if it's "NONE"
    if after_header.upper().startswith('NONE'):
        return (action, [])

    # Extract all quoted strings from lines starting with -
    contents = []
    # Pattern 1: - "content"
    for match in re.finditer(r'-\s*"([^"]*)"', after_header):
        contents.append(match.group(1))
    # Pattern 2: just "content" bullet (if no - prefix)
    if not contents:
        for match in re.finditer(r'"([^"]*)"', after_header):
            contents.append(match.group(1))

    return (action, contents)


# ──────────────────────────────────────────────────────────────
# Content matching helpers
# ──────────────────────────────────────────────────────────────

def find_entry_by_content(
    target_content: str,
    existing_entries: list,
) -> object | None:
    """
    Find a Semantic Memory entry whose content matches target_content.

    Matching strategy:
    1. Exact string match first
    2. Normalized match (via SemanticMemory.normalize) as fallback

    Returns the entry object, or None if no match found.
    """
    from athena.memory.semantic import SemanticMemory

    target_norm = SemanticMemory.normalize(target_content)

    for entry in existing_entries:
        entry_content = str(getattr(entry, 'content', entry))
        # Exact match
        if entry_content == target_content:
            return entry
        # Normalized match
        if SemanticMemory.normalize(entry_content) == target_norm:
            return entry

    return None


# ──────────────────────────────────────────────────────────────
# Memory Reconciler class
# ──────────────────────────────────────────────────────────────

class MemoryReconciler:
    """
    Reconciles knowledge candidates against existing Semantic Memory.

    Uses the reasoning model (via build_reconciliation_prompt) to determine
    if each candidate is a duplicate, conflict, or new knowledge relative to ALL
    existing Semantic Memory entries.

    Provider call count: EXACTLY ONE per validated candidate.
    """

    def __init__(self, llm_provider) -> None:
        self.llm_provider = llm_provider

    # ── Public API ────────────────────────────────────────────

    def reconcile(
        self,
        candidates: List[KnowledgeCandidate],
        semantic_memory: SemanticMemory,
    ) -> dict:
        """
        Reconcile all candidates against existing Semantic Memory.

        Args:
            candidates: List of validated KnowledgeCandidate objects.
            semantic_memory: Reference to SemanticMemory for modifications.

        Returns:
            Dict with reconciliation counts:
            {
                'processed': int,
                'duplicates': int,
                'conflicts': int,
                'new_facts': int,
                'errors': int,
            }
        """
        results = {
            'processed': 0,
            'duplicates': 0,
            'conflicts': 0,
            'new_facts': 0,
            'errors': 0,
        }

        for candidate in candidates:
            outcome = self._reconcile_one(candidate, semantic_memory)
            results['processed'] += 1
            if outcome in results:
                results[outcome] += 1

        return results

    # ── Internal: reconcile a single candidate ─────────────────

    def _reconcile_one(
        self,
        candidate: KnowledgeCandidate,
        semantic_memory: SemanticMemory,
    ) -> str:
        """
        Reconcile a SINGLE candidate.

        Returns one of: 'duplicates', 'conflicts', 'new_facts', 'errors'
        """
        existing_entries = semantic_memory.query()

        # ── Edge case: empty Semantic Memory ──
        if not existing_entries:
            semantic_memory.learn(candidate.statement, {
                "type": "knowledge",
                "confidence": candidate.confidence,
                "category": candidate.category,
            })
            return 'new_facts'

        # ── Build prompt with ALL existing entries ──
        existing_contents = [
            str(getattr(e, 'content', e)) for e in existing_entries
        ]
        prompt = build_reconciliation_prompt(candidate.statement, existing_contents)

        # ── EXACTLY ONE provider call per candidate ──
        try:
            response = self.llm_provider.generate(prompt)
        except Exception as exc:
            # Provider failure — do NOT modify SM
            logger.error(
                f"[RECONCILER] Provider call failed for candidate "
                f'"{candidate.statement[:60]}": {exc}'
            )
            return 'errors'

        # ── Parse structured response ──
        try:
            action, matched_contents = parse_reconciliation_response(response)
        except Exception as exc:
            # Parse failure — do NOT modify SM
            logger.error(
                f"[RECONCILER] Parse failed for candidate "
                f'"{candidate.statement[:60]}": {exc}'
            )
            logger.error(f"[RECONCILER] Raw response: {response[:300]}")
            return 'errors'

        # ── Apply deterministic actions ──

        if action == 'DUPLICATE':
            return 'duplicates'

        if action == 'CONFLICT':
            # Find and remove each conflicting entry by content
            for conflict_content in matched_contents:
                entry = find_entry_by_content(conflict_content, existing_entries)
                if entry is not None:
                    entry_id = getattr(entry, 'id', None)
                    if entry_id:
                        semantic_memory.remove(entry_id)

            # Insert the new candidate (newer fact wins)
            semantic_memory.learn(candidate.statement, {
                "type": "knowledge",
                "confidence": candidate.confidence,
                "category": candidate.category,
            })
            return 'conflicts'

        # DIFFERENT: insert as new knowledge
        if action == 'DIFFERENT':
            semantic_memory.learn(candidate.statement, {
                "type": "knowledge",
                "confidence": candidate.confidence,
                "category": candidate.category,
            })
            return 'new_facts'

        # Unknown action — do NOT modify SM
        logger.error(
            f"[RECONCILER] Unknown action '{action}' for candidate "
            f'"{candidate.statement[:60]}"'
        )
        return 'errors'
