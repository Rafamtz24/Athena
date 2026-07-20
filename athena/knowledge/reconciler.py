"""
Athena Memory Reconciler

Reconciles validated knowledge candidates against existing Semantic Memory.

Reconciliation runs in two passes:

1. Deterministic. Candidates naming a single-valued attribute (name,
   location, operating system...) are resolved by comparing parsed values —
   no model, and therefore reliable regardless of which model is loaded.
2. Batched. Everything left goes to the model in ONE call, together with all
   existing SM entries. Each verdict is parsed and applied.

Provider call count: ONE per turn, not one per candidate — plus one retry for
any candidate the batched response failed to classify.

Edge case, single candidate: sent through the single-pair prompt instead. Its
output shape is simpler for a small learning model, and there is nothing to
batch.

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

from athena.knowledge.attributes import parse_fact
from athena.knowledge.models import KnowledgeCandidate
from athena.memory.semantic import SemanticMemory
from athena.prompt.loader import PromptLoader

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Dedicated prompt builder (isolated from reconciliation logic)
# ──────────────────────────────────────────────────────────────

# PERFORMANCE: The reconciliation profile (rules, examples, response
# format) is loaded from athena/prompts/reconciliation.json once at
# import time, and the static prefix/suffix strings are precomputed
# here so build_reconciliation_prompt() only has to join the dynamic
# candidate/existing-entry pair on every call.
_RECONCILIATION_PROFILE = PromptLoader.load("reconciliation")

_RECONCILIATION_PROMPT_PREFIX = (
    _RECONCILIATION_PROFILE.system_prompt
    + "\n\n"
    + _RECONCILIATION_PROFILE.rules
    + "\n\n=== NOW CLASSIFY THIS PAIR ===\n\n"
)

_RECONCILIATION_PROMPT_SUFFIX = (
    "\n\n" + _RECONCILIATION_PROFILE.response_format
)

_BATCH_PROMPT_PREFIX = (
    _RECONCILIATION_PROFILE.system_prompt
    + "\n\n"
    + _RECONCILIATION_PROFILE.rules
    + "\n\n=== NOW CLASSIFY EVERY NEW FACT BELOW ===\n\n"
)

_BATCH_PROMPT_SUFFIX = (
    "\n\n" + _RECONCILIATION_PROFILE.get(
        "batch_response_format", _RECONCILIATION_PROFILE.response_format
    )
)


def build_batch_reconciliation_prompt(
    candidate_statements: List[str],
    existing_contents: List[str],
) -> str:
    """Build one prompt classifying several candidates against memory.

    A turn usually yields more than one candidate, and reconciling them one
    call at a time made the learning phase scale with candidate count for no
    benefit — every call re-sends the same memory. Batching also lets the
    model see the candidates together, so two facts that duplicate *each
    other* are visible rather than being judged in isolation.

    Args:
        candidate_statements: The new facts, in order. Their 1-based position
            is the FACT number the model must echo back.
        existing_contents: Every existing Semantic Memory entry.

    Returns:
        A prompt string ready to send to the LLM.
    """
    lines = ["EXISTING MEMORY:"]
    if existing_contents:
        for index, content in enumerate(existing_contents, 1):
            lines.append(f'  {index}. "{content}"')
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("NEW FACTS TO CLASSIFY:")
    for index, statement in enumerate(candidate_statements, 1):
        lines.append(f'  FACT {index}: "{statement}"')

    return _BATCH_PROMPT_PREFIX + "\n".join(lines) + _BATCH_PROMPT_SUFFIX


def parse_batch_reconciliation_response(
    response: str,
    expected: int,
) -> dict:
    """Split a batched response into per-candidate verdicts.

    Returns:
        {candidate_index (0-based): (action, matched_contents)} for every
        block that parsed. Indices missing from the result are the caller's
        signal to reconcile those candidates individually — a model that
        drops or garbles one block should cost one extra call, not the whole
        batch.
    """
    text = str(response)
    verdicts: dict = {}

    # Locate each "FACT: n" header, then hand the text up to the next header
    # to the same parser the single-pair path uses.
    headers = list(re.finditer(r'FACT\s*:?\s*(\d+)', text, re.IGNORECASE))
    for position, header in enumerate(headers):
        try:
            number = int(header.group(1))
        except ValueError:
            continue
        if not 1 <= number <= expected:
            continue

        end = headers[position + 1].start() if position + 1 < len(headers) else len(text)
        block = text[header.end():end]

        action, contents = parse_reconciliation_response(block)
        # A block with no ACTION line parses as DIFFERENT, which would insert
        # the fact on no evidence. Treat it as missing so it is retried alone.
        if not re.search(r'ACTION\s*:', block, re.IGNORECASE):
            continue
        verdicts[number - 1] = (action, contents)

    return verdicts


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

    PERFORMANCE: Static rules/examples/response-format are precomputed
    into _RECONCILIATION_PROMPT_PREFIX / _RECONCILIATION_PROMPT_SUFFIX.
    Only the dynamic pair and additional entries are constructed per call.
    """
    # Build the dynamic pair section
    first_existing = existing_contents[0] if existing_contents else "(none)"
    pair_lines = [
        f'EXISTING: "{first_existing}"',
        f'NEW:      "{candidate_statement}"',
    ]

    # Add remaining existing entries (if any) as additional context
    if existing_contents and len(existing_contents) > 1:
        pair_lines.append("")
        pair_lines.append("NOTE - Other existing entries (for context, not the main pair):")
        for i, content in enumerate(existing_contents[1:], 2):
            pair_lines.append(f'  {i}. "{content}"')

    return (
        _RECONCILIATION_PROMPT_PREFIX
        + "\n".join(pair_lines)
        + _RECONCILIATION_PROMPT_SUFFIX
    )


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

    Determines whether each candidate is a duplicate, a conflict, or new
    knowledge, relative to ALL existing Semantic Memory entries.

    Provider call count: ONE per turn (see module docstring), after the
    deterministic pass has removed everything a model is not needed for.
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

        # Pass 1 — deterministic. Single-valued attributes (name, location,
        # ...) resolve without a model at all, so they never reach the batch.
        deferred: List[KnowledgeCandidate] = []
        for candidate in candidates:
            existing_entries = semantic_memory.query()
            if not existing_entries:
                self._learn_candidate(candidate, semantic_memory)
                results['processed'] += 1
                results['new_facts'] += 1
                continue

            outcome = self._reconcile_by_attribute(
                candidate, semantic_memory, existing_entries
            )
            if outcome is None:
                deferred.append(candidate)
            else:
                results['processed'] += 1
                results[outcome] += 1

        # Pass 2 — one model call for everything left.
        for outcome in self._reconcile_deferred(deferred, semantic_memory):
            results['processed'] += 1
            if outcome in results:
                results[outcome] += 1

        return results

    def _reconcile_deferred(
        self,
        candidates: List[KnowledgeCandidate],
        semantic_memory: SemanticMemory,
    ) -> List[str]:
        """Reconcile the candidates the deterministic pass could not judge.

        One provider call for the whole group. A single candidate is sent
        through the single-pair prompt instead — it is the simpler output for
        a small learning model to produce, and there is nothing to batch.

        Returns:
            One outcome string per candidate, in order.
        """
        if not candidates:
            return []
        if len(candidates) == 1:
            return [self._reconcile_one(candidates[0], semantic_memory)]

        existing_entries = semantic_memory.query()
        existing_contents = [str(getattr(e, 'content', e)) for e in existing_entries]
        prompt = build_batch_reconciliation_prompt(
            [c.statement for c in candidates], existing_contents
        )

        try:
            response = self.llm_provider.generate(prompt)
        except Exception as exc:
            logger.error(f"[RECONCILER] Batch provider call failed: {exc}")
            return ['errors'] * len(candidates)

        try:
            verdicts = parse_batch_reconciliation_response(response, len(candidates))
        except Exception as exc:
            logger.error(f"[RECONCILER] Batch parse failed: {exc}")
            logger.error(f"[RECONCILER] Raw response: {str(response)[:300]}")
            verdicts = {}

        outcomes: List[str] = []
        for index, candidate in enumerate(candidates):
            if index not in verdicts:
                # The model dropped or garbled this block. Retrying it alone
                # costs one call; guessing would cost a wrong memory.
                logger.warning(
                    f"[RECONCILER] No verdict for candidate {index + 1}; "
                    f"reconciling it individually."
                )
                outcomes.append(self._reconcile_one(candidate, semantic_memory))
                continue

            action, matched_contents = verdicts[index]
            outcomes.append(
                self._apply_verdict(
                    candidate, action, matched_contents, semantic_memory
                )
            )

        return outcomes

    def _apply_verdict(
        self,
        candidate: KnowledgeCandidate,
        action: str,
        matched_contents: List[str],
        semantic_memory: SemanticMemory,
    ) -> str:
        """Apply one classified verdict to Semantic Memory.

        Memory is re-read here rather than reusing the snapshot the batch was
        classified against: earlier candidates in the same batch may already
        have inserted or removed entries, and a conflict must be resolved
        against what is actually stored now.
        """
        if action == 'DUPLICATE':
            return 'duplicates'

        if action == 'CONFLICT':
            existing_entries = semantic_memory.query()
            for conflict_content in matched_contents:
                entry = find_entry_by_content(conflict_content, existing_entries)
                if entry is not None:
                    entry_id = getattr(entry, 'id', None)
                    if entry_id:
                        semantic_memory.remove(entry_id)
            self._learn_candidate(candidate, semantic_memory)
            return 'conflicts'

        if action == 'DIFFERENT':
            # Two candidates in one batch can duplicate each other; each was
            # judged against memory as it stood before either was stored.
            if self._already_stored(candidate.statement, semantic_memory):
                return 'duplicates'
            self._learn_candidate(candidate, semantic_memory)
            return 'new_facts'

        logger.error(
            f"[RECONCILER] Unknown action '{action}' for candidate "
            f'"{candidate.statement[:60]}"'
        )
        return 'errors'

    @staticmethod
    def _already_stored(statement: str, semantic_memory: SemanticMemory) -> bool:
        """Whether this exact statement (ignoring case/spacing) is in memory."""
        target = ' '.join(statement.lower().split()).rstrip('.')
        for entry in semantic_memory.query():
            content = str(getattr(entry, 'content', entry))
            if ' '.join(content.lower().split()).rstrip('.') == target:
                return True
        return False

    # ── Internal: reconcile a single candidate ─────────────────

    def _learn_candidate(
        self,
        candidate: KnowledgeCandidate,
        semantic_memory: SemanticMemory,
    ) -> None:
        """Persist a candidate as a new Semantic Memory entry."""
        semantic_memory.learn(candidate.statement, {
            "type": "knowledge",
            "confidence": candidate.confidence,
            "category": candidate.category,
        })

    def _reconcile_by_attribute(
        self,
        candidate: KnowledgeCandidate,
        semantic_memory: SemanticMemory,
        existing_entries: list,
    ) -> str | None:
        """Deterministic reconciliation for recognized single-valued attributes.

        Model-independent fast path: if the candidate parses into a known
        single-valued attribute (name, location, ...), compare it against
        existing entries for the SAME (subject, attribute):
            - same value        -> 'duplicates' (no change)
            - different value(s) -> 'conflicts'  (remove old, insert new)
            - no existing value  -> 'new_facts'  (insert)

        Returns the outcome string, or None if the candidate is not a
        recognized single-valued attribute (defer to the LLM reconciler).
        """
        new_fact = parse_fact(candidate.statement)
        if new_fact is None:
            return None

        # Collect existing entries describing the same attribute of the same subject.
        same_attribute: list = []
        for entry in existing_entries:
            existing_fact = parse_fact(str(getattr(entry, 'content', entry)))
            if existing_fact is not None and existing_fact.key == new_fact.key:
                same_attribute.append((entry, existing_fact))

        if not same_attribute:
            self._learn_candidate(candidate, semantic_memory)
            return 'new_facts'

        # Already stored with the same value → duplicate, no change.
        if any(f.value_norm == new_fact.value_norm for _, f in same_attribute):
            return 'duplicates'

        # Same attribute, different value(s) → conflict: newer value wins.
        for entry, _ in same_attribute:
            entry_id = getattr(entry, 'id', None)
            if entry_id:
                semantic_memory.remove(entry_id)
        self._learn_candidate(candidate, semantic_memory)
        return 'conflicts'

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
            self._learn_candidate(candidate, semantic_memory)
            return 'new_facts'

        # ── Deterministic fast path (model-independent) ──
        # Single-valued attributes (name, location, ...) are reconciled without
        # an LLM call, so conflict detection is reliable on any reasoning model.
        deterministic = self._reconcile_by_attribute(
            candidate, semantic_memory, existing_entries
        )
        if deterministic is not None:
            return deterministic

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
            self._learn_candidate(candidate, semantic_memory)
            return 'conflicts'

        # DIFFERENT: insert as new knowledge
        if action == 'DIFFERENT':
            self._learn_candidate(candidate, semantic_memory)
            return 'new_facts'

        # Unknown action — do NOT modify SM
        logger.error(
            f"[RECONCILER] Unknown action '{action}' for candidate "
            f'"{candidate.statement[:60]}"'
        )
        return 'errors'
