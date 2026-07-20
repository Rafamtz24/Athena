"""
Athena Memory Consolidator

Reviews the whole of Semantic Memory and removes entries that no longer earn
their place. Everything else in the learning pipeline runs at write time and
therefore only ever applies to the next fact; this is the one pass that can
act on what is already stored.

Why it is needed:
    Write-time gates only bind going forward. When a quality rule is added —
    or tightened — the entries that rule would now reject are still sitting in
    memory, and they stay there forever. "User performs a system health check"
    survived in the real store for exactly this reason: the durability gate
    that rejects it did not exist when it was written.

    Left alone, a fact store accumulates. Accumulation is what turns memory
    from an asset into noise: stale entries crowd the retrieval budget, and
    contradictions accumulate silently until answers get worse for reasons
    that are hard to trace back.

Deliberately deterministic — no LLM call:
    Every rule here is one Athena can already apply without a model (the
    validator's quality gates, exact-duplicate comparison, and single-valued
    attribute conflicts). That makes consolidation reliable regardless of
    which model is loaded, free to run at startup, and impossible to get
    wrong in a way that silently deletes good facts. Judgement calls that
    genuinely need a model — "are these two differently-worded facts the same
    thing?" — are left to the reconciler at write time.

Conservative by construction: a rule that cannot decide leaves the entry
alone. Losing a real fact is worse than keeping a stale one.
"""

import logging
from typing import List

from athena.knowledge.attributes import parse_fact
from athena.knowledge.validator import KnowledgeValidator
from athena.memory.semantic import SemanticMemory

logger = logging.getLogger(__name__)


def _timestamp_of(entry) -> float:
    """Sort key for recency. Entries without a usable timestamp sort oldest,
    so a newer, well-formed entry always wins a conflict against them."""
    stamp = getattr(entry, 'timestamp', None)
    try:
        return stamp.timestamp()
    except (AttributeError, TypeError, ValueError, OSError):
        return 0.0


def consolidate(semantic_memory: SemanticMemory) -> dict:
    """Remove stale, duplicated and contradictory entries from memory.

    Args:
        semantic_memory: The store to clean, modified in place.

    Returns:
        Counts of what was removed:
        {'stale': int, 'duplicates': int, 'conflicts': int, 'removed': int,
         'remaining': int}
    """
    results = {'stale': 0, 'duplicates': 0, 'conflicts': 0, 'removed': 0}

    entries = list(semantic_memory.query())
    doomed: dict = {}  # entry id -> reason, so one entry is only counted once

    def condemn(entry, reason: str) -> None:
        entry_id = getattr(entry, 'id', None)
        if entry_id and entry_id not in doomed:
            doomed[entry_id] = reason

    # ── 1. Entries the current quality gates would now reject ──────────
    # The validator is the single definition of what is not worth storing, so
    # consolidation inherits every rule it gains without being touched.
    for entry in entries:
        content = str(getattr(entry, 'content', entry))
        if KnowledgeValidator._is_low_quality(content):
            condemn(entry, 'stale')

    # ── 2. Exact duplicates, ignoring case, spacing and trailing stops ──
    # Newest wins: if the same fact was stored twice, the later write is the
    # one the user most recently confirmed.
    seen: dict = {}
    for entry in sorted(entries, key=_timestamp_of, reverse=True):
        if getattr(entry, 'id', None) in doomed:
            continue
        key = SemanticMemory.normalize(str(getattr(entry, 'content', entry)))
        if key in seen:
            condemn(entry, 'duplicates')
        else:
            seen[key] = entry

    # ── 3. Single-valued attributes holding more than one value ────────
    # "User lives in X" and "User lives in Y" cannot both be true. parse_fact
    # only recognises attributes where that is genuinely the case, so anything
    # it does not parse is left alone.
    by_attribute: dict = {}
    for entry in entries:
        if getattr(entry, 'id', None) in doomed:
            continue
        fact = parse_fact(str(getattr(entry, 'content', entry)))
        if fact is not None:
            by_attribute.setdefault(fact.key, []).append((entry, fact))

    for _, group in by_attribute.items():
        if len(group) < 2:
            continue
        values = {fact.value_norm for _, fact in group}
        if len(values) < 2:
            continue  # same value repeated — already handled as a duplicate
        # Keep the most recent value, drop the entries stating the others.
        group.sort(key=lambda pair: _timestamp_of(pair[0]), reverse=True)
        winner_value = group[0][1].value_norm
        for entry, fact in group[1:]:
            if fact.value_norm != winner_value:
                condemn(entry, 'conflicts')

    # ── Apply ──────────────────────────────────────────────────────────
    for entry_id, reason in doomed.items():
        semantic_memory.remove(entry_id)
        results[reason] += 1

    results['removed'] = len(doomed)
    results['remaining'] = len(semantic_memory.query())

    if doomed:
        logger.info(
            f"[CONSOLIDATOR] Removed {results['removed']} entries "
            f"(stale={results['stale']}, duplicates={results['duplicates']}, "
            f"conflicts={results['conflicts']}); {results['remaining']} remain."
        )

    return results


def describe(results: dict) -> str:
    """One human-readable line for the counts, or '' if nothing was removed.

    Consolidation edits the user's memory, so it should never be silent — but
    a startup that cleans nothing has nothing to report either.
    """
    if not results.get('removed'):
        return ''

    parts: List[str] = []
    if results.get('stale'):
        parts.append(f"{results['stale']} stale")
    if results.get('duplicates'):
        parts.append(f"{results['duplicates']} duplicate")
    if results.get('conflicts'):
        parts.append(f"{results['conflicts']} outdated")

    detail = ', '.join(parts)
    noun = 'entry' if results['removed'] == 1 else 'entries'
    return f"Tidied memory: removed {results['removed']} {noun} ({detail})."
