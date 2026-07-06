"""Knowledge manager placeholder."""

import re
import sys
import traceback
from typing import Any, Dict, List, Optional

from athena.context.models import LearningContextPackage
from .models import KnowledgeEntry, KnowledgeQuery, KnowledgeResult, KnowledgeCandidate


# ── Deterministic extraction gating ────────────────────────────────────
# Lightweight pattern set: if the user input matches any of these patterns,
# extraction is skipped because the interaction contains no learnable facts.

_GREETINGS_RE = re.compile(
    r'^\s*(hi|hello|hey|greetings|hi there|hello there|hey there|'
    r'yo|sup|howdy|good morning|good afternoon|good evening|'
    r'what\'s up|wassup|hiya)\s*[.!]*\s*$',
    re.IGNORECASE,
)

_ACKNOWLEDGEMENTS_RE = re.compile(
    r'^\s*(ok|okay|k|kk|got it|i see|understood|understood|'
    r'sure|alright|fine|right|yeah|yes|no|nope|yep|'
    r'cool|nice|great|awesome|perfect)\s*[.!]*\s*$',
    re.IGNORECASE,
)

_THANKS_RE = re.compile(
    r'^\s*(thanks|thank you|ty|thx|thank you very much|'
    r'thanks a lot|thanks so much|appreciate it|appreciated|'
    r'much appreciated|cheers)\s*[.!]*\s*$',
    re.IGNORECASE,
)

_GOODBYES_RE = re.compile(
    r'^\s*(bye|goodbye|see you|cya|see ya|take care|'
    r'gotta go|talk later|bye bye|bye for now|later|'
    r'farewell|have a good one|have a nice day)\s*[.!]*\s*$',
    re.IGNORECASE,
)

_SHORT_CONFIRMATION_RE = re.compile(r'^\s*\w{1,10}\s*[.!]*\s*$')

# Threshold: any input shorter than this is unlikely to contain facts
_SHORT_INPUT_THRESHOLD = 3


def _is_empty_input(text: str) -> bool:
    """Return True for blank or whitespace-only input."""
    return not text or not text.strip()


def _is_extraction_needed(text: str) -> bool:
    """Deterministic check whether knowledge extraction should run.

    Returns False for greetings, acknowledgements, thanks, goodbyes,
    short confirmations, and empty input.

    Returns True when the input might contain learnable facts.

    This is intentionally lightweight — no LLM call, no external service.
    """
    stripped = text.strip()

    # Trivially empty
    if not stripped:
        return False

    # Very short inputs (< threshold) are unlikely to contain facts
    # unless they look like a substantive short sentence.
    # For safety, only skip if it matches a known non-factual pattern.
    if len(stripped) <= _SHORT_INPUT_THRESHOLD:
        return False

    # Greetings
    if _GREETINGS_RE.match(stripped):
        return False

    # Acknowledgements
    if _ACKNOWLEDGEMENTS_RE.match(stripped):
        return False

    # Thanks
    if _THANKS_RE.match(stripped):
        return False

    # Goodbyes
    if _GOODBYES_RE.match(stripped):
        return False

    # Short single-word confirmations that didn't match above
    if len(stripped.split()) == 1 and len(stripped) <= 10:
        if _SHORT_CONFIRMATION_RE.match(stripped):
            return False

    # Everything else — run extraction
    return True


class KnowledgeManager:
    """Placeholder for knowledge management operations."""

    def __init__(self, working_memory=None, provider=None, memory_manager=None) -> None:
        """Initialize the knowledge manager using WorkingMemory for candidates."""
        self.working_memory = working_memory
        self.provider = provider
        self.memory_manager = memory_manager
        self.knowledge: list[str] = []
        self._ensure_extraction_template_cached()

    def add_entry(self, entry: KnowledgeEntry) -> str:
        """Add a knowledge entry. Returns empty ID."""
        return ""

    def query(self, query: KnowledgeQuery) -> KnowledgeResult:
        """Search knowledge entries. Returns empty results."""
        return KnowledgeResult()

    def get_entries(self) -> List[KnowledgeEntry]:
        """Get all knowledge entries. Returns empty list."""
        return []

    def delete_entry(self, entry_id: str) -> bool:
        """Delete a knowledge entry. Always returns False."""
        return False

    def update_entry(self, entry_id: str, data: Dict[str, Any]) -> bool:
        """Update a knowledge entry. Returns empty ID."""
        return ""

    def add_candidate(self, candidate: KnowledgeCandidate) -> None:
        """Add a knowledge candidate to WorkingMemory."""
        if self.working_memory is not None:
            self.working_memory.store_candidate(
                statement=candidate.statement,
                confidence=candidate.confidence,
                category=candidate.category
            )

    def get_candidates(self) -> List[KnowledgeCandidate]:
        """Get all knowledge candidates from WorkingMemory."""
        if self.working_memory is not None:
            return self.working_memory.get_candidates()
        return []

    # ── Cached extraction prompt (loaded via PromptLoader) ───────────
    # Loaded lazily on first access from athena/prompts/extraction.json.
    _EXTRACTION_PROFILE = None

    # PERFORMANCE: precomputed static prefix (system_prompt + instructions +
    # response_format), built once in __init__ so every _build_extraction_prompt()
    # call skips re-joining the static profile fields.
    _EXTRACTION_TEMPLATE_PREFIX = None

    def _get_extraction_profile(self):
        """Lazy-load the extraction prompt profile.

        Loaded once via PromptLoader, then cached at class level.
        Editing athena/prompts/extraction.json will take effect when the
        cache is cleared (PromptLoader.clear_cache('extraction')).
        """
        if self.__class__._EXTRACTION_PROFILE is None:
            from athena.prompt.loader import PromptLoader
            self.__class__._EXTRACTION_PROFILE = PromptLoader.load("extraction")
        return self.__class__._EXTRACTION_PROFILE

    def _ensure_extraction_template_cached(self) -> None:
        """Build and cache the static extraction prompt prefix, once per class."""
        if self.__class__._EXTRACTION_TEMPLATE_PREFIX is None:
            profile = self._get_extraction_profile()
            self.__class__._EXTRACTION_TEMPLATE_PREFIX = (
                profile.system_prompt + "\n\n"
                + profile.instructions + "\n\n"
                + profile.response_format
            )

    def _build_extraction_prompt(self, package: LearningContextPackage) -> str:
        """Build extraction prompt from a LearningContextPackage.

        The prompt enforces a strict output contract so the LLM returns
        only durable knowledge explicitly stated in the conversation.
        No concrete examples are provided to prevent hallucination.

        The Context Budget Manager has already determined which context
        sources are visible for learning via learning_visible metadata.
        The KnowledgeManager remains completely unaware of individual tools.

        When tool_context_content is present in the package, the extractor
        MAY learn stable long-term hardware facts from the System Snapshot,
        but MUST NOT learn transient runtime values.

        PERFORMANCE: The static prompt prefix is cached in
        _EXTRACTION_TEMPLATE_PREFIX. Only the conversation content and tool
        context content are built dynamically per request.
        """
        profile = self._get_extraction_profile()
        conversation = package.conversation
        tool_context_content = package.tool_context_content

        prompt_parts = [self.__class__._EXTRACTION_TEMPLATE_PREFIX]

        # ── Tool Context extraction rules ──
        if tool_context_content and profile.has("tool_context_rules"):
            prompt_parts.append("\n\n")
            prompt_parts.append(profile.tool_context_rules)
            prompt_parts.append(f"\nSystem Snapshot:\n{tool_context_content}\n\n")

        prompt_parts.append("\n\nConversation:\n")
        prompt_parts.append(conversation)

        return "".join(prompt_parts)

    # ── Special query pattern for "what do you remember about me" ────
    # Matches queries that explicitly ask for ALL stored knowledge.
    _MEMORY_QUERY_RE = re.compile(
        r'(what\s+do\s+you\s+(remember|know|have))\s+.*'
        r'|(tell\s+me\s+(everything|all)\s+(you\s+)?(remember|know))'
        r'|(what\s+(information|facts|knowledge)\s+do\s+you\s+(have|remember|store))'
        r'|(what\s+can\s+you\s+tell\s+me\s+(about\s+me|about\s+myself))'
        r'|(recall\s+(everything|all)\s+(you\s+)?(know|remember))',
        re.IGNORECASE,
    )

    @staticmethod
    def _stem(word: str) -> str:
        """Strip a trailing plural / 3rd-person 's' so related forms match.

        This lets "lives"/"live" and "dogs"/"dog" match while keeping
        genuinely different words apart — critically, "name" and "named"
        stem to different roots, so a query about a name no longer matches
        an entry that merely contains "named". Only a single trailing 's'
        is removed, and only when at least 3 characters remain, avoiding
        over-aggressive trimming (e.g. "is" is left untouched).
        """
        if word.endswith('s') and len(word) - 1 >= 3:
            return word[:-1]
        return word

    @staticmethod
    def _score_relevance(entry: str, query_words: list[str]) -> float:
        """Compute a weighted relevance score for an entry against query words.

        Matching is done on whole, stemmed words (not substrings): the entry
        is tokenized into words, each is stemmed, and a query word counts as a
        match only if its stem equals a stemmed entry token. This prevents
        false matches like query "name" against entry "named".

        Scoring factors:
        1. Match ratio: proportion of query words found in the entry (0-1)
        2. Longer matched words carry more weight (more specific terms)

        Args:
            entry: A semantic memory entry string.
            query_words: List of normalized discriminative query words.

        Returns:
            A relevance score (0.0 = no match, higher = more relevant).
        """
        entry_tokens = {
            KnowledgeManager._stem(tok)
            for tok in re.findall(r'[a-z]+', entry.lower())
        }
        if not entry_tokens or not query_words:
            return 0.0

        matched_count = 0
        total_weight = 0.0
        for word in query_words:
            if KnowledgeManager._stem(word) in entry_tokens:
                matched_count += 1
                total_weight += len(word)  # Longer words carry more weight

        if matched_count == 0:
            return 0.0

        match_ratio = matched_count / len(query_words)
        avg_weight = total_weight / matched_count

        # Final score: match ratio * average weight normalization
        return match_ratio * (1.0 + avg_weight / 10.0)

    def retrieve(self, query: str) -> Optional[str]:
        """Retrieve relevant semantic memory based on user query.
        
        Uses deterministic relevance scoring:
        1. Special case: "What do you remember about me?" → all entries
        2. Extract discriminative words from the query
        3. Score each entry by lexical overlap with query words
        4. Return only entries above relevance threshold
        
        Returns None when no relevant knowledge is found (no fallback to
        returning all entries, preventing irrelevant fact injection).
        """
        # Get entries from SemanticMemory (single source of truth)
        entries = []
        if self.memory_manager is not None:
            semantic_entries = self.memory_manager.query_semantic()
            for entry in semantic_entries:
                if hasattr(entry, 'content') and entry.content:
                    entries.append(str(entry.content))
        else:
            # Fallback to self.knowledge if no memory_manager configured
            entries = list(self.knowledge)

        if not entries:
            return None

        # Special case: explicit memory recall query → return everything
        if self._MEMORY_QUERY_RE.search(query):
            return "\n".join(entries)

        # Normalize query: lowercase, extract alpha-only words
        # Lowercase BEFORE tokenizing: matching on [a-z]+ against a mixed-case
        # query would drop the leading capital of each word (e.g. "User" -> "ser").
        query_words = re.findall(r'[a-z]+', query.lower())

        # Filter out common stop words that are too generic to be discriminative
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'can', 'could', 'should', 'would', 'what', 'which', 'who',
                      'how', 'when', 'where', 'why', 'not', 'no', 'and', 'or',
                      'but', 'for', 'with', 'at', 'on', 'in', 'to', 'of', 'it',
                      'this', 'that', 'you', 'i', 'we', 'they', 'he', 'she'}

        discriminative_words = [w for w in query_words if len(w) > 2 and w not in stop_words]

        # No discriminative words → no specific query → retrieve nothing
        # This prevents greetings and vague statements from injecting facts.
        if not discriminative_words:
            return None

        # Score each entry by relevance, keeping only those at or above the
        # relevance threshold. A weak incidental overlap on a single short
        # word is not enough to inject a fact into the prompt — this is the
        # main guard against unrelated memories surfacing mid-conversation.
        min_score = getattr(self, '_min_relevance_score', 0.3)
        scored_entries = []
        for entry in entries:
            score = self._score_relevance(entry, discriminative_words)
            if score >= min_score:
                scored_entries.append((score, entry))

        # No relevant entries found → return nothing
        # This prevents unrelated facts from being injected into the prompt.
        if not scored_entries:
            return None

        # Sort by score descending, return top entries
        scored_entries.sort(key=lambda x: x[0], reverse=True)

        # Apply max_results limit from settings
        max_results = getattr(self, '_max_retrieval_results', 10)

        return "\n".join(entry for _, entry in scored_entries[:max_results])

    def extract_candidates(self, package: LearningContextPackage) -> List[Any]:
        """Extract knowledge candidates from a LearningContextPackage.

        The Context Budget Manager has already determined which context
        sources are visible for learning via learning_visible metadata.
        The KnowledgeManager remains completely unaware of individual tools.

        The extraction prompt is built exclusively from the package contents.
        If the provider fails (e.g., unavailable, network error), returns an
        empty list without corrupting Semantic Memory. Learning is skipped gracefully.

        PERFORMANCE: A lightweight deterministic gate runs BEFORE building the
        extraction prompt. Greetings, thanks, goodbyes, and short confirmations
        are detected and skipped — avoiding an unnecessary provider call.

        Args:
            package: A LearningContextPackage containing conversation and
                     optional tool context that is visible for learning.

        Returns:
            List of KnowledgeCandidate objects extracted from the package.
        """
        if self.provider is None:
            return []

        # ── PERFORMANCE: Deterministic extraction gate ──
        # Before building the extraction prompt or calling the provider,
        # check whether the current user input contains anything learnable.
        # Extract the most recent user message from the conversation.
        _last_user_input = ""
        for _line in reversed(package.conversation.split("\n")):
            _stripped = _line.strip()
            if _stripped.startswith("User:") or _stripped.startswith("User :"):
                _last_user_input = _stripped.split(":", 1)[1].strip()
                break

        if not _is_extraction_needed(_last_user_input or package.conversation):
            return []

        try:
            prompt = self._build_extraction_prompt(package)
            response = self.provider.call(prompt)
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            print(f"\n[EXTRACT] Provider call() threw exception:")
            print(f"[EXTRACT] Exception type: {exc_type.__name__}")
            print(f"[EXTRACT] Exception message: {exc_value}")
            print(f"[EXTRACT] Full traceback:\n{tb_str}")
            # Provider failed — skip learning, do not corrupt memory
            return []

        # Parse response into candidate facts following the output contract.
        # Expected format: one fact per line, or "NONE" if no facts.
        extracted_candidates = []
        if not response:
            return extracted_candidates

        text = str(response).strip()

        # If the LLM returns NONE, produce zero candidates
        if text == "NONE":
            return extracted_candidates

        for line in text.split("\n"):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip lines that are too short (likely formatting artifacts)
            if len(line) < 5:
                continue

            # Skip lines containing markdown formatting
            if any(marker in line for marker in ['```', '``', '**', '__', '==']):
                continue

            # Skip lines that are formatting characters
            if all(c in '-*#>`_' for c in line[:3]):
                continue

            # Skip lines that indicate conversation labels or formatting
            if any(line.startswith(prefix) for prefix in
                   ['User:', 'Assistant:', '- ', '* ', '# ', '> ', '`',
                    '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '0.',
                    'Fact:', 'Facts:', 'Example', 'Examples',
                    'Long-term', 'Short-term', 'Conversation', 'Summary',
                    'Valid format', 'Invalid format']):
                continue

            # Skip lines that look like assistant commentary or meta-content
            if any(line.lower().startswith(p) for p in
                   ['the assistant', 'this conversation', 'in this conversation',
                    'the user:', 'assistant:', 'note:', 'summary:',
                    'in this conversation', 'these facts', 'this response',
                    'here are', 'below are', 'the following']):
                continue

            # Skip lines that contain conversation role labels anywhere
            if 'User:' in line or 'Assistant:' in line:
                continue

            from athena.knowledge.models import KnowledgeCandidate
            candidate = KnowledgeCandidate(
                statement=line,
                confidence=0.8,
                category="extracted"
            )
            extracted_candidates.append(candidate)
            if self.working_memory is not None:
                self.working_memory.store_candidate(
                    statement=candidate.statement,
                    confidence=candidate.confidence,
                    category=candidate.category
                )

        return extracted_candidates

    def add(self, knowledge) -> None:
        """Add a knowledge entry to the storage."""
        self.knowledge.append(knowledge)

    def all(self):
        """Return all stored knowledge entries."""
        return self.knowledge

    def promote_candidate(self, candidate) -> bool:
        """Promote a candidate fact to semantic memory. Returns True if successful."""
        from athena.memory.models import MemoryEntry
        entry = MemoryEntry(content=candidate.statement, metadata={
            "type": "knowledge",
            "confidence": candidate.confidence,
            "category": candidate.category
        })
        self.knowledge.append(entry.content)
        return True

    def discard_candidate(self, index: int) -> bool:
        """Remove a candidate from WorkingMemory by index."""
        if self.working_memory is not None:
            return self.working_memory.remove_candidate(index)
        return False
