"""Tool Planner — decides whether Athena needs an external tool.

The Tool Planner executes AFTER Working Memory and Semantic Memory have
already been loaded. It determines whether Athena already has sufficient
information to answer the user's request or whether a tool is needed.

The planner does NOT execute tools. It only produces a PlannerDecision.
The Tool Router is the sole component responsible for invocation.
"""

import re
from typing import Any, Optional

from athena.planner.models import PlannerDecision


# ── Keywords that suggest tool needs ──────────────────────────────────

_SYSTEM_HEALTH_KEYWORDS = [
    "health", "healthy", "health check",
    "performance", "slow", "speed",
    "usage", "utilization", "running",
    "bottleneck", "throttling",
    "ram usage", "cpu usage", "gpu usage",
    "how is my", "how healthy",
]

# Thermal words are ambiguous: "cpu temperature" is hardware telemetry, but
# "temperature outside" is weather. They count as a system-health signal ONLY
# when a hardware-context word is also present; otherwise they mean weather.
_THERMAL_KEYWORDS = [
    "temperature", "temp", "hot", "cold", "cooling", "overheating", "thermal",
]
_SYSTEM_CONTEXT_WORDS = [
    "cpu", "gpu", "ram", "vram", "pc", "computer", "system", "machine",
    "rig", "processor", "chip", "fan", "hardware", "laptop",
]

# Weather-specific terms always route to the weather tool.
_WEATHER_KEYWORDS = [
    "weather", "forecast", "rain", "raining", "rainy", "snow", "snowing",
    "humidity", "sunny", "cloudy", "storm", "climate",
]

_WEB_SEARCH_KEYWORDS = [
    "search", "search for", "search the web",
    "look up", "find", "find out",
    "what is", "what are", "who is",
    "google", "browse", "internet",
    "latest", "news about", "recent",
]

# Topics that are inherently live / external and cannot be answered from
# memory — they always require a web search for a current, truthful answer.
_LIVE_INFO_KEYWORDS = [
    "weather", "temperature", "forecast", "raining", "snowing",
    "price of", "cost of", "how much is", "stock price", "share price",
    "exchange rate", "conversion rate",
    "news", "headlines",
    "who won", "score", "final score", "match result",
    "release date", "when does", "when is",
]

_HARDWARE_FACT_PREFIXES = [
    "gpu", "graphics card", "graphics",
    "cpu", "processor",
    "ram", "memory",
    "motherboard", "mainboard",
    "storage", "disk", "drive", "ssd", "hdd",
    "display", "monitor", "screen",
    "power supply", "psu",
    "os", "operating system",
    "computer", "pc", "system",
]


def _has_working_memory_info(thought: Any) -> bool:
    """Check if the thought has conversation history context."""
    history = getattr(thought, "history", None)
    return bool(history and len(history) > 0)


def _has_semantic_knowledge(thought: Any) -> bool:
    """Check if the thought has retrieved semantic knowledge."""
    knowledge = getattr(thought, "knowledge", None)
    return knowledge is not None and bool(str(knowledge).strip())


def _knowledge_contains_hardware(knowledge_str: str) -> bool:
    """Check if semantic memory contains hardware/fact information."""
    lower = knowledge_str.lower()
    for prefix in _HARDWARE_FACT_PREFIXES:
        if prefix in lower:
            return True
    # Check for common hardware patterns
    if re.search(r'(rx\s*\d+|rtx\s*\d+|gtx\s*\d+|iris|xeon|core\s*i\d|ryzen\s*\d|threadripper)', lower, re.IGNORECASE):
        return True
    if re.search(r'\d+\s*gb', lower) and any(w in lower for w in ["ram", "memory", "vram", "storage"]):
        return True
    return False


def _has_system_context(lower: str) -> bool:
    """True if the text mentions the user's own hardware/computer."""
    return any(word in lower for word in _SYSTEM_CONTEXT_WORDS)


def _is_system_health_query(user_input: str) -> bool:
    """Detect if user is asking about system health or performance."""
    lower = user_input.lower()
    for kw in _SYSTEM_HEALTH_KEYWORDS:
        if kw in lower:
            return True
    # Thermal words are system telemetry ONLY with hardware context present
    # (e.g. "cpu temperature"); "temperature outside" is weather, not this.
    if any(t in lower for t in _THERMAL_KEYWORDS) and _has_system_context(lower):
        return True
    # Detect runtime telemetry questions (all require an explicit hardware term)
    runtime_patterns = [
        r'how.*(fast|hot|much).*(cpu|gpu|ram|memory|computer|system)',
        r'(cpu|gpu|ram|memory|computer).*(temperature|temp|speed|usage|utilization)',
        r'(temperature|temp|speed|usage)\s+of\s+(my\s+)?(cpu|gpu|ram|memory|computer|pc|system)',
        r'check.*(system|health|performance)',
        r'(do|run)\s+a\s+health\s+check',
        r'why\s+is\s+my\s+(computer|pc|system)\s+(slow|lagging|hot)',
    ]
    for pat in runtime_patterns:
        if re.search(pat, lower):
            return True
    return False


def _is_weather_query(user_input: str) -> bool:
    """Detect a weather query (routes to the dedicated weather tool).

    Weather terms always match. Thermal terms (temperature/hot/cold) match
    only WITHOUT hardware context — with it they are system telemetry.
    """
    lower = user_input.lower()
    if any(kw in lower for kw in _WEATHER_KEYWORDS):
        return True
    if any(t in lower for t in _THERMAL_KEYWORDS) and not _has_system_context(lower):
        return True
    return False


def _extract_location(user_input: str) -> str:
    """Extract a location from a weather query (best effort).

    Looks for a trailing "in/at/on/for <place>" phrase; returns "" when none is
    found (the weather tool then falls back to IP geolocation).
    """
    lower = user_input.lower().rstrip("?.! ")
    match = re.search(r"\b(?:in|at|on|for)\s+([a-z][a-z .,'\-]*)$", lower)
    if not match:
        return ""
    location = match.group(1)
    # Drop trailing time words that are not part of the place name.
    location = re.sub(
        r"\b(today|tomorrow|tonight|right now|now|currently|this week|this weekend)\b",
        "",
        location,
    )
    return location.strip(" ,")


def _is_personal_hardware_query(user_input: str) -> bool:
    """Detect if the query is purely about the user's own hardware specs.

    These queries should use the system tool (not web search) when the
    hardware is unknown, or no tool when hardware is known.
    
    Distinguished from queries like "latest BIOS for my motherboard" which
    contain external-info indicators and are caught as web searches.
    """
    lower = user_input.lower()
    # "my X" patterns like "my GPU", "my CPU", "my RAM"
    has_my_hardware = re.search(r'\bmy\s+(gpu|cpu|ram|memory|computer|pc|system|motherboard|graphics|processor)', lower)
    if not has_my_hardware:
        return False
    # If query also contains external info indicators, it's a web search,
    # not a personal hardware query
    if re.search(r'\b(latest|current|recent|new|upcoming|version|release|update|bios|driver|firmware)\b', lower):
        return False
    return True


def _is_personal_query(user_input: str) -> bool:
    """Detect if the query is about the user's personal information.

    These queries reference 'my' or 'I' and ask about personal attributes
    (name, age, preferences, pets, family, etc.) that should be answered
    from memory, not from web search.
    """
    lower = user_input.lower()

    # Must contain a personal reference
    if not ('my ' in lower or " my" in lower or lower.startswith("my ")):
        return False

    # If query also contains external info indicators, it's a web search
    external_patterns = [
        r'\b(latest|current|recent|new|upcoming|version|release|update)\b',
        r'\b(bios|driver|firmware|price|cost|buy|review)\b',
        r'\b(how (much|many|to)|where (to|can|is))\b',
    ]
    for pat in external_patterns:
        if re.search(pat, lower):
            return False

    return True


def _normalize_contractions(text: str) -> str:
    """Expand common question-word contractions ("whats" -> "what is")."""
    return re.sub(r"\b(what|how|where|who|when)('s|s)\b", r"\1 is", text.lower())


def _is_live_info_query(user_input: str) -> bool:
    """Detect a query about inherently live/external information.

    Weather, prices, news, scores and the like can never be answered from
    memory, so they always need a fresh web search and must never be
    short-circuited by incidental semantic-memory overlap (e.g. the query
    mentions the user's city, which happens to be a stored fact).
    """
    lower = user_input.lower()
    normalized = _normalize_contractions(lower)
    return any(kw in lower or kw in normalized for kw in _LIVE_INFO_KEYWORDS)


def _is_web_search_query(user_input: str) -> bool:
    """Detect if user is asking for information likely needing a web search."""
    lower = user_input.lower()

    # Pure personal hardware queries (e.g., "what is my GPU") are NOT web searches
    if _is_personal_hardware_query(user_input):
        return False

    # Pure personal queries (e.g., "what is my dog's name") are NOT web searches
    if _is_personal_query(user_input):
        return False

    # Live / external topics (weather, prices, news, scores...) can never be
    # answered from memory and always need a fresh web search.
    if _is_live_info_query(user_input):
        return True

    # Normalize common contractions so "whats"/"hows"/"wheres"/"whos" are
    # treated like their expanded forms ("what is", ...). Without this, a query
    # like "whats the weather" slipped past the "what is" keyword and no tool
    # ran, leaving the model to fabricate an answer.
    normalized = _normalize_contractions(lower)

    # Check explicit web search commands and prefixes
    for kw in _WEB_SEARCH_KEYWORDS:
        if kw in lower or kw in normalized:
            return True

    # Detect "latest/current/recent <something>" patterns
    if re.search(r'\b(latest|current|recent|new|upcoming)\b', lower):
        return True

    # Detect explicit version/firmware/driver queries
    if re.search(r'\b(version|release|update|bios|driver|firmware)\b', lower):
        return True

    # Detect "can I run X" or "does my PC meet X requirements" patterns
    if re.search(r'\b(can I run|requirements|system requirements|specs? for|compatible)\b', lower):
        return True

    return False


def _is_requirements_check_query(user_input: str) -> bool:
    """Detect a "can I run X / do I meet the requirements" style query.

    These need BOTH external info (the software's system requirements) and the
    user's local hardware specs to answer, so the planner chains a web search
    with a system snapshot.
    """
    lower = user_input.lower()
    patterns = [
        r'\b(can|could|would|will|should) i (run|play|handle)\b',
        r'\bi (can|could|would|will) (run|play|handle)\b',
        r'\b(will|can|could) it run\b',
        r'\bwill .+ run on\b',
        r'\b(be )?able to run\b',
        r'\benough (to run|for)\b',
        r'\bhandle running\b',
        r'\bdo i meet\b',
        r'\bmeet .*requirements\b',
        r'\bmy (pc|computer|system|rig|setup|machine)\b.*\b(run|play|handle)\b',
        r'\b(run|play|handle)\b.*\bon my (pc|computer|system|rig|setup|machine)\b',
    ]
    return any(re.search(pat, lower) for pat in patterns)


def _build_requirements_query(user_input: str) -> str:
    """Build a web-search query for the software's system requirements.

    Strips common lead-in/filler words so the software (e.g. game) name
    remains, then appends "system requirements".
    """
    text = user_input.strip().rstrip("?.!")
    filler = (
        r'\b(do|does|you|think|can|could|would|will|i|is|it|be|able|to|'
        r'run|running|play|handle|on|my|the|a|pc|computer|system|rig|'
        r'setup|machine|enough|for|meet|requirements?)\b'
    )
    subject = re.sub(filler, ' ', text, flags=re.IGNORECASE)
    subject = re.sub(r'\s+', ' ', subject).strip(" ,")
    if not subject:
        subject = text
    return f"{subject} system requirements".strip()


def _is_direct_system_command(user_input: str) -> bool:
    """Detect explicit /system command."""
    return user_input.strip().lower().startswith("/system")


def _extract_web_query(user_input: str) -> str:
    """Extract a search query from the user input where possible."""
    # If user said "/web something" or similar
    match = re.search(r'/(?:web|search)\s+(.+)', user_input, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Strip common prefixes
    lower = user_input.lower()
    for prefix in ["search for ", "search the web for ", "look up ", "find ", "find out ", "what is ", "what are ", "who is "]:
        if lower.startswith(prefix):
            return user_input[len(prefix):].strip()

    return user_input.strip()


def plan(thought: Any) -> PlannerDecision:
    """Determine whether Athena needs an external tool.

    Args:
        thought: A Thought object with history (working memory) and
                 knowledge (semantic memory) already populated.

    Returns:
        A PlannerDecision specifying which tool (if any) to invoke.
    """
    user_input = getattr(thought, "user_input", "")
    knowledge_str = str(getattr(thought, "knowledge", "") or "")
    has_hardware = _knowledge_contains_hardware(knowledge_str)
    has_history = _has_working_memory_info(thought)

    # ── Rule 0: Empty / blank input — no tool needed ──
    if not user_input or not user_input.strip():
        return PlannerDecision(
            tool="none",
            reason="Empty or blank user input — no action required.",
        )

    # ── Rule 1: Explicit /system command ──
    if _is_direct_system_command(user_input):
        prompt = user_input[len("/system"):].strip()
        return PlannerDecision(
            tool="system",
            prompt=prompt,
            reason="Explicit /system command detected.",
        )

    # ── Rule 2: Health / runtime query ──
    if _is_system_health_query(user_input):
        if has_hardware:
            # Hardware is known but runtime telemetry is always fresh
            return PlannerDecision(
                tool="system",
                reason="System health query with known hardware — runtime telemetry needed.",
            )
        # Hardware is unknown — need system snapshot to learn it
        return PlannerDecision(
            tool="system",
            reason="System health query without known hardware — full snapshot needed.",
        )

    # ── Rule 2b: Weather query → dedicated weather tool ──
    # Runs after system health so "cpu temperature" stays a system check, but
    # before web search so ambient weather uses a real weather source instead
    # of generic search snippets.
    if _is_weather_query(user_input):
        from athena.config.settings import get_settings
        if not get_settings().web_search.enabled:
            return PlannerDecision(
                tool="none",
                reason="Weather needs network access, which is disabled in settings.",
            )
        return PlannerDecision(
            tool="weather",
            query=_extract_location(user_input),
            reason="Weather query — fetching current conditions from a weather source.",
        )

    # ── Rule 3: History recall (no tool needed) ──
    # If user is asking about what they said, working memory has it
    recall_keywords = [
        "what did i", "what was", "what have i",
        "what did you", "you said", "i said",
        "a minute ago", "earlier", "before",
        "previous", "last", "recently",
    ]
    if has_history:
        lower = user_input.lower()
        for kw in recall_keywords:
            if kw in lower:
                return PlannerDecision(
                    tool="none",
                    reason="Information available in working memory (conversation history).",
                )

    # ── Rule 4: Web search query ──
    # Check BEFORE hardware facts because queries like "latest BIOS version"
    # contain hardware terms but require external information.
    if _is_web_search_query(user_input):
        # Check if web search is enabled in settings
        from athena.config.settings import get_settings
        settings = get_settings()
        if not settings.web_search.enabled:
            return PlannerDecision(
                tool="none",
                reason="Web search is disabled in settings — skipping web tool.",
            )

        query = _extract_web_query(user_input)

        # Prefer existing knowledge before requesting a web search — but NEVER
        # for live/external topics. Otherwise an incidental overlap (e.g. the
        # user's stored city appearing in a "weather in <city>" query) would
        # wrongly suppress the search and leave the model with no live data.
        if not _is_live_info_query(user_input) and _has_semantic_knowledge(thought):
            knowledge_lower = knowledge_str.lower()
            query_words = [w.lower() for w in re.findall(r'[a-z]+', query) if len(w) > 3]
            has_relevant_knowledge = any(w in knowledge_lower for w in query_words) if query_words else False
            if has_relevant_knowledge:
                return PlannerDecision(
                    tool="none",
                    reason="Relevant information already exists in semantic memory — no web search needed.",
                )

        return PlannerDecision(
            tool="web",
            query=query,
            reason="Query suggests external or current information is needed — web search requested.",
        )

    # ── Rule 5: Hardware fact question ──
    # If user asks "what's my GPU" and we have it in knowledge → no tool
    for prefix in _HARDWARE_FACT_PREFIXES:
        if prefix in user_input.lower():
            if has_hardware:
                return PlannerDecision(
                    tool="none",
                    reason=f"Hardware information ('{prefix}') available in semantic memory.",
                )
            else:
                # Hardware question but we don't know it — need system snapshot
                return PlannerDecision(
                    tool="system",
                    prompt=user_input,
                    reason=f"Hardware query ('{prefix}') but unknown in semantic memory.",
                )

    # ── Default: No tool needed ──
    return PlannerDecision(
        tool="none",
        reason="Athena has sufficient information to answer.",
    )


def plan_tools(thought: Any) -> list:
    """Determine which tool(s) Athena needs — supports chaining multiple.

    Most queries need at most one tool, in which case this returns a single
    decision (delegating to `plan`). Some queries need several: a
    "can I run X" compatibility check needs both a web search (the software's
    system requirements) and a system snapshot (the user's hardware), so this
    returns both decisions for the Tool Router to execute in order.

    Returns:
        A non-empty list of PlannerDecision objects. When no tool is needed
        the list contains a single "none" decision.
    """
    user_input = getattr(thought, "user_input", "")

    if user_input and user_input.strip() and _is_requirements_check_query(user_input):
        from athena.config.settings import get_settings

        decisions = []
        if get_settings().web_search.enabled:
            decisions.append(PlannerDecision(
                tool="web",
                query=_build_requirements_query(user_input),
                reason="Compatibility check — fetch the software's system requirements.",
            ))
        decisions.append(PlannerDecision(
            tool="system",
            reason="Compatibility check — gather local hardware specs to compare.",
        ))
        return decisions

    # Fall back to single-tool planning (unchanged behavior).
    return [plan(thought)]
