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
    "temperature", "temp", "hot", "cooling",
    "usage", "utilization", "running",
    "bottleneck", "throttling",
    "memory", "ram usage", "cpu usage", "gpu usage",
    "how is my", "how healthy",
]

_WEB_SEARCH_KEYWORDS = [
    "search", "search for", "search the web",
    "look up", "find", "find out",
    "what is", "what are", "who is",
    "google", "browse", "internet",
    "latest", "news about", "recent",
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


def _is_system_health_query(user_input: str) -> bool:
    """Detect if user is asking about system health or performance."""
    lower = user_input.lower()
    for kw in _SYSTEM_HEALTH_KEYWORDS:
        if kw in lower:
            return True
    # Detect runtime telemetry questions
    runtime_patterns = [
        r'how.*(fast|hot|much).*(cpu|gpu|ram|memory|computer|system)',
        r'(cpu|gpu|ram|memory|computer).*(temperature|temp|speed|usage|utilization)',
        r'what.*(temperature|temp|speed|usage).*',
        r'check.*(system|health|performance)',
        r'(do|run)\s+a\s+health\s+check',
        r'why\s+is\s+my\s+(computer|pc|system)\s+(slow|lagging|hot)',
    ]
    for pat in runtime_patterns:
        if re.search(pat, lower):
            return True
    return False


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


def _is_web_search_query(user_input: str) -> bool:
    """Detect if user is asking for information likely needing a web search."""
    lower = user_input.lower()

    # Pure personal hardware queries (e.g., "what is my GPU") are NOT web searches
    if _is_personal_hardware_query(user_input):
        return False

    # Check explicit web search commands and prefixes
    for kw in _WEB_SEARCH_KEYWORDS:
        if kw in lower:
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

    # ── Rule 4: Web search query (stub) ──
    # Check BEFORE hardware facts because queries like "latest BIOS version"
    # contain hardware terms but require external information.
    if _is_web_search_query(user_input):
        query = _extract_web_query(user_input)
        return PlannerDecision(
            tool="web",
            query=query,
            reason="Query suggests external information is needed (web search stub).",
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
