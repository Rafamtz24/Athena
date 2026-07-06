"""Tool Router — the single component responsible for executing tools.

The Tool Router receives a PlannerDecision and executes the requested tool.
It is the ONLY component in Athena that invokes tool implementations.

Each tool:
1. Registers itself with the router (or is called by name).
2. Produces a ToolContext.
3. Returns ToolContext to PromptBuilder.

Future tools (web search, document reader, vision, calendar, filesystem,
email, code repository) should simply add a handler here without any
architectural redesign.
"""

from typing import Any, Optional

from athena.logging.logger import logger
from athena.planner.models import PlannerDecision
from athena.tools.models import ToolContext


# ── Registered tool handlers ─────────────────────────────────────────

def _execute_system_tool(
    decision: PlannerDecision,
    thought: Any,
    memory_manager: Optional[Any] = None,
    provider: Optional[Any] = None,
) -> ToolContext:
    """Execute the System Snapshot tool.

    Gathers live hardware telemetry, Athena runtime info, and memory stats.
    Produces a ToolContext with the rendered snapshot.
    """
    from athena.tools.system_snapshot import generate_system_snapshot
    from athena.config.settings import get_settings

    settings = get_settings()

    # Gather Athena runtime info
    provider_info = {
        "provider": settings.provider.provider,
        "reasoning_model": settings.provider.reasoning_model,
        "backend": getattr(settings.provider, "backend", "N/A"),
        "gpu_layers": getattr(settings.provider, "gpu_layers", "N/A"),
        "context_size": getattr(settings, "context_size", "N/A"),
        "threads": getattr(settings, "threads", "N/A"),
        "batch_size": getattr(settings, "batch_size", "N/A"),
    }

    # Gather memory info if available
    memory_info = {}
    if memory_manager is not None:
        try:
            wm_size = len(memory_manager.working_memory.retrieve()) if memory_manager.working_memory else 0
            sm_count = len(memory_manager.query_semantic()) if hasattr(memory_manager, "query_semantic") else 0
            ch_size = len(getattr(thought, "history", []))
            memory_info = {
                "working_memory_size": wm_size,
                "semantic_memory_count": sm_count,
                "chat_history_size": ch_size,
            }
        except Exception:
            pass

    # Generate the snapshot
    snapshot_content = generate_system_snapshot(
        tool_prompt=decision.prompt,
        provider_info=provider_info,
        memory_info=memory_info,
    )

    # ── Console notification ──
    print("Performed system check.")

    return ToolContext(
        tool_name="system",
        content=snapshot_content,
        prompt=decision.prompt,
        metadata={
            "decision_reason": decision.reason,
            "provider_info": provider_info,
            "memory_info": memory_info,
        },
    )


def _execute_web_tool(decision: PlannerDecision) -> ToolContext:
    """Execute the Web Search tool.

    Uses the configured web search provider (default: DuckDuckGo) to
    perform a live web search. Produces a ToolContext with the results.

    If the search fails (no connection, timeout, empty results, provider
    error), logs the failure and returns a minimal ToolContext without
    crashing Athena.
    """
    from athena.config.settings import get_settings
    from athena.tools.web_search import get_provider

    settings = get_settings()
    ws_settings = settings.web_search

    # ── Resolve the provider ──
    provider_name = ws_settings.provider
    provider_func = get_provider(provider_name)

    if provider_func is None:
        logger.warning("Web search provider '%s' not found.", provider_name)
        return ToolContext(
            tool_name="web",
            content="[Web search provider '{provider}' not configured.]".format(
                provider=provider_name
            ),
            prompt=decision.query,
            metadata={
                "decision_reason": decision.reason,
                "query": decision.query,
                "status": "provider_not_found",
            },
        )

    # ── Execute the search ──
    query = decision.query or ""
    if not query.strip():
        return ToolContext(
            tool_name="web",
            content="",
            prompt="",
            metadata={
                "decision_reason": "Empty query — no search performed.",
                "query": "",
                "status": "empty_query",
            },
        )

    results = provider_func(
        query=query,
        max_results=ws_settings.max_results,
        timeout=ws_settings.timeout,
        user_agent=ws_settings.user_agent,
    )

    # ── Handle failure ──
    if results is None:
        logger.error("Web search failed for query: %s", query)
        return ToolContext(
            tool_name="web",
            content="",
            prompt=query,
            metadata={
                "decision_reason": decision.reason,
                "query": query,
                "status": "failed",
            },
        )

    # ── Handle empty results ──
    if not results:
        return ToolContext(
            tool_name="web",
            content="[No results found for query: {query}]".format(query=query),
            prompt=query,
            metadata={
                "decision_reason": decision.reason,
                "query": query,
                "status": "empty_results",
            },
        )

    # ── Format results (compact) ──
    # PERFORMANCE: Compact format reduces token consumption in the
    # reasoning prompt. Each result is a single line with title, URL,
    # and snippet separated by pipes. This is ~40% more token-efficient
    # than the multi-line format while preserving all information.
    lines = [
        f"Web search: {query}",
        "",
    ]
    for i, result in enumerate(results, start=1):
        title = result.get("title", "(No title)").strip()
        url = result.get("href", "").strip()
        snippet = result.get("body", "(No snippet)").strip()
        # Compact single-line format: "1. Title | URL | Snippet"
        lines.append(
            f"{i}. {title} | {url} | {snippet}"
        )

    content = "\n".join(lines)

    # ── Console notification ──
    print("Performed web search.")

    return ToolContext(
        tool_name="web",
        content=content,
        prompt=query,
        metadata={
            "decision_reason": decision.reason,
            "query": query,
            "result_count": len(results),
            "status": "success",
        },
    )


def _execute_weather_tool(decision: PlannerDecision) -> ToolContext:
    """Execute the Weather tool.

    Fetches current conditions from a weather source (wttr.in). Weather is
    transient, so the ToolContext is NOT learning-visible. On failure, returns
    an empty ToolContext so the model reports it has no data rather than
    fabricating one.
    """
    from athena.config.settings import get_settings
    from athena.tools.weather import fetch_weather, format_weather

    ws = get_settings().web_search
    location = decision.query or ""
    data = fetch_weather(location, timeout=ws.timeout, user_agent=ws.user_agent)

    if data is None:
        logger.warning("Weather lookup failed for location: %s", location or "(auto)")
        return ToolContext(
            tool_name="weather",
            content="",
            prompt=location,
            learning_visible=False,
            metadata={"query": location, "status": "failed"},
        )

    print("Fetched weather.")
    return ToolContext(
        tool_name="weather",
        content=format_weather(data),
        prompt=location,
        learning_visible=False,
        metadata={"query": location, "status": "success"},
    )


def route(
    decision: PlannerDecision,
    thought: Any,
    memory_manager: Optional[Any] = None,
    provider: Optional[Any] = None,
) -> Optional[ToolContext]:
    """Execute a PlannerDecision and return the resulting ToolContext.

    Args:
        decision: The PlannerDecision produced by the Tool Planner.
        thought: The current Thought object (for context).
        memory_manager: MemoryManager instance (needed by some tools).
        provider: LLM provider instance (needed by some tools).

    Returns:
        A ToolContext if a tool was executed, or None if decision.tool == "none".

    Raises:
        ValueError: If decision.tool is unknown/unregistered.
    """
    if not decision.requires_execution:
        return None

    # ── Route to the appropriate tool handler ──
    if decision.tool == "system":
        return _execute_system_tool(decision, thought, memory_manager, provider)

    if decision.tool == "web":
        return _execute_web_tool(decision)

    if decision.tool == "weather":
        return _execute_weather_tool(decision)

    # ── Unknown tool — raise so the caller knows a tool is missing ──
    raise ValueError(
        f"Unknown tool '{decision.tool}'. "
        f"Available tools: system, web, weather. "
        f"Register new tools in athena/tools/router.py"
    )


def route_all(
    decisions: list,
    thought: Any,
    memory_manager: Optional[Any] = None,
    provider: Optional[Any] = None,
) -> list:
    """Execute a list of PlannerDecisions and return their ToolContexts.

    Each decision is routed via `route()`. Decisions that require no execution
    (tool == "none") yield no context and are skipped. The returned list
    preserves execution order, so a compatibility check produces the web
    results first, then the system snapshot.

    Args:
        decisions: PlannerDecisions produced by the Tool Planner.
        thought: The current Thought object (for context).
        memory_manager: MemoryManager instance (needed by some tools).
        provider: LLM provider instance (needed by some tools).

    Returns:
        A list of ToolContext objects (may be empty).
    """
    contexts = []
    for decision in decisions:
        context = route(decision, thought, memory_manager, provider)
        if context is not None:
            contexts.append(context)
    return contexts
