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
    """Execute the Web Search tool (stub implementation).

    The web search engine is not yet implemented. This stub returns a
    ToolContext indicating that the web tool was requested but is not
    yet available. The architecture fully supports it.
    """
    return ToolContext(
        tool_name="web",
        content="[Web tool not yet implemented. Query would be: {query}]".format(
            query=decision.query or "(no query)"
        ),
        prompt=decision.query,
        metadata={
            "decision_reason": decision.reason,
            "query": decision.query,
            "status": "stub",
        },
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

    # ── Unknown tool — raise so the caller knows a tool is missing ──
    raise ValueError(
        f"Unknown tool '{decision.tool}'. "
        f"Available tools: system, web. "
        f"Register new tools in athena/tools/router.py"
    )
