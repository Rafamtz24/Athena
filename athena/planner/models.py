"""PlannerDecision data model.

Represents the output of the Tool Planner: a decision about whether an
external tool is needed and which tool to invoke.
"""

from dataclasses import dataclass, field


@dataclass
class PlannerDecision:
    """Deterministic decision produced by the Tool Planner.

    The planner does NOT execute tools. It only produces a decision.
    The Tool Router is the sole component responsible for invocation.

    Attributes:
        tool: The tool to invoke. One of:
            - "none": No tool needed — Athena has sufficient information.
            - "system": System snapshot — gather live runtime telemetry.
            - "web": Web search — query external information (stub).
        query: Search query for the web tool (empty for non-web tools).
        prompt: Additional prompt or context for the tool (e.g., what to
                focus on in a system snapshot).
        reason: Human-readable explanation of why this decision was made.
    """
    tool: str = "none"
    query: str = ""
    prompt: str = ""
    reason: str = ""

    @property
    def requires_execution(self) -> bool:
        """Returns True if this decision requires a tool to be executed."""
        return self.tool != "none"