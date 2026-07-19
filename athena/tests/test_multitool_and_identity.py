"""
Tests for two features:

  Identity — the reasoning system prompt is delivered in the `system` role so
  the model adopts Athena's identity instead of the base model's (which made it
  insist "I am Qwen"). Verified by the CognitiveEngine passing a system prompt
  containing "Athena" to the provider.

  Multi-tool planning — a "can I run X" compatibility query chains a web search
  (the software's requirements) with a system snapshot (the user's hardware),
  and the Context Budget Manager injects one prompt source per tool.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from athena.planner.planner import (
    plan_tools,
    _is_requirements_check_query,
    _build_requirements_query,
)
from athena.planner.models import PlannerDecision
from athena.tools.router import route_all
from athena.tools.models import ToolContext
from athena.context.manager import ContextBudgetManager
from athena.thought.models import Thought
from athena.context.models import ReasoningContextPackage, ContextSource
from athena.cognition.engine import CognitiveEngine
from athena.config.settings import get_settings


class MockProvider:
    def count_tokens(self, text):
        return len(text) // 4

    def get_context_window(self):
        return 4096


# ──────────────────────────────────────────────────────────────
# Identity
# ──────────────────────────────────────────────────────────────

def test_engine_delivers_athena_identity_as_system_prompt():
    captured = {}

    class RecordingProvider:
        def generate(self, prompt, system=None):
            captured["system"] = system
            captured["prompt"] = prompt
            return "ok"

    engine = CognitiveEngine(RecordingProvider())
    thought = Thought(user_input="who are you?")
    thought.reasoning_package = ReasoningContextPackage(
        sources=[ContextSource(name="user_input", content="who are you?", priority=100)],
        total_tokens=0,
    )
    engine.process(thought)

    assert captured["system"] is not None, "System prompt must be passed to the provider"
    assert "Athena" in captured["system"]
    print("  [OK] Engine delivers Athena identity in the system role")


# ──────────────────────────────────────────────────────────────
# Requirements-check detection
# ──────────────────────────────────────────────────────────────

def test_requirements_check_detection():
    assert _is_requirements_check_query("can I run GTA 6?")
    assert _is_requirements_check_query("do you think i could run gta 6")
    assert _is_requirements_check_query("will Cyberpunk run on my PC?")
    assert _is_requirements_check_query("is my pc able to run Elden Ring")
    # Negatives
    assert not _is_requirements_check_query("what is my GPU")
    assert not _is_requirements_check_query("hello")
    assert not _is_requirements_check_query("what's my dog's name")
    print("  [OK] Requirements-check queries detected, others rejected")


def test_requirements_query_build():
    q = _build_requirements_query("do you think i could run gta 6?")
    assert "gta 6" in q.lower()
    assert "system requirements" in q.lower()
    print(f"  [OK] Built requirements query: {q!r}")


def test_plan_tools_chains_web_and_system():
    decisions = plan_tools(Thought(user_input="can I run GTA 6?"))
    tools = [d.tool for d in decisions]

    # System snapshot is always part of a compatibility check.
    assert "system" in tools, "Compatibility check must include a system snapshot"

    if get_settings().web_search.enabled:
        assert "web" in tools, "Web search should be chained when enabled"
        # Web (requirements) is fetched before the system snapshot.
        assert tools.index("web") < tools.index("system")
        web = next(d for d in decisions if d.tool == "web")
        assert "requirements" in web.query.lower()
    print(f"  [OK] plan_tools chained tools: {tools}")


def test_plan_tools_single_tool_unchanged():
    # A plain query still yields exactly one decision (delegates to plan()).
    decisions = plan_tools(Thought(user_input="hello"))
    assert len(decisions) == 1
    assert decisions[0].tool == "none"
    print("  [OK] Non-compatibility query yields a single decision")


def test_live_info_and_contractions_trigger_web():
    from athena.planner.planner import _is_web_search_query
    # Live topics + contractions that previously slipped through as "no tool"
    # (causing the model to fabricate an answer) now trigger a web search.
    for q in [
        "whats the weather today in lisbon mexico",
        "hows the weather in paris",
        "whats the price of bitcoin",
        "who won the game last night",
    ]:
        assert _is_web_search_query(q), f"Expected web search for {q!r}"
    print("  [OK] Live-info / contraction queries trigger a web search")


def test_personal_queries_do_not_web_search():
    from athena.planner.planner import _is_web_search_query
    # Contraction handling must not hijack personal/memory queries.
    for q in ["whats my name", "what is my dog called", "hello"]:
        assert not _is_web_search_query(q), f"Did not expect web search for {q!r}"
    print("  [OK] Personal queries are not sent to web search")


def test_live_info_not_suppressed_by_memory_overlap():
    from athena.planner.planner import plan
    # A weather query for the user's own city routes to the weather tool even
    # though the city ("Lisbon") is a stored fact — no memory suppression.
    t = Thought(user_input="hows the weather today in lisbon mexico?")
    t.knowledge = "User lives in Lisbon"
    assert plan(t).tool == "weather"

    # A non-weather live-info query mentioning a stored fact still web-searches
    # rather than being short-circuited by the incidental overlap.
    t2 = Thought(user_input="whats the latest news about lisbon")
    t2.knowledge = "User lives in Lisbon"
    assert plan(t2).tool == "web"

    # A genuinely memory-answerable, non-live query still short-circuits.
    t3 = Thought(user_input="what is the capital of france")
    t3.knowledge = "France capital is Paris"
    assert plan(t3).tool == "none"
    print("  [OK] Live-info query not suppressed by incidental memory overlap")


# ──────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────

def test_route_all_skips_none_and_returns_empty():
    contexts = route_all([PlannerDecision(tool="none")], Thought(user_input="x"))
    assert contexts == []
    print("  [OK] route_all skips no-op decisions")


# ──────────────────────────────────────────────────────────────
# Weather routing + temperature disambiguation
# ──────────────────────────────────────────────────────────────

def _tool_for(q):
    return [d.tool for d in plan_tools(Thought(user_input=q))]


def test_temperature_disambiguation():
    # Ambient temperature -> weather
    assert _tool_for("whats the temperature outside") == ["weather"]
    assert _tool_for("but whats the aproximate temperature like?") == ["weather"]
    # CPU/GPU temperature -> system telemetry
    assert _tool_for("is my cpu temperature ok") == ["system"]
    assert _tool_for("whats my gpu temp") == ["system"]
    print("  [OK] 'temperature' routes to weather vs system by hardware context")


def test_weather_queries_route_to_weather_tool():
    for q in ["whats the weather like today on lisbon?",
              "is it hot in paris",
              "will it rain tomorrow"]:
        assert "weather" in _tool_for(q), f"Expected weather tool for {q!r}"
    print("  [OK] Weather queries route to the weather tool")


def test_extract_location():
    from athena.planner.planner import _extract_location
    assert _extract_location("whats the weather like today on lisbon?") == "lisbon"
    assert _extract_location("weather in paris today") == "paris"
    assert _extract_location("hows the weather") == ""
    print("  [OK] Location extraction from weather queries")


def test_weather_tool_parse_and_failure():
    import json
    from unittest.mock import patch
    from athena.tools import weather

    fake = {
        "current_condition": [{
            "temp_C": "22", "FeelsLikeC": "24", "humidity": "50",
            "windspeedKmph": "11", "weatherDesc": [{"value": "Partly cloudy"}],
        }],
        "nearest_area": [{
            "areaName": [{"value": "Lisbon"}],
            "region": [{"value": "Nuevo Leon"}],
            "country": [{"value": "Mexico"}],
        }],
    }

    class FakeResp:
        def __init__(self, d): self._d = json.dumps(d).encode()
        def read(self): return self._d
        def __enter__(self): return self
        def __exit__(self, *a): return False

    with patch("urllib.request.urlopen", return_value=FakeResp(fake)):
        data = weather.fetch_weather("lisbon")
    assert data["temp_c"] == "22"
    assert "Lisbon" in data["location"]
    rendered = weather.format_weather(data)
    assert "22" in rendered and "Partly cloudy" in rendered

    # Any network/parse failure returns None (caller degrades gracefully).
    with patch("urllib.request.urlopen", side_effect=Exception("no net")):
        assert weather.fetch_weather("x") is None
    print("  [OK] Weather tool parses results and fails safely")


# ──────────────────────────────────────────────────────────────
# Context Budget Manager — multiple tool sources
# ──────────────────────────────────────────────────────────────

def test_multiple_tool_sources_injected():
    mgr = ContextBudgetManager(MockProvider())
    web = ToolContext(tool_name="web", content="Web search: gta 6 system requirements | ...")
    system = ToolContext(tool_name="system", content="CPU: Ryzen 5 5600\nGPU VRAM: 8 GB")

    thought = Thought(user_input="can I run gta 6?")
    thought.tool_contexts = [web, system]

    reasoning_pkg, learning_pkg = mgr.compile(thought)
    names = [s.name for s in reasoning_pkg.sources]

    assert "tool:web" in names, "Web tool must be its own reasoning source"
    assert "tool:system" in names, "System tool must be its own reasoning source"

    # Learning's System Snapshot slot must contain ONLY system content.
    assert "Ryzen" in learning_pkg.tool_context_content
    assert "Web search" not in learning_pkg.tool_context_content
    print("  [OK] Both tools injected; learning slot scoped to system snapshot")


if __name__ == "__main__":
    tests = [
        test_engine_delivers_athena_identity_as_system_prompt,
        test_requirements_check_detection,
        test_requirements_query_build,
        test_plan_tools_chains_web_and_system,
        test_plan_tools_single_tool_unchanged,
        test_live_info_and_contractions_trigger_web,
        test_personal_queries_do_not_web_search,
        test_live_info_not_suppressed_by_memory_overlap,
        test_route_all_skips_none_and_returns_empty,
        test_temperature_disambiguation,
        test_weather_queries_route_to_weather_tool,
        test_extract_location,
        test_weather_tool_parse_and_failure,
        test_multiple_tool_sources_injected,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
    print(f"\nResults: {len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
