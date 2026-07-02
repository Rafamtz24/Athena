"""
Regression tests for Planner/Router state isolation bug.

Verifies that:
    ✓ /system works
    ✓ Automatic system planning works
    ✓ Greetings still work
    ✓ Working-memory recall still works
    ✓ Empty user input no longer breaks Athena
    ✓ After empty input, /system still works
    ✓ After any tool exception, subsequent requests still work
    ✓ Multiple consecutive tool invocations succeed

Each test creates a fresh AthenaBrain to simulate independent sessions,
then exercises the exact scenario described in the bug report.
"""

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from athena.brain.brain import AthenaBrain
from athena.planner.models import PlannerDecision
from athena.planner.planner import plan
from athena.tools.router import route
from athena.tools.models import ToolContext


# ─────────────────────────────────────────────────────────
# Mock Thought for planner tests
# ─────────────────────────────────────────────────────────

class _MockThought:
    """Minimal thought mock for testing planner decisions."""
    def __init__(self, user_input: str = "", history: list = None, knowledge=None):
        self.user_input = user_input
        self.history = history or []
        self.knowledge = knowledge


# ─────────────────────────────────────────────────────────
# Tool Planner Tests
# ─────────────────────────────────────────────────────────

class TestPlannerStateIsolation(unittest.TestCase):
    """Planner must be stateless — output depends ONLY on input."""

    def test_direct_system_command(self):
        """/system commands must be detected."""
        thought = _MockThought(user_input="/system do a health check")
        decision = plan(thought)
        self.assertEqual(decision.tool, "system")
        self.assertEqual(decision.prompt, "do a health check")

    def test_automatic_system_planning(self):
        """System health keywords must trigger automatic planning."""
        thought = _MockThought(user_input="do a system health check please")
        decision = plan(thought)
        self.assertEqual(decision.tool, "system")

    def test_greeting_no_tool(self):
        """Greetings must NOT trigger any tool."""
        thought = _MockThought(user_input="hello")
        decision = plan(thought)
        self.assertEqual(decision.tool, "none")

    def test_empty_input_no_tool(self):
        """Empty input must NOT corrupt planner state."""
        thought = _MockThought(user_input="")
        decision = plan(thought)
        self.assertEqual(decision.tool, "none")

    def test_blank_input_no_tool(self):
        """Whitespace-only input must NOT corrupt planner state."""
        thought = _MockThought(user_input="   ")
        decision = plan(thought)
        self.assertEqual(decision.tool, "none")

    def test_planner_stateless_across_calls(self):
        """Multiple planner calls must not leak state between them."""
        # Call planner with greeting first
        thought1 = _MockThought(user_input="hello")
        d1 = plan(thought1)
        self.assertEqual(d1.tool, "none")

        # Call planner with system command immediately after
        thought2 = _MockThought(user_input="/system do a health check")
        d2 = plan(thought2)
        self.assertEqual(d2.tool, "system")

    def test_planner_stateless_after_empty(self):
        """Calling planner with empty input must not break subsequent calls."""
        # Empty input
        thought1 = _MockThought(user_input="")
        d1 = plan(thought1)
        self.assertEqual(d1.tool, "none")

        # System command immediately after empty input
        thought2 = _MockThought(user_input="/system do a health check")
        d2 = plan(thought2)
        self.assertEqual(d2.tool, "system")
        self.assertEqual(d2.prompt, "do a health check")

    def test_working_memory_recall_no_tool(self):
        """Queries that recall working memory must NOT trigger a tool."""
        thought = _MockThought(
            user_input="what did i say earlier",
            history=["User: hello", "Assistant: Hi there!"],
        )
        decision = plan(thought)
        self.assertEqual(decision.tool, "none")

    def test_health_query_with_history(self):
        """System health query must work even with conversation history."""
        thought = _MockThought(
            user_input="do a system health check",
            history=["User: hello", "Assistant: Hi there!"],
        )
        decision = plan(thought)
        self.assertEqual(decision.tool, "system")


# ─────────────────────────────────────────────────────────
# Tool Router Tests
# ─────────────────────────────────────────────────────────

class TestRouterStateIsolation(unittest.TestCase):
    """Router must be stateless — each call independent."""

    def test_router_returns_none_for_noop(self):
        """Decision with tool='none' must return None from router."""
        decision = PlannerDecision(tool="none")
        result = route(decision, thought=None)
        self.assertIsNone(result)

    def test_router_returns_context_for_system(self):
        """Decision with tool='system' must produce a ToolContext."""
        decision = PlannerDecision(tool="system", prompt="health check")
        result = route(decision, thought=_MockThought())
        self.assertIsNotNone(result)
        self.assertIsInstance(result, ToolContext)
        self.assertEqual(result.tool_name, "system")

    def test_router_is_stateless(self):
        """Multiple router calls must not interfere."""
        decision1 = PlannerDecision(tool="none")
        decision2 = PlannerDecision(tool="system", prompt="health check")

        r1 = route(decision1, thought=None)
        r2 = route(decision2, thought=_MockThought())

        self.assertIsNone(r1)
        self.assertIsNotNone(r2)
        self.assertIsInstance(r2, ToolContext)

    def test_router_after_empty_decision_still_works(self):
        """Router must work correctly after handling a no-op decision."""
        decision1 = PlannerDecision(tool="none")
        route(decision1, thought=None)

        decision2 = PlannerDecision(tool="system", prompt="health check")
        result = route(decision2, thought=_MockThought())
        self.assertIsNotNone(result)
        self.assertIsInstance(result, ToolContext)


# ─────────────────────────────────────────────────────────
# Full Pipeline Integration Tests
# ─────────────────────────────────────────────────────────

class TestPipelineStateIsolation(unittest.IsolatedAsyncioTestCase):
    """End-to-end tests using mocked provider to verify state isolation."""

    async def asyncSetUp(self):
        """Create a fresh brain for each test."""
        self.brain = AthenaBrain()
        # Replace the real provider with a mock
        self.mock_provider = MagicMock()
        self.brain.provider = self.mock_provider
        self.brain.pipeline.provider = self.mock_provider

    async def test_system_command_works(self):
        """/system must return successfully."""
        self.mock_provider.generate.return_value = "System check complete."
        response = await self.brain.process("/system do a health check")
        self.assertEqual(response, "System check complete.")

    async def test_automatic_system_planning(self):
        """Health query without /system must trigger system tool."""
        self.mock_provider.generate.return_value = "Health check done."
        response = await self.brain.process("do a system health check please")
        self.assertEqual(response, "Health check done.")

    async def test_greeting_still_works(self):
        """Simple greeting must work without any tool."""
        self.mock_provider.generate.return_value = "Hello! How can I help?"
        response = await self.brain.process("hello")
        self.assertEqual(response, "Hello! How can I help?")

    async def test_empty_input_does_not_break_athena(self):
        """Empty input must be handled gracefully."""
        self.mock_provider.generate.return_value = "How can I help you?"
        response = await self.brain.process("")
        # Must return a response, not crash
        self.assertIsNotNone(response)

    async def test_system_works_after_empty_input(self):
        """After empty input, /system must still work."""
        self.mock_provider.generate.side_effect = [
            MockException("Provider failed"),  # Simulate provider crash on empty input
            "System check complete.",           # System command works after
        ]
        # First send empty input — provider throws
        response1 = await self.brain.process("")
        self.assertIsNotNone(response1)

        # Then send /system — must work
        response2 = await self.brain.process("/system do a health check")
        self.assertEqual(response2, "System check complete.")

    async def test_consecutive_tool_invocations_succeed(self):
        """Multiple consecutive system tool invocations must all succeed."""
        self.mock_provider.generate.return_value = "System OK."
        for i in range(5):
            response = await self.brain.process("/system check")
            self.assertEqual(response, "System OK.", msg=f"Failed on iteration {i+1}")

    async def test_mixed_greeting_and_system_invocations(self):
        """Mixed greetings and system commands must all work."""
        self.mock_provider.generate.side_effect = [
            "Hello!",             # greeting
            "System OK.",         # /system
            "Hi again!",          # greeting
            "System OK again.",   # /system
        ]

        r1 = await self.brain.process("hello")
        self.assertEqual(r1, "Hello!")

        r2 = await self.brain.process("/system check")
        self.assertEqual(r2, "System OK.")

        r3 = await self.brain.process("hi")
        self.assertEqual(r3, "Hi again!")

        r4 = await self.brain.process("/system status")
        self.assertEqual(r4, "System OK again.")

    async def test_working_memory_preserved_across_requests(self):
        """Working memory must survive normal request processing."""
        self.mock_provider.generate.return_value = "I remember."
        await self.brain.process("my name is TestUser")
        # Check that history was saved
        self.assertTrue(any("TestUser" in entry for entry in self.brain.history))

    async def test_provider_exception_isolated(self):
        """A provider exception must NOT break future requests."""
        # Provider fails on first call, succeeds on second
        self.mock_provider.generate.side_effect = [
            MockException("Provider crashed!"),
            "Recovered successfully.",
        ]

        r1 = await self.brain.process("/system health")
        self.assertIsNotNone(r1)  # Should get fallback response

        r2 = await self.brain.process("/system health")
        self.assertEqual(r2, "Recovered successfully.")

    async def test_working_memory_cleared_after_each_request(self):
        """Working memory must be cleared after each request."""
        self.mock_provider.generate.return_value = "OK."
        await self.brain.process("hello")
        wm_entries = self.brain.memory_manager.get_working()
        # Working memory should be clean for next request
        self.assertEqual(len(wm_entries), 0)

    async def test_planner_decision_fresh_per_request(self):
        """Each request must produce a fresh planner decision."""
        self.mock_provider.generate.return_value = "Hello!"
        thought = self.brain.pipeline.create("hello")
        thought.history = list(self.brain.history)
        await self.brain.pipeline.process(thought)

        # planner_decision must be set on the thought for this request
        self.assertIsNotNone(thought.planner_decision)
        self.assertEqual(thought.planner_decision.tool, "none")


class MockException(Exception):
    """Custom exception for simulating provider failures."""
    pass


# ─────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)