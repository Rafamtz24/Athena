"""
Athena Thought Pipeline

Defines the ThoughtPipeline class that orchestrates the processing of a Thought object
through multiple stages. Each stage publishes an event on the EventBus for decoupled
communication with other subsystems.

EventBus integration:
    - Each pipeline stage publishes a corresponding event
    - Events allow other modules to observe thought lifecycle without tight coupling
"""

from typing import Any, Optional

from athena.events.bus import get_event_bus
from athena.events.models import Event
from athena.thought.models import Thought
from athena.cognition.engine import CognitiveEngine


class ThoughtPipeline:
    """
    Orchestrates the processing lifecycle of a Thought object.

    The pipeline exposes two public methods:
        create(): Factory method to create and initialize a new Thought.
        process(): Run a Thought through all processing stages.

    Each processing stage is implemented as a private method with placeholder
    implementations for future expansion. Events are published at each stage.

    Pipeline Stages:
        _initialize()      -> publishes ThoughtCreated event
        _load_memory()     -> publishes MemoryLoaded event
        _reason()          -> publishes ReasoningStarted event
        _plan()            -> publishes PlanningCompleted event (partial)
        _prepare_tools()   -> publishes ToolsPrepared event
        _build_response()  -> publishes ResponseGenerated event
        _reflect()         -> publishes ReflectionStarted event
        _finalize()        -> publishes ThoughtCompleted event

    Events published:
        - ThoughtCreated
        - MemoryLoaded
        - ReasoningStarted
        - PlanningStarted
        - ToolsPrepared
        - ResponseGenerated
        - ReflectionStarted
        - ThoughtCompleted
    """

    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager

    @staticmethod
    def create(user_input: str) -> Thought:
        """
        Factory method to create a new Thought with initial user input.

        Args:
            user_input: The raw string input from the user.

        Returns:
            Thought: A newly initialized Thought object.
        """
        thought = Thought(user_input=user_input)
        bus = get_event_bus()
        event = Event(
            type="ThoughtCreated",
            source="thought_pipeline",
            payload={"user_input": user_input},
            metadata={"stage": "created"},
        )
        bus.publish(event)
        return thought

    async def process(self, thought: Thought) -> Any:
        """
        Run a Thought through all processing stages sequentially.

        Each stage is called in order and may modify the Thought object.
        Events are published at each stage for observability.
        The final response is extracted from the thought after completion.

        Args:
            thought: The Thought object to process.

        Returns:
            The final response string from the thought, or None if not set.
        """
        self._initialize(thought)
        self._load_memory(thought)
        self._reason(thought)
        self._plan(thought)
        self._prepare_tools(thought)
        engine = CognitiveEngine()
        thought = engine.process(thought)
        self._build_response(thought)
        self._reflect(thought)
        self._finalize(thought)

        return thought.get_response()

    def _initialize(self, thought: Thought) -> None:
        """Stage 1: Initialize the thought with basic metadata."""
        thought.metadata["stage"] = "initialized"
        bus = get_event_bus()
        event = Event(
            type="ThoughtCreated",
            source="thought_pipeline",
            payload={"user_input": thought.user_input},
            metadata={"stage": "initialized"},
        )
        bus.publish(event)

    def _load_memory(self, thought: Thought) -> None:
        """Stage 2: Load relevant episodic memories into the thought."""
        if self.memory_manager is not None:
            try:
                episodic_memories = self.memory_manager.recall()
                if episodic_memories:
                    for mem in episodic_memories:
                        thought.memories.append(mem.content)
            except Exception:
                pass
        
        thought.metadata["stage"] = "memory_loaded"
        bus = get_event_bus()
        event = Event(
            type="MemoryLoaded",
            source="thought_pipeline",
            payload={"user_input": thought.user_input},
            metadata={"stage": "memory_loaded"},
        )
        bus.publish(event)

    def _reason(self, thought: Thought) -> None:
        """Stage 3: Perform reasoning on the user input."""
        thought.metadata["stage"] = "reasoned"
        bus = get_event_bus()
        event = Event(
            type="ReasoningStarted",
            source="thought_pipeline",
            payload={"user_input": thought.user_input},
            metadata={"stage": "reasoning"},
        )
        bus.publish(event)

    def _plan(self, thought: Thought) -> None:
        """Stage 4: Generate an execution plan."""
        thought.metadata["stage"] = "planned"
        bus = get_event_bus()
        event = Event(
            type="PlanningStarted",
            source="thought_pipeline",
            payload={"user_input": thought.user_input},
            metadata={"stage": "planning"},
        )
        bus.publish(event)

    def _prepare_tools(self, thought: Thought) -> None:
        """Stage 5: Prepare tool requests if needed."""
        thought.metadata["stage"] = "tools_prepared"
        bus = get_event_bus()
        event = Event(
            type="ToolsPrepared",
            source="thought_pipeline",
            payload={"user_input": thought.user_input},
            metadata={"stage": "tools_prepared"},
        )
        bus.publish(event)

    def _build_response(self, thought: Thought) -> None:
        """Stage 6: Construct the final response."""
        thought.metadata["stage"] = "response_built"
        bus = get_event_bus()
        event = Event(
            type="ResponseGenerated",
            source="thought_pipeline",
            payload={"user_input": thought.user_input, "response": thought.get_response()},
            metadata={"stage": "response_generated"},
        )
        bus.publish(event)

    def _reflect(self, thought: Thought) -> None:
        """Stage 7: Self-evaluate the processing outcome."""
        thought.reflection = {"status": "completed"}
        thought.metadata["stage"] = "reflected"
        bus = get_event_bus()
        event = Event(
            type="ReflectionStarted",
            source="thought_pipeline",
            payload={"user_input": thought.user_input},
            metadata={"stage": "reflection"},
        )
        bus.publish(event)

    def _finalize(self, thought: Thought) -> None:
        """Stage 8: Finalize the thought and prepare for return."""
        thought.metadata["stage"] = "finalized"
        bus = get_event_bus()
        event = Event(
            type="ThoughtCompleted",
            source="thought_pipeline",
            payload={"user_input": thought.user_input, "response": thought.get_response()},
            metadata={"stage": "completed"},
        )
        bus.publish(event)
