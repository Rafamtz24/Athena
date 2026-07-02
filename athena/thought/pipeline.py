"""
Athena Thought Pipeline

Defines the ThoughtPipeline class that orchestrates the processing of a Thought object
through multiple stages. Each stage publishes an event on the EventBus for decoupled
communication with other subsystems.

EventBus integration:
    - Each pipeline stage publishes a corresponding event
    - Events allow other modules to observe thought lifecycle without tight coupling

Pipeline stages (in order):
    1. _initialize()        — Set metadata, publish ThoughtCreated
    2. _load_memory()       — Load episodic memories (Working Memory)
    3. _load_knowledge()    — Retrieve Semantic Memory
    4. _plan_tool()         — Tool Planner: decide if a tool is needed
    5. _execute_tool()      — Tool Router: execute tool if needed
    6. _reason()            — Publish ReasoningStarted
    7. _plan()              — Publish PlanningStarted
    8. _prepare_tools()     — Verify tool context, publish ToolsPrepared
    9. CognitiveEngine      — Build prompt, call provider, set response
    10. _build_response()   — Publish ResponseGenerated
    11. _extract_candidates() — Extract knowledge candidates
    12. _validate_knowledge() — Validate and promote to Semantic Memory
    13. _reflect()          — Self-evaluate outcome
    14. _finalize()         — Publish ThoughtCompleted
"""

import sys
import traceback
from typing import Any, Optional

from athena.events.bus import get_event_bus
from athena.events.models import Event
from athena.thought.models import Thought
from athena.cognition.engine import CognitiveEngine
from athena.planner.planner import plan as plan_tool
from athena.tools.router import route as execute_tool


class ThoughtPipeline:
    """
    Orchestrates the processing lifecycle of a Thought object.

    The pipeline exposes two public methods:
        create(): Factory method to create and initialize a new Thought.
        process(): Run a Thought through all processing stages.

    Pipeline Stages:
        1. _initialize()        -> publishes ThoughtCreated
        2. _load_memory()       -> publishes MemoryLoaded
        3. _load_knowledge()    -> publishes KnowledgeLoaded
        4. _plan_tool()         -> publishes ToolPlanned
        5. _execute_tool()      -> publishes ToolExecuted
        6. _reason()            -> publishes ReasoningStarted
        7. _plan()              -> publishes PlanningStarted
        8. _prepare_tools()     -> publishes ToolsPrepared
        9. CognitiveEngine      -> PromptBuilder + provider call
        10. _build_response()   -> publishes ResponseGenerated
        11. _extract_candidates() -> publishes CandidatesExtracted
        12. _validate_knowledge() -> validates & promotes to Semantic Memory
        13. _reflect()          -> publishes ReflectionStarted
        14. _finalize()         -> publishes ThoughtCompleted

    Events published:
        - ThoughtCreated
        - MemoryLoaded
        - KnowledgeLoaded
        - ToolPlanned
        - ToolExecuted
        - ReasoningStarted
        - PlanningStarted
        - ToolsPrepared
        - ResponseGenerated
        - CandidatesExtracted
        - KnowledgeValidated
        - ReflectionStarted
        - ThoughtCompleted
    """

    def __init__(self, memory_manager=None, knowledge_manager=None, provider=None):
        self.memory_manager = memory_manager
        self.knowledge_manager = knowledge_manager
        self.provider = provider

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

        EVERY stage is isolated: exceptions from a single stage are caught,
        logged, and the pipeline continues. This guarantees that a failed
        tool invocation or provider error NEVER corrupts future requests.

        Args:
            thought: The Thought object to process.

        Returns:
            The final response string from the thought, or None if not set.
        """
        try:
            # Reasoning Phase: Load memory and knowledge, generate response
            self._initialize(thought)
            self._load_memory(thought)        # Working Memory
            self._load_knowledge(thought)     # Semantic Memory
            self._plan_tool(thought)          # Tool Planner (decides if tool needed)
            self._execute_tool(thought)       # Tool Router (executes tool if needed)
            self._reason(thought)
            self._plan(thought)
            self._prepare_tools(thought)      # Verify tool context, publish event
            engine = CognitiveEngine(self.provider)
            thought = engine.process(thought)
            self._build_response(thought)
        except Exception:
            # Stage-level failure caught — isolate, do not poison future requests
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            print(f"\n[PIPELINE TRACE] Reasoning phase exception at stage='{thought.metadata.get('stage', 'unknown')}'")
            print(f"[PIPELINE TRACE] Exception type: {exc_type.__name__}")
            print(f"[PIPELINE TRACE] Exception message: {exc_value}")
            print(f"[PIPELINE TRACE] Full traceback:\n{tb_str}")
            thought.trace["pipeline_error"] = {
                "stage": thought.metadata.get("stage", "unknown"),
                "exception_type": exc_type.__name__,
                "exception_message": str(exc_value),
                "traceback": tb_str,
            }
            if thought.get_response() is None:
                thought.set_response(
                    "I'm sorry, I'm currently unable to process your request."
                )

        # Learning Phase: runs even if reasoning failed,
        # but is itself isolated so failures don't propagate.
        try:
            self._extract_candidates(thought)
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            print(f"\n[PIPELINE TRACE] extract_candidates exception:")
            print(f"[PIPELINE TRACE] Exception type: {exc_type.__name__}")
            print(f"[PIPELINE TRACE] Exception message: {exc_value}")
            print(f"[PIPELINE TRACE] Full traceback:\n{tb_str}")
            thought.trace["extract_candidates_error"] = {
                "exception_type": exc_type.__name__,
                "exception_message": str(exc_value),
                "traceback": tb_str,
            }

        try:
            await self._validate_knowledge(thought)
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            print(f"\n[PIPELINE TRACE] validate_knowledge exception:")
            print(f"[PIPELINE TRACE] Exception type: {exc_type.__name__}")
            print(f"[PIPELINE TRACE] Exception message: {exc_value}")
            print(f"[PIPELINE TRACE] Full traceback:\n{tb_str}")
            thought.trace["validate_knowledge_error"] = {
                "exception_type": exc_type.__name__,
                "exception_message": str(exc_value),
                "traceback": tb_str,
            }

        try:
            self._reflect(thought)
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            print(f"\n[PIPELINE TRACE] reflect exception:")
            print(f"[PIPELINE TRACE] Full traceback:\n{tb_str}")
            thought.trace["reflect_error"] = {
                "exception_type": exc_type.__name__,
                "exception_message": str(exc_value),
                "traceback": tb_str,
            }

        try:
            self._finalize(thought)
        except Exception:
            exc_type, exc_type, exc_tb = sys.exc_info()
            tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            print(f"\n[PIPELINE TRACE] finalize exception:")
            print(f"[PIPELINE TRACE] Full traceback:\n{tb_str}")
            thought.trace["finalize_error"] = {
                "exception_type": exc_type.__name__,
                "exception_message": str(exc_value),
                "traceback": tb_str,
            }

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

    def _load_knowledge(self, thought: Thought) -> None:
        """Stage 3: Load knowledge via KnowledgeManager retrieval."""
        if self.knowledge_manager is not None:
            knowledge = self.knowledge_manager.retrieve(thought.user_input)
            thought.knowledge = knowledge
            thought.trace["knowledge"] = {
                "query": thought.user_input,
                "value": knowledge
            }
        else:
            thought.knowledge = None
            thought.trace["knowledge"] = {
                "query": None,
                "value": None
            }
        
        thought.metadata["stage"] = "knowledge_loaded"
        bus = get_event_bus()
        event = Event(
            type="KnowledgeLoaded",
            source="thought_pipeline",
            payload={"user_input": thought.user_input},
            metadata={"stage": "knowledge_loaded"},
        )
        bus.publish(event)

    def _plan_tool(self, thought: Thought) -> None:
        """Stage 4: Run the Tool Planner to decide if a tool is needed.

        The planner has access to:
            - thought.user_input (current user input)
            - thought.history (Working Memory / conversation history)
            - thought.knowledge (Semantic Memory / retrieved knowledge)

        The planner does NOT have access to Tool Context (it doesn't exist yet).
        The planner produces a PlannerDecision stored on the thought.
        """
        decision = plan_tool(thought)
        thought.planner_decision = decision
        thought.trace["planner_decision"] = {
            "tool": decision.tool,
            "query": decision.query,
            "prompt": decision.prompt,
            "reason": decision.reason,
        }
        thought.metadata["planner_decision"] = decision.tool

        thought.metadata["stage"] = "tool_planned"
        bus = get_event_bus()
        event = Event(
            type="ToolPlanned",
            source="thought_pipeline",
            payload={
                "user_input": thought.user_input,
                "decision_tool": decision.tool,
                "decision_reason": decision.reason,
            },
            metadata={"stage": "tool_planned"},
        )
        bus.publish(event)

    def _execute_tool(self, thought: Thought) -> None:
        """Stage 5: Execute the tool if the planner requested one.

        The Tool Router is the ONLY component responsible for invoking tools.
        It receives the PlannerDecision and returns a ToolContext (or None).
        The ToolContext is stored on the thought for prompt injection.
        """
        decision = getattr(thought, "planner_decision", None)
        if decision is None or not decision.requires_execution:
            thought.metadata["tool_executed"] = False
            thought.metadata["stage"] = "tool_executed"
            bus = get_event_bus()
            event = Event(
                type="ToolExecuted",
                source="thought_pipeline",
                payload={
                    "user_input": thought.user_input,
                    "tool": None,
                    "executed": False,
                },
                metadata={"stage": "tool_executed"},
            )
            bus.publish(event)
            return

        # Route and execute the tool
        tool_context = execute_tool(
            decision=decision,
            thought=thought,
            memory_manager=self.memory_manager,
            provider=self.provider,
        )

        if tool_context is not None:
            thought.tool_context = tool_context
            thought.metadata["tool_executed"] = True
            thought.trace["tool_context"] = {
                "tool_name": tool_context.tool_name,
                "content_length": len(tool_context.content),
            }
        else:
            thought.metadata["tool_executed"] = False

        thought.metadata["stage"] = "tool_executed"
        bus = get_event_bus()
        event = Event(
            type="ToolExecuted",
            source="thought_pipeline",
            payload={
                "user_input": thought.user_input,
                "tool": decision.tool,
                "executed": tool_context is not None,
            },
            metadata={"stage": "tool_executed"},
        )
        bus.publish(event)

    def _extract_candidates(self, thought: Thought) -> None:
        """Stage 3a: Extract knowledge candidates from the completed conversation.

        Passes tool context content (if present) so the extractor can learn
        stable hardware facts while rejecting transient runtime values.
        """
        if self.knowledge_manager is not None:
            # Use full conversation context (history + current input + assistant response) for extraction
            # Include history + user input + assistant response so extractor can learn from the complete interaction
            parts = list(thought.history) if thought.history else []
            parts.append(f"User: {thought.user_input}")
            if thought.response:
                parts.append(f"Assistant: {thought.response}")
            conversation = "\n".join(parts) if parts else thought.user_input

            # Get tool context content if present
            tool_context_content = ""
            tool_context = getattr(thought, 'tool_context', None)
            if tool_context is not None and tool_context.content:
                tool_context_content = tool_context.content

            self.knowledge_manager.extract_candidates(conversation, tool_context_content)
        
        thought.metadata["stage"] = "candidates_extracted"
        bus = get_event_bus()
        event = Event(
            type="CandidatesExtracted",
            source="thought_pipeline",
            payload={"user_input": thought.user_input},
            metadata={"stage": "candidates_extracted"},
        )
        bus.publish(event)

    def _reason(self, thought: Thought) -> None:
        """Stage 4: Perform reasoning on the user input."""
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
        """Stage 8: Verify tool context and publish ToolsPrepared event.

        Tool Context is generated by the Tool Router (Stage 5).
        This stage confirms the context is ready for the PromptBuilder
        and publishes the ToolsPrepared event for observability.
        """
        tool_context = getattr(thought, 'tool_context', None)
        if tool_context is not None:
            thought.trace["tool_context_ready"] = {
                "tool_name": tool_context.tool_name,
                "has_content": bool(tool_context.content),
                "content_length": len(tool_context.content) if tool_context.content else 0,
            }

        thought.metadata["stage"] = "tools_prepared"
        bus = get_event_bus()
        event = Event(
            type="ToolsPrepared",
            source="thought_pipeline",
            payload={
                "user_input": thought.user_input,
                "tool_name": tool_context.tool_name if tool_context else None,
            },
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

    async def _validate_knowledge(self, thought: Thought) -> None:
        """Stage 7: Validate candidate facts and reconcile against Semantic Memory.

        Two-phase process:
        1. QUALITY GATE (deterministic):
           - KnowledgeValidator rejects low-quality / placeholder facts
           - KnowledgeValidator rejects exact normalized duplicates (fast-path)
        2. RECONCILIATION (LLM-based):
           - MemoryReconciler compares each remaining candidate against ALL
             existing Semantic Memory entries in a SINGLE provider call
           - Determines: duplicate, conflict, or different
           - Applies deterministic memory modifications

        Provider call count: EXACTLY ONE per candidate that passes validation.
        """
        if self.memory_manager is not None:
            candidates = self.memory_manager.get_candidates()
            if not candidates:
                thought.metadata["knowledge_validation"] = {"candidates": 0}
                thought.metadata["stage"] = "knowledge_validated"
                bus = get_event_bus()
                event = Event(
                    type="KnowledgeValidated",
                    source="thought_pipeline",
                    payload={"user_input": thought.user_input},
                    metadata={"stage": "knowledge_validation"},
                )
                bus.publish(event)
                return

            from athena.knowledge.validator import KnowledgeValidator
            from athena.knowledge.reconciler import MemoryReconciler

            semantic_mem = self.memory_manager.semantic_memory
            validator = KnowledgeValidator(semantic_mem)

            # Phase 1: Deterministic quality gating
            valid_candidates = []
            validation_counts = {'low_quality': 0, 'duplicate': 0, 'valid': 0}

            for candidate in candidates:
                classification, _ = validator.classify(
                    candidate.statement,
                    candidate.confidence,
                    candidate.category,
                )
                validation_counts[classification] = validation_counts.get(classification, 0) + 1

                if classification == 'valid':
                    valid_candidates.append(candidate)
                # low_quality and duplicate are silently discarded

            # Phase 2: LLM-based reconciliation for valid candidates
            reconciliation_results = None

            if valid_candidates and self.provider is not None:
                reconciler = MemoryReconciler(self.provider)
                reconciliation_results = reconciler.reconcile(valid_candidates, semantic_mem)
            elif valid_candidates and self.provider is None:
                # No provider available — do NOT modify SM (fail-safe)
                reconciliation_results = {
                    'processed': len(valid_candidates),
                    'duplicates': 0,
                    'conflicts': 0,
                    'new_facts': 0,
                    'errors': len(valid_candidates),
                }

            # Store metadata
            thought.metadata["knowledge_validation"] = {
                "candidates": len(candidates),
                "validation": validation_counts,
                "reconciliation": reconciliation_results,
            }

            # Clear candidates after processing
            self.memory_manager.working_memory.clear()

        thought.metadata["stage"] = "knowledge_validated"
        bus = get_event_bus()
        event = Event(
            type="KnowledgeValidated",
            source="thought_pipeline",
            payload={"user_input": thought.user_input},
            metadata={"stage": "knowledge_validation"},
        )
        bus.publish(event)

    def _reflect(self, thought: Thought) -> None:
        """Stage 8: Self-evaluate the processing outcome."""
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
