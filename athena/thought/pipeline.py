"""""
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

        Args:
            thought: The Thought object to process.

        Returns:
            The final response string from the thought, or None if not set.
        """
        # Reasoning Phase: Load memory and knowledge, generate response (semantic memory only)
        self._initialize(thought)
        self._load_memory(thought)
        self._load_knowledge(thought)
        self._reason(thought)
        self._plan(thought)
        self._prepare_tools(thought)
        engine = CognitiveEngine(self.provider)
        thought = engine.process(thought)
        self._build_response(thought)
        
        # Learning Phase: Extract, validate knowledge (separate from reasoning)
        self._extract_candidates(thought)
        self._validate_knowledge(thought)
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

    def _extract_candidates(self, thought: Thought) -> None:
        """Stage 3a: Extract knowledge candidates from the completed conversation."""
        if self.knowledge_manager is not None:
            # Use full conversation context (history + current input + assistant response) for extraction
            # Include history + user input + assistant response so extractor can learn from the complete interaction
            parts = list(thought.history) if thought.history else []
            parts.append(f"User: {thought.user_input}")
            if thought.response:
                parts.append(f"Assistant: {thought.response}")
            conversation = "\n".join(parts) if parts else thought.user_input
            self.knowledge_manager.extract_candidates(conversation)
        
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

    def _validate_knowledge(self, thought: Thought) -> None:
        """Stage 7: Validate candidate facts and promote verified ones to semantic memory.
        
        Uses KnowledgeValidator to classify candidates as:
        - Duplicate: already exists in Semantic Memory (rejected)
        - New Fact: unique knowledge worth storing (promoted)
        - Possible Conflict: contradicts existing entry (reconciled via LLM)
        
        For possible conflicts, invokes the reconciler to resolve using LLM.
        """
        from athena.knowledge.validator import KnowledgeValidator
        from athena.knowledge.reconciler import MemoryReconciler
        
        if self.memory_manager is not None:
            candidates = self.memory_manager.get_candidates()
            
            # Get semantic memory reference for conflict detection
            semantic_mem = self.memory_manager.semantic_memory
            
            validator = KnowledgeValidator(semantic_mem)
            
            # Track which conflicts correspond to which candidates
            conflict_to_candidate = {}
            
            for idx, candidate in enumerate(candidates):
                classification, conflict_id = validator.classify(
                    candidate.statement,
                    candidate.confidence,
                    candidate.category
                )
                
                if classification == 'duplicate':
                    # Duplicate: skip (do not update Semantic Memory)
                    pass
                elif classification == 'new_fact':
                    # New fact: promote to semantic memory
                    self.memory_manager.learn(candidate.statement, {
                        "type": "knowledge",
                        "confidence": candidate.confidence,
                        "category": candidate.category
                    })
                elif classification == 'possible_conflict':
                    # Possible conflict: reconcile via LLM
                    conflict_to_candidate[idx] = (candidate, conflict_id)
            
            # Reconcile all possible conflicts using the reconciler
            if self.provider is not None and conflict_to_candidate:
                reconciler = MemoryReconciler(self.provider)
                
                # Build list of conflict records for reconciliation
                conflicts = []
                for idx, (candidate, conflict_id) in conflict_to_candidate.items():
                    # Find the conflict record from validator.conflicts
                    for c in validator.get_conflicts():
                        if c['existing_id'] == conflict_id or c['candidate_statement'] == candidate.statement:
                            conflicts.append(c)
                            break
                
                # Reconcile all conflicts at once
                reconciliation_results = await reconciler.reconcile(conflicts, semantic_mem)
                
                # Store reconciliation results in thought metadata
                thought.metadata["reconciliation"] = {
                    "total_conflicts": len(conflicts),
                    "results": reconciliation_results
                }
            
            elif conflict_to_candidate:
                # No provider available - keep existing memory for all conflicts
                thought.metadata["reconciliation"] = {
                    "skipped": True,
                    "reason": "No LLM provider configured"
                }
            
            # Clear candidates after processing (they've been promoted or discarded)
            self.memory_manager.working_memory.clear()
            
            # Store detected conflicts in thought metadata for future reconciliation
            thought.metadata["conflicts"] = validator.get_conflicts()
        
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
