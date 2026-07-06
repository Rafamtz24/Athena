# Athena Architecture

## Overview

Athena implements a **cognitive pipeline architecture** where a `Thought` object flows through processing stages, each with a single responsibility. The Brain coordinates the pipeline without performing reasoning itself.

## Module Hierarchy

```
athena/
├── main.py                    # FastAPI entry point
├── brain/
│   └── brain.py               # Coordinates cognitive pipeline
├── thought/
│   ├── models.py              # Thought data class
│   └── pipeline.py            # Processing pipeline (15 stages)
├── cognition/
│   └── engine.py              # Cognitive processing engine
├── context/
│   ├── models.py              # ContextSource, ReasoningContextPackage, LearningContextPackage
│   └── manager.py             # Context Budget Manager
├── memory/
│   ├── episodic.py            # Episodic memory (past experiences)
│   ├── semantic.py            # Semantic memory (durable facts)
│   ├── working.py             # Working memory (sliding window + candidate storage)
│   └── manager.py             # MemoryManager (coordinates all memory systems)
├── knowledge/
│   ├── models.py              # Knowledge data models
│   ├── manager.py             # KnowledgeManager (extraction, retrieval)
│   ├── validator.py           # KnowledgeValidator (quality gate)
│   └── reconciler.py          # MemoryReconciler (LLM-based conflict resolution)
├── planner/
│   ├── models.py              # PlannerDecision data model
│   └── planner.py             # Tool Planner (decides if a tool is needed)
├── tools/
│   ├── models.py              # ToolContext data model
│   ├── router.py              # Tool Router (executes tools)
│   ├── system_snapshot.py     # /system tool implementation
│   └── web_search.py          # /web tool implementation
├── providers/
│   ├── base.py                # LLMProvider abstract base
│   ├── factory.py             # ProviderFactory (creates provider instances)
│   ├── llamacpp.py            # LlamaCpp implementation (local GGUF)
│   └── lmstudio.py            # LM Studio implementation (HTTP API)
├── prompt/
│   └── builder.py             # PromptBuilder (renders ReasoningContextPackage)
├── config/
│   ├── settings.py            # Configuration management
│   └── inference.py           # InferenceConfiguration + AutoConfigurator
├── hardware/
│   └── detector.py            # HardwareDetector (auto-detect CPU/GPU/RAM)
├── events/
│   ├── bus.py                 # Event bus for module communication
│   └── models.py              # Event data models
├── logging/
│   └── logger.py              # Structured logging
└── debug/
    └── manager.py             # Debug utilities
```

## Core Architecture

### Brain (`athena/brain/brain.py`)

The Brain is the coordinator. It:
- Creates and manages `Thought` objects
- Owns shared managers (memory, knowledge)
- Loads and persists Working Memory from `working_memory.json`
- Loads and persists Chat History from `chat_history.json`
- Prunes Working Memory after each interaction via `WorkingMemory.prune()`
- Does NOT perform reasoning itself

### Thought (`athena/thought/models.py`)

A Thought is the temporary cognitive workspace for one interaction. It carries:
- `user_input` — The raw user input
- `history` — Working Memory / conversation history
- `memories` — Episodic memories
- `knowledge` — Semantic Memory retrieval
- `planner_decision` — Output of the Tool Planner
- `tool_context` — ToolContext produced by the Tool Router
- `reasoning_package` — ReasoningContextPackage (budgeted context for PromptBuilder)
- `learning_package` — LearningContextPackage (budgeted context for KnowledgeExtractor)
- `response` — The generated assistant response
- `candidates` — Knowledge candidates for learning
- `trace` — Debug trace information

### Thought Pipeline (`athena/thought/pipeline.py`)

The pipeline processes a single interaction through 15 stages:

1. `_initialize()` — Set metadata, publish ThoughtCreated
2. `_load_memory()` — Load episodic memories
3. `_load_knowledge()` — Retrieve Semantic Memory
4. `_plan_tool()` — Tool Planner: decide if a tool is needed
5. `_execute_tool()` — Tool Router: execute tool if needed
6. `_reason()` — Publish ReasoningStarted
7. `_plan()` — Publish PlanningStarted
8. `_prepare_tools()` — Verify tool context, publish ToolsPrepared
9. `_budget_context()` — **Context Budget Manager** (two-phase):
   - Phase 1: Compute Working Memory budget, call `WorkingMemory.prune()`
   - Phase 2: Compile Reasoning and Learning packages
10. `CognitiveEngine` — PromptBuilder renders ReasoningPackage → provider generates response
11. `_build_response()` — Publish ResponseGenerated
12. `_extract_candidates()` — KnowledgeExtractor consumes LearningContextPackage
13. `_validate_knowledge()` — Validate + reconcile candidates against Semantic Memory
14. `_reflect()` — Self-evaluate outcome
15. `_finalize()` — Publish ThoughtCompleted

### Context Budget Manager (`athena/context/manager.py`)

The Context Budget Manager compiles all available context into packages that fit within the active provider's context window.

Inputs:
- User Input
- System Prompt
- Working Memory
- Semantic Memory
- Tool Context(s)
- Chat History
- Candidate Facts

Outputs:
- `ReasoningContextPackage` — Ordered, budgeted sources for PromptBuilder
- `LearningContextPackage` — Sources with `learning_visible=True` for KnowledgeExtractor

Responsibilities:
- Detect the active provider's context window via `get_context_window()`
- Reserve generation budget (25% of context window by default)
- Compute Working Memory budget via `compute_wm_budget()` before compilation
- Order sources by priority (100=User Input down to 60=Chat History)
- Trim lowest-priority truncatable sources when budget is exceeded
- NEVER rewrite, summarize, or paraphrase content
- NEVER check tool names (uses metadata: `priority`, `learning_visible`)

### Working Memory (`athena/memory/working.py`)

Two responsibilities:
1. **Knowledge candidate storage** — Stores temporary KnowledgeCandidate objects during the learning pipeline
2. **Conversation history pruning** — `prune(max_tokens, entries)` removes oldest entries to fit within a token budget

Working Memory is pruned twice per interaction:
1. **In-pipeline** — The Context Budget Manager computes the budget and calls `prune()` before compiling the final package
2. **Post-response** — `AthenaBrain._prune_to_budget()` delegates to `WorkingMemory.prune()` to persist the pruned history

### Memory System (`athena/memory/`)

Three types of memory, accessed through `MemoryManager`:
- **Episodic**: Past experiences and events (`remember()` / `recall()`)
- **Semantic**: Durable facts consulted during reasoning (`learn()` / `query_semantic()`)
- **Working**: Temporary context + candidate storage (`store()` / `retrieve()` / `clear()`)

### Knowledge System (`athena/knowledge/`)

Two-phase knowledge pipeline:
1. **Extractor** (`KnowledgeManager.extract_candidates()`) — Consumes a `LearningContextPackage` and produces KnowledgeCandidates via an LLM call
2. **Validator** (`KnowledgeValidator`) — Deterministic quality gate (rejects low-quality / duplicate facts)
3. **Reconciler** (`MemoryReconciler`) — LLM-based comparison against existing Semantic Memory (duplicate, conflict, or new)

### Tool System (`athena/tools/`)

- **Tool Planner** (`planner/planner.py`) — Decides if a tool is needed based on user input
- **Tool Router** (`tools/router.py`) — The sole component that executes tools. Produces a `ToolContext`
- **ToolContext** (`tools/models.py`) — Temporary context with metadata: `tool_name`, `content`, `priority`, `learning_visible`
- Current tools: `/system` (System Snapshot), `/web` (Web Search via DuckDuckGo)

### Providers (`athena/providers/`)

All providers implement `LLMProvider` (abstract base):
- `generate(message)` — Generate a response
- `health_check()` — Check availability
- `count_tokens(text)` — Count tokens using the provider's native tokenizer
- `get_context_window()` — Get the maximum context window size in tokens

Supported providers:
- **LlamaCppProvider** — Local GGUF model inference via `llama-cpp-python`. Uses actual tokenizer for `count_tokens()`.
- **LMStudioProvider** — HTTP API to LM Studio local server. Uses heuristic for `count_tokens()`.

### PromptBuilder (`athena/prompt/builder.py`)

A pure renderer. Receives a `ReasoningContextPackage` and renders it into a formatted prompt string. Does NOT perform budgeting.

### Event Bus (`athena/events/bus.py`)

Publish-subscribe event system for decoupled module communication. Each pipeline stage publishes a corresponding event.

## Data Flow

```
User → AthenaBrain.process(message)
              │
              ▼
        Create Thought
              │
              ▼
    _load_memory() → Episodic Memory
              │
              ▼
    _load_knowledge() → Semantic Memory
              │
              ▼
    _plan_tool() → Tool Planner
              │
              ▼
    _execute_tool() → Tool Router → ToolContext
              │
              ▼
    _budget_context() → Context Budget Manager
         ├── Phase 1: Compute WM budget → WorkingMemory.prune()
         └── Phase 2: Compile ReasoningPackage + LearningPackage
              │
              ▼
    CognitiveEngine → PromptBuilder → provider.generate()
              │
              ▼
    _extract_candidates() → KnowledgeManager (consumes LearningPackage)
              │
              ▼
    _validate_knowledge() → Validator → Reconciler → Semantic Memory
              │
              ▼
    Return response
```

## Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Coordination vs Reasoning | Brain coordinates, LLM reasons | Separation of concerns |
| Thought lifecycle | Single cycle per interaction | Clear boundaries |
| Memory access | Through MemoryManager | Single abstraction layer |
| Module communication | Events where practical | Decoupled architecture |
| Provider dependency | Abstracted via `LLMProvider` | Local-first, swappable |
| Tool decisions | Planner decides, Router executes | Single responsibility |
| Context budgeting | Dedicated Budget Manager | Provider-independence |
| Knowledge extraction | Consumes LearningContextPackage | Tool-unaware extractor |
| Working Memory pruning | `WorkingMemory.prune()` method | Self-managed sliding window |

## Priority System (Context Budgeting)

| Priority | Source | Behavior |
|----------|--------|----------|
| 100 | User Input | Never trimmed |
| 95 | System Prompt | Never trimmed |
| 90 | Working Memory | Never trimmed (but pre-pruned to budget) |
| 80 | Semantic Memory | Not trimmed in current milestone |
| 70 | Tool Context | May be trimmed or truncated |
| 60 | Chat History | May be trimmed or truncated |

## Architectural Invariants

1. Reasoning and learning are independent cognitive processes
2. Learning never changes the response currently being generated
3. Semantic Memory is the only long-term factual memory used during reasoning
4. Knowledge extraction operates on completed interactions
5. Knowledge validation is the only mechanism that may modify Semantic Memory
6. The Brain coordinates workers; Workers perform cognition
7. Every worker has exactly one responsibility
8. Core favors simplicity over optimization
9. The Context Budget Manager is the only component that budgets context
10. The Knowledge Extractor has no knowledge of individual tools
11. Working Memory manages its own sliding window via `prune()`

## See Also

- Engineering principles: [ENGINEERING.md](ENGINEERING.md)
- Core principles: [ATHENA_CORE_PRINCIPLES.md](ATHENA_CORE_PRINCIPLES.md)
- v1 specification: [athena/ATHENA_CORE_V1_SPECIFICATION.md](athena/ATHENA_CORE_V1_SPECIFICATION.md)
- Pipeline explanation: [pipelineexplanation.md](pipelineexplanation.md)
