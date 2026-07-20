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
│   ├── semantic.py            # Semantic memory (durable facts)
│   ├── working.py             # Working memory (sliding window + candidate storage)
│   └── manager.py             # MemoryManager (coordinates all memory systems)
├── knowledge/
│   ├── models.py              # Knowledge data models
│   ├── manager.py             # KnowledgeManager (extraction, retrieval)
│   ├── validator.py           # KnowledgeValidator (quality gate)
│   ├── reconciler.py          # MemoryReconciler (write-time conflict resolution)
│   └── consolidator.py        # Consolidator (periodic cleanup of stored facts)
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
├── debug/
│   └── manager.py             # Debug utilities
└── tests/
    ├── test_*.py              # The pytest suite — every test lives here
    └── diagnose_*.py          # Manual scripts, NOT collected by pytest
```

Tests live in `athena/tests/` and nowhere else. They used to sit in both that
directory and beside the code they covered, which meant no single place
answered "is this covered?". Source packages now contain source only.

The two prefixes are a real distinction, not a style preference. `test_*` is
collected by pytest and must be safe to run unattended. `diagnose_*` is run by
hand and may be destructive — several load a real model, and
`diagnose_memory_reconciliation.py` clears the semantic memory store to set up
its scenarios. Naming one like the other is how someone loses their memory
file to a routine test run.

`conftest.py` at the project root points every persistent store at a temporary
directory for the duration of a run, so the collected suite cannot write to
real user data even when a test drives a full AthenaBrain.

## Core Architecture

### Brain (`athena/brain/brain.py`)

The Brain is the coordinator. It:
- Creates and manages `Thought` objects
- Owns shared managers (memory, knowledge)
- Resets Working Memory at the start of each session (see below), then persists the active window to `working_memory.json` during the session
- Loads and persists Chat History from `chat_history.json`
- Prunes Working Memory after each interaction via `WorkingMemory.prune()`
- Does NOT perform reasoning itself

### Thought (`athena/thought/models.py`)

A Thought is the temporary cognitive workspace for one interaction. It carries:
- `user_input` — The raw user input
- `history` — Working Memory / conversation history
- `knowledge` — Semantic Memory retrieval
- `planner_decision` / `planner_decisions` — Output of the Tool Planner (primary decision, and the full list when tools are chained)
- `tool_context` / `tool_contexts` — ToolContext(s) produced by the Tool Router (primary, and the full list when tools are chained)
- `reasoning_package` — ReasoningContextPackage (budgeted context for PromptBuilder)
- `learning_package` — LearningContextPackage (budgeted context for KnowledgeExtractor)
- `response` — The generated assistant response
- `candidates` — Knowledge candidates for learning
- `trace` — Debug trace information

### Thought Pipeline (`athena/thought/pipeline.py`)

The pipeline processes a single interaction through 14 stages:

1. `_initialize()` — Set metadata, publish ThoughtCreated
2. `_load_knowledge()` — Retrieve Semantic Memory
3. `_plan_tool()` — Tool Planner: decide which tool(s) are needed
4. `_execute_tool()` — Tool Router: execute the planned tool(s) if any
5. `_reason()` — Publish ReasoningStarted
6. `_plan()` — Publish PlanningStarted
7. `_prepare_tools()` — Verify tool context, publish ToolsPrepared
8. `_budget_context()` — **Context Budget Manager** (two-phase):
   - Phase 1: Compute Working Memory budget, call `WorkingMemory.prune()`
   - Phase 2: Compile Reasoning and Learning packages
9. `CognitiveEngine` — PromptBuilder renders ReasoningPackage → provider generates response
10. `_build_response()` — Publish ResponseGenerated
11. `_extract_candidates()` — KnowledgeExtractor consumes LearningContextPackage
12. `_validate_knowledge()` — Validate + reconcile candidates against Semantic Memory
13. `_reflect()` — Self-evaluate outcome
14. `_finalize()` — Publish ThoughtCompleted

There is no memory-loading stage. The conversation window is already on the Thought when the pipeline starts (`AthenaBrain` copies it in), and the episodic store that this stage read has been removed — it held verbatim transcripts of the same turns, which reached the prompt as a second copy.

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

Working Memory is **session-scoped**: it is the sliding window of the *current* conversation, not long-term storage. Each session starts with an empty window (`AthenaBrain._reset_working_memory()`), so stale turns from prior sessions cannot replay and override durable facts. Anything meant to persist across sessions is captured in Semantic Memory by the learning pipeline.

### Memory System (`athena/memory/`)

Two types of memory, accessed through `MemoryManager`:
- **Semantic**: Durable facts consulted during reasoning (`learn()` / `query_semantic()`)
- **Working**: Temporary context + candidate storage (`store()` / `retrieve()` / `clear()`)

An **Episodic** store sat alongside these, holding verbatim `User: …/Assistant: …` transcripts. It was removed: it was never persisted, so it could not serve long-term recall, and what it held was the conversation Working Memory already carries — so it reached the prompt as a second copy of every turn. Storing transcripts verbatim is a known dead end for agent memory; the durable part of an exchange is the fact it established, which is what Semantic Memory keeps.

### Knowledge System (`athena/knowledge/`)

Knowledge pipeline, ordered so each phase hands the next only what it could not settle deterministically:
1. **Extractor** (`KnowledgeManager.extract_candidates()`) — Consumes a `LearningContextPackage` and produces KnowledgeCandidates via an LLM call. Gated by a regex pre-check, so greetings and acknowledgements cost nothing.
2. **Validator** (`KnowledgeValidator`) — Deterministic quality gate. Rejects low-quality and duplicate facts, and applies the durability test: a fact must still be true next week, so "User ran a backup" is rejected while "User runs backups every Sunday" is kept.
3. **Reconciler** (`MemoryReconciler`) — Two passes. Single-valued attributes (name, location, OS) resolve by comparing parsed values with no model at all; everything left is classified in **one** batched LLM call per turn, not one per candidate. A candidate the batch fails to classify is retried alone rather than guessed at.
4. **Consolidator** (`consolidator.py`) — Runs once at startup over the whole store. Every other phase runs at write time and so binds only future facts; this is the pass that applies today's rules to what was stored yesterday, dropping entries the gates now reject, exact duplicates, and outdated values of single-valued attributes. Deterministic, so it never needs a model and cannot delete a good fact through a bad generation.

Without phase 4 a fact store only accumulates, which is the documented failure mode for agent memory: stale entries crowd the retrieval budget and contradictions build up silently.

### Tool System (`athena/tools/`)

- **Tool Planner** (`planner/planner.py`) — Decides which tool(s) are needed based on user input. `plan_tools()` returns one or more `PlannerDecision`s; most queries need at most one, but some chain several (e.g. a "can I run X" compatibility check returns `/web` for the software's requirements **and** `/system` for the user's hardware).
- **Tool Router** (`tools/router.py`) — The sole component that executes tools. `route_all()` executes each decision in order and produces one `ToolContext` per tool.
- **ToolContext** (`tools/models.py`) — Temporary context with metadata: `tool_name`, `content`, `priority`, `learning_visible`. When tools are chained, each produces its own reasoning source (`tool:<name>`); only the `/system` snapshot feeds the learning extractor's hardware-fact slot.
- Current tools: `/system` (System Snapshot), `/web` (Web Search via DuckDuckGo), `weather` (current conditions via wttr.in — a keyless weather source, so weather queries return real values instead of search snippets).
- **Query routing note:** thermal terms are disambiguated by context — "cpu temperature" is a system check, "temperature outside" is weather. Live/external topics (weather, prices, news) are never satisfied from memory, so a stored fact that merely overlaps the query (e.g. the user's city) never suppresses the lookup.

### Reading Mode (`athena/books/`)

A **separate, pipeline-free path** for answering questions grounded strictly in a local PDF. Entered from the console with `/book`.

- Books live in the `books/` directory (`storage.books_path`). `list_books()` discovers PDFs; the user selects one by number.
- The selected PDF is extracted (`pypdf`) and split into overlapping word-window chunks. Because a book far exceeds the context window, only the passages most relevant to each question (keyword-scored) are injected — retrieval, not whole-book injection.
- `AthenaBrain.answer_from_book()` bypasses the Thought pipeline entirely: **no tools, no memory retrieval/injection, no knowledge extraction**. It sends the retrieved passages with the strict `book` prompt profile ("answer only from these excerpts") directly to the provider.

### Tarot Mode (`athena/tarot/`)

Another **separate, pipeline-free path**, entered from the console with `/tarot`.

- Cards are drawn by a system RNG (OS entropy) from a pluggable deck **before** any interpretation, so the model cannot handpick them — it only reads the draw. Spreads (1–6 cards), reversals, and follow-up one-card pulls are handled in `athena/tarot/reading.py`.
- `AthenaBrain.tarot_reading()` bypasses the Thought pipeline: **no tools, no web search, no memory retrieval/injection, no knowledge extraction**. The drawn spread is sent with the `tarot` prompt profile directly to the provider.

### Serve Mode (`athena/serve/`)

A **separate, pipeline-free path** that exposes the resident reasoning model over an **OpenAI-compatible HTTP API** for external clients (e.g. Open WebUI). Entered from the console with `/serve [port]` (default `8080`); it blocks the terminal until `Ctrl+C`.

- `build_app()` wraps the **already-loaded** `llama_cpp.Llama` instance in a FastAPI app — no second copy of the model is loaded. It serves `GET /v1/models` and `POST /v1/chat/completions` (streaming and non-streaming). llama-cpp-python already emits OpenAI-shaped payloads, so the handlers are near pass-throughs.
- Because `llama_cpp.Llama` is **not thread-safe**, all generation is serialized behind a lock; concurrent requests wait their turn.
- Serve mode bypasses the Thought pipeline entirely: **no tools, no memory, no knowledge extraction** — it is pure model inference.
- **Memory management:** if a *dedicated* learning model is loaded, it is unloaded (`LlamaCppProvider.unload()`) to free its memory while only the reasoning model is served, then restored (`reload()`) when serving stops. When learning simply falls back to the reasoning model, there is nothing separate to unload.

### First-Run Onboarding (`athena/onboarding.py`) and Launcher (`Athena.bat`)

- **Launcher:** `Athena.bat` at the repo root starts Athena from any location — it `cd`s to its own folder (`%~dp0`), prefers a project `.venv`/`venv` Python when present, verifies Python is installed (with a friendly install pointer if not), and keeps the console open on errors so double-click users can read the message.
- **Onboarding:** before constructing the Brain, the terminal checks whether any `.gguf` exists under `models/reason`. If not, instead of crashing with a `FileNotFoundError`, it creates the model folders and prints a consumer-friendly guide: what a `.gguf` file is, how to download one from Hugging Face (search "<model> GGUF", pick the `Q4_K_M` file), and why an optional small model in `models/learning` makes replies feel faster.
- **Hardware-tailored recommendations:** the guide reuses `HardwareDetector` to read CPU/RAM/GPU (true VRAM via the registry on Windows) and recommends reasoning + learning model sizes for the machine — budgeting by VRAM when a GPU is present, else by ~half of system RAM, with headroom reserved for the context window. Detection failures degrade to generic size advice, never a crash.

### Providers (`athena/providers/`)

All providers implement `LLMProvider` (abstract base):
- `generate(message, system=None)` — Generate a response. The optional `system` prompt is delivered in the model's `system` role so it adopts Athena's identity rather than the base model's (which otherwise makes it identify as e.g. "Qwen").
- `health_check()` — Check availability
- `count_tokens(text)` — Count tokens using the provider's native tokenizer
- `get_context_window()` — Get the maximum context window size in tokens

**Reasoning traces and streaming.** Thinking models produce a chain-of-thought that never belongs in the answer, so providers always separate the two. The split happens two ways because the backends differ: the in-process provider parses `<think>…</think>` out of the completion itself, while llama-server parses it server-side and returns it in a separate `reasoning_content` field. Both feed `providers/reasoning_trace.py`, which the terminal drains for `/think show`.

The answer call streams token by token through `providers/streaming.py` — a module-level sink the terminal registers, so the layers between it and the provider (brain, pipeline, engine) do not have to relay output. Only `CognitiveEngine`'s call passes `stream=True`; the planner and learning calls run through the same providers silently. Providers advertise support with `supports_streaming`, so a provider without it is called exactly as before.

The sink also carries prompt-evaluation progress (`return_progress` on llama-server), which the terminal turns into a spinner label rather than output. That phase — the model reading the prompt before any token exists — was measured at 38s of a 45s turn on a 35B with experts on the CPU, and is the bulk of what used to look like a hang. Tool execution is named from the `ToolPlanned` event, so the terminal reports stages by subscribing to the existing event bus rather than by the pipeline calling into it.

Supported providers:
- **LlamaCppProvider** — Local GGUF model inference via `llama-cpp-python`. Uses actual tokenizer for `count_tokens()`.
- **LMStudioProvider** — HTTP API to LM Studio local server. Uses heuristic for `count_tokens()`.

### PromptBuilder (`athena/prompt/builder.py`)

A pure renderer. Receives a `ReasoningContextPackage` and renders it into a formatted prompt string. Does NOT perform budgeting.

### Event Bus (`athena/events/bus.py`)

Publish-subscribe event system for decoupled module communication. Each pipeline stage publishes a corresponding event.

`EventBus` is the public face; the `Dispatcher` holds the one and only subscription registry, and every bus method forwards to it. Subscribe through either — they are the same registry. (They were not always: the bus used to keep a second `_subscribers` dict that publishing never read, so bus subscribers were silently never called.)

Consumers: the terminal subscribes to `ToolPlanned` / `ToolExecuted` to name the phase the spinner is showing. `logging/logger.py` can mirror every event to the console via `subscribe_logger_to_bus()`, which is opt-in — it prints roughly eight lines per turn, which belongs in a debugging session and not in an answer.

## Data Flow

```
User → AthenaBrain.process(message)
              │
              ▼
        Create Thought
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

Authority order (high → low): the user's current message, then confirmed durable facts (Semantic Memory), then the replayed conversation (Working Memory). Semantic Memory ranks **above** Working Memory so a stored fact outranks a stale conversational turn, while User Input stays highest so corrections are always accepted (recency preserved).

| Priority | Source | Behavior |
|----------|--------|----------|
| 100 | User Input | Never trimmed |
| 95 | System Prompt | Never trimmed |
| 90 | Semantic Memory | Not trimmed in current milestone |
| 80 | Working Memory | Never trimmed (but pre-pruned to budget) |
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
