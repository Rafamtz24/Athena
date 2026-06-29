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
│   └── pipeline.py            # Processing pipeline (single responsibility)
├── memory/
│   ├── episodic.py            # Episodic memory (past experiences)
│   ├── semantic.py            # Semantic memory (durable facts)
│   └── working.py             # Working memory (current context)
├── knowledge/
│   ├── models.py              # Knowledge data models
│   └── manager.py             # Extraction and validation
├── cognition/
│   └── engine.py              # Cognitive processing engine
├── providers/
│   ├── base.py                 # LLM provider abstraction
│   └── lmstudio.py            # LM Studio implementation
├── events/
│   ├── bus.py                  # Event bus for module communication
│   └── models.py               # Event data models
├── prompt/
│   └── builder.py              # Prompt construction
├── config/
│   └── settings.py             # Configuration management
└── debug/
    └── manager.py               # Debug utilities
```

## Core Architecture

### Brain (`athena/brain/brain.py`)

The Brain is the coordinator. It:
- Creates and manages `Thought` objects
- Owns shared managers (memory, knowledge)
- Orchestrates the cognitive pipeline
- Does NOT perform reasoning itself

### Thought Pipeline (`athena/thought/pipeline.py`)

The pipeline processes a single interaction through stages:
1. Initialize thought with user input
2. Load conversation history from memory
3. Retrieve semantic memory
4. Generate response via LLM (Response Reasoner)
5. Extract knowledge candidates
6. Validate and promote to semantic memory
7. Finalize the thought

### Memory System (`athena/memory/`)

Three types of memory, each with a single responsibility:
- **Episodic**: Past experiences and events
- **Semantic**: Durable facts consulted during reasoning
- **Working**: Temporary context for current interaction

Memory is accessed through `MemoryManager` which coordinates access.

### Knowledge System (`athena/knowledge/`)

Two-stage knowledge pipeline:
1. **Extractor**: Produces knowledge candidates from completed interactions
2. **Validator**: Determines which candidates become permanent semantic memory

Knowledge extraction never modifies semantic memory directly.

### Event Bus (`athena/events/bus.py`)

Modules communicate through a publish-subscribe event system for decoupled communication.

## Data Flow

```
User Input → Brain.process()
              ↓
         Create Thought
              ↓
    Load Conversation History (Memory)
              ↓
      Retrieve Semantic Memory
              ↓
     Response Reasoner (LLM Provider)
              ↓
       Generate User Response
              ↓
   Build Completed Interaction
              ↓
    Knowledge Extraction (LLM)
              ↓
  Knowledge Validation
              ↓
    Update Semantic Memory
```

## Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Coordination vs Reasoning | Brain coordinates, LLM reasons | Separation of concerns |
| Thought lifecycle | Single cycle per interaction | Clear boundaries |
| Memory access | Through MemoryManager | Single abstraction layer |
| Module communication | Events where practical | Decoupled architecture |
| Provider dependency | Abstracted via providers | Local-first, swappable |

## Extension Points

Future capabilities can be added without modifying Core:
- Cognitive Planner
- Tool Execution
- Vision processing
- Memory Reconciliation
- Multi-provider support
- Embeddings
- Plugins

Extensions build upon the Core rather than replacing it.

## Architectural Invariants

1. Reasoning and learning are independent cognitive processes
2. Learning never changes the response currently being generated
3. Semantic memory is the only long-term factual memory used during reasoning
4. Knowledge extraction operates on completed interactions
5. Knowledge validation is the only mechanism that may modify semantic memory
6. The Brain coordinates workers; Workers perform cognition
7. Every worker has exactly one responsibility
8. Core favors simplicity over optimization

## See Also

- Detailed architecture proposal: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Engineering principles: [ENGINEERING.md](ENGINEERING.md)
- Core principles: [ATHENA_CORE_PRINCIPLES.md](ATHENA_CORE_PRINCIPLES.md)
- v1 specification: [athena/ATHENA_CORE_V1_SPECIFICATION.md](athena/ATHENA_CORE_V1_SPECIFICATION.md)
