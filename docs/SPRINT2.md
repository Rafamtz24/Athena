# SPRINT 2: Memory Foundation

## Background

Sprint 1 established `AthenaBrain` and the core architecture. Sprint 2 introduces the architectural foundation for Athena's memory system. This sprint is NOT about persistence, embeddings, databases, or retrieval. It is about defining clean abstractions.

## Memory Architecture

### Package Structure

```
athena/memory/
├── __init__.py      # Package initialization and exports
├── models.py        # MemoryEntry data model
├── working.py       # WorkingMemory implementation
├── episodic.py      # EpisodicMemory implementation
├── semantic.py      # SemanticMemory implementation
└── manager.py       # MemoryManager coordinator
```

### Components

#### MemoryEntry (models.py)
The atomic unit of memory storage. Every piece of memory is stored as a `MemoryEntry`.

**Fields:**
- `id`: Unique identifier (UUID)
- `timestamp`: Creation time (UTC)
- `content`: The actual data/payload
- `metadata`: Additional contextual information

#### WorkingMemory (working.py)
Short-term session memory for temporary storage during a single interaction cycle.

**Methods:**
- `store(content, metadata)` → `str` — Add data to working memory
- `retrieve()` → `list[MemoryEntry]` — Get all data from working memory
- `clear()` — Remove all data from working memory

#### EpisodicMemory (episodic.py)
Stores experiences (past events) for recall later.

**Methods:**
- `remember(content, metadata)` → `str` — Store an experience
- `recall()` → `list[MemoryEntry]` — Get all stored experiences

#### SemanticMemory (semantic.py)
Stores factual knowledge that can be queried later.

**Methods:**
- `learn(content, metadata)` → `str` — Store factual data
- `query()` → `list[MemoryEntry]` — Get all stored factual data

#### MemoryManager (manager.py)
Central coordinator for all memory systems. Provides single public interface for AthenaBrain to interact with memory.

**Methods:**
- `store_working(content, metadata)` — Store in working memory
- `get_working()` — Retrieve from working memory
- `clear_working()` — Clear working memory
- `remember(content, metadata)` — Store in episodic memory
- `recall()` — Retrieve from episodic memory
- `learn(content, metadata)` — Store in semantic memory
- `query_semantic()` — Retrieve from semantic memory

## Responsibilities

| Component | Responsibility |
|-----------|----------------|
| `MemoryEntry` | Data model for a single memory unit |
| `WorkingMemory` | Temporary session storage |
| `EpisodicMemory` | Store past experiences |
| `SemanticMemory` | Store factual knowledge |
| `MemoryManager` | Coordinate all memory systems |
| `AthenaBrain` | Initialize and use MemoryManager only |

### Communication Rules
- AthenaBrain communicates ONLY with MemoryManager
- Memory systems never communicate directly with AthenaBrain
- All memory operations go through MemoryManager public interface

## Future Roadmap

1. **Persistence Layer** — SQLite/JSON storage for memory entries
2. **Vector Embeddings** — Semantic search capabilities
3. **Retrieval Ranking** — Prioritize relevant memories
4. **Memory Consolidation** — Move working → episodic/semantic automatically
5. **Memory Decay** — Remove stale entries based on age/frequency
6. **Context Window Management** — Limit memory size per session

## What Was NOT Implemented

- Persistence (SQLite, JSON storage)
- Vector databases
- Embeddings
- RAG (Retrieval-Augmented Generation)
- LLM integration
- Retrieval ranking
- Plugins
- Tool usage

## Verification Results

- ✓ Application starts successfully
- ✓ Existing endpoints continue working
- ✓ AthenaBrain owns MemoryManager instance
- ✓ Memory package exists at `athena/memory/`
- ✓ Documentation completed at `docs/SPRINT2.md`

## Files Created

1. `athena/memory/__init__.py` — Package initialization
2. `athena/memory/models.py` — MemoryEntry model
3. `athena/memory/working.py` — WorkingMemory class
4. `athena/memory/episodic.py` — EpisodicMemory class
5. `athena/memory/semantic.py` — SemanticMemory class
6. `athena/memory/manager.py` — MemoryManager coordinator
7. `docs/SPRINT2.md` — This documentation

## Files Modified

1. `athena/brain/brain.py` — Added MemoryManager initialization in `__init__`