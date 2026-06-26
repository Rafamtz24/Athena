# Athena Architecture Proposal

## 1. Design Goals

- **Modularity**: Every capability is pluggable and replaceable. Modules communicate through well-defined interfaces; no module has implicit knowledge of another's internals.
- **Extensibility**: New tools, plugins, skills, and backends can be added without modifying existing code (Open/Closed Principle).
- **Clarity**: The architecture must be self-documenting. A new engineer should understand the system's structure by reading this document alone.
- **Performance**: System monitoring, tool execution, and API responses must remain responsive even as capabilities grow. Async-first design where appropriate.
- **Portability**: Run on Windows (primary), with Linux/macOS compatibility. No OS-specific hard dependencies in core modules.
- **Observability**: Every request is traceable through structured logging, metrics, and optional distributed tracing.

## 2. Core Principles

| Principle | Description |
|-----------|-------------|
| **Dependency Inversion** | High-level modules depend on abstractions (interfaces), not concrete implementations. |
| **Single Responsibility** | Each module owns exactly one reason to change. |
| **Interface Segregation** | Contracts are narrow and focused; clients never depend on methods they don't use. |
| **Explicit over Implicit** | All dependencies, configuration sources, and data flows are declared explicitly. |
| **Plugin-First** | No feature is hardwired. Every capability is a plugin loaded at runtime. |
| **Stateless Core** | The API layer holds no mutable state; all state lives in dedicated managers/services. |
| **Fail-Fast** | Invalid configurations, missing dependencies, or bad tool definitions raise on startup, not mid-request. |

## 3. Proposed Module Hierarchy

```
athena/                          # Project root (Python package)
├── main.py                      # FastAPI application entry point
├── config/
│   ├── __init__.py
│   ├── loader.py                # Loads configuration files / env vars
│   └── schema.py                 # Pydantic models for validation
├── core/
│   ├── __init__.py
│   ├── module.py                 # Base Module class (abstract)
│   ├── registry.py              # Plugin/tool registry
│   ├── lifecycle.py              # Start / stop / health hooks
│   └── context.py                # Request-scoped execution context
├── api/
│   ├── __init__.py
│   ├── router.py                 # FastAPI router assembly
│   ├── middleware.py             # Auth, logging, error handling
│   └── responses.py              # Standardized response shapes
├── tools/
│   ├── __init__.py
│   ├── base.py                   # Tool protocol (interface)
│   ├── dispatcher.py             # Dispatches tool calls to registered implementations
│   └── [tool plugins...]         # e.g., system_info, file_ops, web
├── skills/
│   ├── __init__.py
│   ├── skill.py                  # Skill base class
│   └── [skill plugins...]        # e.g., coding, research, automation
├── memory/
│   ├── __init__.py
│   ├── short_term.py             # Session-based working memory
│   ├── long_term.py              # Persistent knowledge store
│   └── retrieval.py              # Semantic search / RAG layer
├── services/
│   ├── __init__.py
│   ├── execution_service.py      # Orchestrates tool+skill pipelines
│   ├── conversation_service.py   # Manages dialogue state
│   └── notification_service.py  # Alerts, events, webhooks
├── knowledge/
│   ├── __init__.py
│   ├── store.py                  # Knowledge graph / vector store interface
│   └── sync.py                   # Ingestion pipeline
├── logs/
│   ├── __init__.py
│   ├── logger.py                 # Structured logging setup
│   └── audit.py                  # Audit trail for tool usage
├── models/
│   ├── __init__.py
│   ├── llm.py                    # LLM provider abstraction
│   └── embedding.py              # Embedding model abstraction
└── infra/
    ├── __init__.py
    ├── storage.py                # Filesystem / cloud storage adapter
    ├── scheduler.py              # Background job scheduling
    └── telemetry.py              # Metrics, tracing, profiling
```

## 4. Responsibilities of Every Module

| Module | Responsibility |
|--------|---------------|
| `core.module.BaseModule` | Abstract base defining lifecycle hooks (`initialize`, `shutdown`, `health_check`). All plugins inherit from this. |
| `core.registry.PluginRegistry` | Central registry that discovers, validates, and exposes tools/skills at runtime. Supports hot-reload. |
| `config.loader` | Reads `.env`, YAML/JSON config files, and environment variables. Produces a frozen configuration object. |
| `config.schema` | Pydantic models that define expected fields, types, defaults, and validation rules for every config section. |
| `api.router` | Assembles FastAPI routers, mounts them on the app, handles route registration. |
| `api.middleware` | CORS, request ID injection, timing headers, structured error mapping. |
| `tools.base.Tool` | Interface: `name`, `description`, `parameters` (JSON Schema), `execute(context, params)`. |
| `tools.dispatcher.ToolDispatcher` | Resolves a tool name to its implementation, validates parameters against the JSON schema, and handles execution errors. |
| `skills.skill.Skill` | Interface: `name`, `steps() -> list[ToolCall]`, `result_from_tool_results()`. Skills orchestrate multiple tools. |
| `memory.short_term.ShortTermMemory` | Per-session working memory with TTL-based eviction. Stores recent tool outputs, conversation snippets. |
| `memory.long_term.LongTermMemory` | Persistent store (SQLite / SQLite-JSON functions). CRUD for knowledge entries. |
| `memory.retrieval.Retriever` | Embeds queries, searches long-term memory, returns ranked results. Pluggable embedding backend. |
| `services.execution_service.ExecutionService` | Orchestrates a tool call or skill execution: context setup → dispatch → result collection → post-processing. |
| `services.conversation_service.ConversationService` | Manages conversation history, session lifecycle, and dialogue state persistence. |
| `knowledge.store.KnowledgeStore` | Abstraction over vector/graph storage. Supports ingestion, query, and deletion of knowledge units. |
| `logs.logger` | Configures Python's logging with structured JSON output, log levels per module, and file rotation. |
| `models.llm.LLMProvider` | Interface: `complete(messages) -> str`. Concrete implementations for OpenAI, local models, Azure, etc. |
| `infra.storage.StorageAdapter` | Abstracts filesystem operations, cloud blob storage, and temporary file handling. |

## 5. Data Flow

```
┌──────────┐     ┌─────────────┐     ┌──────────────┐
│ Client    │     │ FastAPI     │     │ Tool/Skill    │
│ (HTTP/WS) │◄───►│ Router      │◄───►│ Dispatcher    │
└──────────┘     └─────────────┘     └──────────────┘
                          │
               ┌──────────▼──────────┐
               │   ExecutionService   │
               │  (context + dispatch) │
               └──────────┬──────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   ┌──────────┐   ┌────────────┐   ┌────────────┐
   │ Tool A    │   │ Skill B    │   │ Tool C     │
   │ (system)  │   │ (multi-    │   │ (web)      │
   └──────────┘   │  step)      │   └────────────┘
                  └────────────┘
                          │
               ┌──────────▼──────────┐
               │   Memory / Knowledge │
               │   (read + write)     │
               └─────────────────────┘
```

1. **Request arrives** at the FastAPI router. Middleware injects a request ID and timing headers.
2. Router dispatches to an endpoint handler which creates an `ExecutionContext` (session ID, user identity, temp directory).
3. `ExecutionService` resolves the requested tool/skill name via the `PluginRegistry`.
4. Parameters are validated against the tool's JSON Schema.
5. Tool/Skill executes. It may read/write to short-term memory, query long-term memory, or call external services.
6. Results flow back through the execution service → API response layer → client.
7. Audit log entry is written asynchronously.

## 6. Startup Sequence

```
1. Python process starts → athena/main.py
2. FastAPI app created; lifespan context entered.
3. config.loader reads .env / config files → validates against config.schema models.
4. core.registry.PluginRegistry scans configured directories for tool/skill plugins.
   - Each plugin is instantiated, validated (schema check), and registered.
5. services.execution_service.ExecutionService initializes dispatchers.
6. memory modules initialize persistent stores (e.g., create SQLite tables).
7. knowledge.store.KnowledgeStore connects to vector/graph backend.
8. logs.logger configures structured output with rotation.
9. Background scheduler starts (infra.scheduler).
10. Application is ready; FastAPI lifespan yields control.

Shutdown:
- Signal received or lifespan exits.
- core.lifecycle.shutdown() called → calls shutdown() on every registered module.
- Database connections closed, temp files cleaned, audit logs flushed.
```

Each module's `initialize()` and `shutdown()` hooks are invoked by the registry, ensuring orderly startup and teardown.

## 7. Plugin / Tool Architecture

### Protocol

Every tool implements:

```python
class Tool(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def parameters(self) -> dict: ...   # JSON Schema draft-07

    async def execute(self, ctx: ExecutionContext, params: dict) -> ToolResult: ...
```

### Discovery

- Plugins live in `athena/tools/`, `athena/skills/`, or external directories specified by the `plugin_dirs` config key.
- On startup, the registry scans these directories for classes that inherit from `Tool` / `Skill`.
- Each plugin declares its schema; the registry validates and stores it.
- Hot-reload: watching configured directories for file changes and re-registering changed plugins.

### Tool Categories

| Category | Purpose | Examples |
|----------|---------|----------|
| **System** | OS-level operations | `system_info`, `file_ops`, `process_manager` |
| **Web** | HTTP, scraping, APIs | `web_fetch`, `api_client`, `scraper` |
| **Knowledge** | Store/retrieve facts | `knowledge_write`, `knowledge_query`, `knowledge_delete` |
| **Automation** | Multi-step workflows | `coding_assistant`, `research_agent`, `data_pipeline` |
| **Notification** | Alerts and delivery | `email_send`, `webhook_dispatch`, `desktop_notify` |

### Skill Composition

Skills are higher-level abstractions that compose multiple tools into a workflow:

```python
class CodingSkill(Skill):
    name = "coding"
    def steps(self) -> list[ToolCall]:
        return [
            ToolCall("file_ops", {"action": "read", "path": "..."}),
            ToolCall("system_info", {"action": "analyze"}),
        ]
```

## 8. Memory Architecture

### Short-Term (Working) Memory

- In-memory, per-session dictionary with TTL eviction.
- Stores: last N tool outputs, conversation context window, temporary variables.
- Eviction policy: LRU when capacity exceeded.
- Access via `ShortTermMemory` service (thread-safe).

### Long-Term Memory

- SQLite database with JSON extension for semi-structured storage.
- Tables: `knowledge_entries`, `tags`, `session_history`.
- CRUD operations exposed by `LongTermMemory` service.
- Backed by full-text search (FTS5) and vector similarity (via `sqlite-vss` or custom).

### Retrieval Layer

- Embedding model (pluggable) converts text queries to vectors.
- Vector store returns top-K similar entries ranked by cosine similarity.
- Results are post-filtered by tags, date range, or relevance threshold.
- Cache layer: recent query results cached in short-term memory for performance.

### Data Lifecycle

```
Write → Embed → Store (long-term) → Cache (short-term)
Read  → Embed → Search (vector) → Filter → Rank → Return
```

## 9. Future Expansion Strategy

| Phase | Scope | Description |
|-------|-------|-------------|
| **Phase 0** | Current | Single-process FastAPI, local SQLite, local tools. |
| **Phase 1** | Plugin ecosystem | External plugin directories, hot-reload, versioned API contracts. |
| **Phase 2** | Multi-LLM support | Abstraction over OpenAI, Anthropic, local models, Azure AI. |
| **Phase 3** | Distributed execution | Worker pool for heavy tools; Redis-backed task queue (Celery/RQ). |
| **Phase 4** | Multi-user & auth | JWT authentication, per-user plugin namespaces, RBAC. |
| **Phase 5** | Web UI + WebSocket | Real-time streaming of tool output, live conversation interface. |
| **Phase 6** | Cloud storage | S3/GCS adapter for large file handling, object storage for knowledge base. |

Each phase adds new modules without modifying existing ones (Open/Closed Principle). Interfaces are stabilized before Phase 1+ to avoid breaking changes.

## 10. Risks and Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Over-engineering** | New architecture slows initial development. | Keep the current minimal surface area; add modules only when a need is clear. Gate new abstractions behind interfaces first. |
| **SQLite bottleneck at scale** | Concurrency limits under heavy write load. | Switch to PostgreSQL or a dedicated vector DB when read/write volume exceeds thresholds. Use WAL mode and connection pooling in the interim. |
| **Plugin discovery overhead** | Slow startup if many plugins exist. | Lazy loading: discover on first use, not at boot. Cache discovered registry. Provide an explicit plugin manifest file. |
| **Memory leaks in long sessions** | Short-term memory grows unbounded. | Enforce strict TTL and LRU eviction. Periodic health checks that trigger cleanup. Configurable max session size. |
| **Tool execution failures** | A broken tool crashes the request thread. | Wrap all tool calls in try/except; return structured `ToolResult` with error info. Timeouts per tool. Circuit breaker pattern for external services. |
| **Tight coupling to FastAPI** | Hard to swap or run without HTTP layer. | Keep `core.registry`, `tools.base`, and `skills.skill` framework-agnostic. FastAPI is just one consumer of the registry. |
| **Embedding model lock-in** | Vendor-specific embedding APIs increase cost. | Abstract embeddings behind `EmbeddingProvider` interface. Ship a local fallback (e.g., sentence-transformers) alongside cloud providers. |
| **Startup complexity** | More modules = more failure points. | Each module's `initialize()` returns a status. Fail-fast on invalid config. Provide a `--dry-run` mode that validates without starting the server. |

### Trade-off Summary

| Decision | Chosen Approach | Rationale |
|----------|----------------|-----------|
| Sync vs Async | Async-first for I/O, sync for CPU-bound | Keeps the API responsive; tools can run blocking code in a threadpool. |
| Monolith vs Microservices | Modular monolith first | Simpler deployment; split into services only when module boundaries show natural separation. |
| SQLite vs Dedicated DB | SQLite with FTS + vector extension | Zero external dependencies, sufficient for current scale, migratable later. |
| Config file vs Env vars | Both (file overrides env) | Flexibility for complex configs; env vars for secrets and deployment-time values. |
| Explicit plugin manifest vs Auto-discovery | Auto-scan with optional manifest | Convenience of auto-discovery + performance of explicit list. Manifest is the source of truth when present. |

---

*This document is a living proposal. Update it as the architecture evolves.*