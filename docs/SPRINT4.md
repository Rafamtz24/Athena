# SPRINT 4: Athena Event System Foundation

## Purpose

Sprint 4 introduces the internal EventBus system to decouple communication between all Athena subsystems. Prior to this sprint, modules had tight coupling through direct imports and method calls. The EventBus provides a publish-subscribe pattern where every subsystem can communicate without knowing about other specific modules.

### Why an Event Bus?

- **Decoupling**: Modules publish events without knowing who consumes them
- **Observability**: Any component can subscribe to events it cares about
- **Extensibility**: New subscribers can be added without modifying existing code
- **Testability**: Events provide clear hooks for unit testing each stage

## Architecture

### Package Structure

```
athena/events/
├── __init__.py    # Package initialization and exports
├── models.py      # Event dataclass definition
├── bus.py         # EventBus class (subscribe/unsubscribe/publish)
└── dispatcher.py  # Dispatcher class (routing events to subscribers)
```

### Components

#### Event Data Model (`models.py`)

The `Event` dataclass is the foundation of the event system. It contains:

| Field       | Type              | Description                              |
|-------------|---------------------|------------------------------------------|
| `id`        | `str`               | Unique identifier (auto-generated UUID)  |
| `timestamp` | `datetime`          | When the event occurred (UTC timezone)   |
| `type`      | `str`               | Event type/category string               |
| `source`    | `str`               | Which module/subsystem emitted this event|
| `payload`   | `Any`               | Data associated with the event           |
| `metadata`  | `Dict[str, Any]`    | Additional contextual information        |

The Event class is immutable (frozen dataclass) to prevent mutation after creation.

#### EventBus (`bus.py`)

The EventBus is a singleton that manages subscriber registries:

- **`subscribe(event_type, callback)`**: Register a callback for specific event types
- **`unsubscribe(event_type, callback)`**: Remove a previously registered callback
- **`publish(event)`**: Publish an event to all subscribers of its type
- **`get_subscriber_count(event_type)`**: Return number of subscribers

All operations are synchronous. No threading, async, queues, or background processing.

#### Dispatcher (`dispatcher.py`)

The Dispatcher routes published events to the correct subscribers:

- Routes by event type (primary routing)
- Routes by source (secondary routing when source differs from type)
- Swallows exceptions in callbacks so one failure doesn't break others
- Maintains registration order for predictable callback invocation

## Event Lifecycle

1. **Creation**: A module creates an `Event` instance with type, source, payload, and metadata
2. **Publishing**: The module calls `bus.publish(event)` on the EventBus singleton
3. **Dispatching**: The Dispatcher routes the event to all registered callbacks for that event type
4. **Handling**: Each callback receives the event and processes it (e.g., logging)
5. **Completion**: All callbacks complete synchronously in the calling thread

### Example Flow

```
ThoughtPipeline._initialize()
    → Creates Event(type="ThoughtCreated", source="thought_pipeline")
    → bus.publish(event)
        → Dispatcher routes to all subscribers of "ThoughtCreated"
            → Logger callback logs the event
            → Any other subscriber processes it
```

## Current Events

The following events are published by the ThoughtPipeline at each stage:

| Event Type           | Source              | Payload                              | Metadata Key     |
|----------------------|---------------------|--------------------------------------|------------------|
| `ThoughtCreated`     | `thought_pipeline`  | `{user_input}`                       | `created/initialized` |
| `MemoryLoaded`       | `thought_pipeline`  | `{user_input}`                       | `memory_loaded`  |
| `ReasoningStarted`   | `thought_pipeline`  | `{user_input}`                       | `reasoning`      |
| `PlanningStarted`    | `thought_pipeline`  | `{user_input}`                       | `planning`       |
| `ToolsPrepared`      | `thought_pipeline`  | `{user_input}`                       | `tools_prepared` |
| `ResponseGenerated`  | `thought_pipeline`  | `{user_input, response}`             | `response_generated` |
| `ReflectionStarted`  | `thought_pipeline`  | `{user_input}`                       | `reflection`     |
| `ThoughtCompleted`   | `thought_pipeline`  | `{user_input, response}`             | `completed`      |

All events include the original user input in the payload for traceability.

## Logger Integration

The logger module (`athena/logging/logger.py`) subscribes to all ThoughtPipeline events by default:

- `_event_callback()` receives every event
- Logs event type, source, and string representation of payload
- Provides observability into the system's internal event flow

## Future Expansion

### Planned Enhancements (Not Implemented)

1. **Async Support**: Add async callback support for non-blocking event handling
2. **Threading**: Background thread processing for expensive callbacks
3. **Event Filtering**: Allow subscribers to filter events by payload content or metadata criteria
4. **Event Sources**: Extend source-based routing for multi-brain architectures
5. **Persistence Layer**: Persist events to a database for audit trails
6. **Network Distribution**: Replace in-process EventBus with Redis/RabbitMQ/Kafka
7. **WebSocket Broadcasting**: Stream events to connected clients via WebSocket
8. **Plugin System**: Allow external modules to subscribe to specific event types

### Design Decisions

- **Synchronous only**: Keeps the implementation simple and predictable
- **No queues**: Events are delivered immediately in publish order
- **Exception swallowing**: One failing callback doesn't affect others
- **Singleton pattern**: Single EventBus instance for consistent routing
- **Immutable events**: Prevents mutation after creation

## Verification

To verify the event system:

1. Run `python -c "from athena.events import Event, EventBus; print('EventBus OK')"`
2. Start the API server and process a message through `/process?message=test`
3. Check logs for event publications from each pipeline stage

## Files Created

- `athena/events/__init__.py` - Package initialization
- `athena/events/models.py` - Event dataclass definition
- `athena/events/bus.py` - EventBus implementation
- `athena/events/dispatcher.py` - Dispatcher routing logic

## Files Modified

- `athena/thought/pipeline.py` - Added event publishing at each stage
- `athena/logging/logger.py` - Added EventBus subscription for logging all events

## Summary

Sprint 4 establishes the foundation for decoupled communication in Athena. The EventBus provides a clean publish-subscribe pattern that allows any module to observe system activity without direct dependencies. The ThoughtPipeline now publishes events at every stage, and the Logger receives and logs all events automatically.