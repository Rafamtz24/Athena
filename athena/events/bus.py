"""
Athena Event Bus

Provides a simple synchronous event bus for publishing events
and subscribing/unsubscribing to receive events of specific types.
"""

from typing import Any, Callable, Dict, List

from athena.events.models import Event


class EventBus:
    """
    A simple synchronous event bus that routes published events to subscribers.

    The EventBus supports:
        - subscribe(event_type, callback): Register a callback for a specific event type.
        - unsubscribe(event_type, callback): Remove a previously registered callback.
        - publish(event): Publish an event to all subscribed callbacks.

    All operations are synchronous and run in the calling thread.
    No threading, async, queues, or background processing is used.
    """

    def __init__(self) -> None:
        """Initialize the EventBus with empty subscriber registries."""
        # Maps event_type string -> list of callbacks
        self._subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        """
        Register a callback to receive events of the given type.

        Args:
            event_type: The event type string to subscribe to (e.g., 'ThoughtCreated').
            callback: A callable that accepts an Event instance as its only argument.
        """
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must be a non-empty string")

        if not callable(callback):
            raise TypeError("callback must be callable")

        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        # Avoid duplicate subscriptions
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        """
        Remove a previously registered callback for the given event type.

        Args:
            event_type: The event type string to unsubscribe from.
            callback: The callback that was previously registered.
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(callback)
                # Clean up empty lists
                if not self._subscribers[event_type]:
                    del self._subscribers[event_type]
            except ValueError:
                pass  # Callback was not subscribed

    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers of the event's type.

        The event's type and source are used to determine which callbacks
        should receive this event. If no explicit source is set on the event,
        the event_type is also used as the source for routing.

        Args:
            event: The Event instance to publish.
        """
        if not isinstance(event, Event):
            raise TypeError("event must be an instance of Event")

        # Notify the dispatcher to route this event
        from athena.events.dispatcher import get_dispatcher

        get_dispatcher().route(event)

    def get_subscriber_count(self, event_type: str) -> int:
        """Return the number of subscribers for a given event type."""
        return len(self._subscribers.get(event_type, []))


# Module-level singleton instance
_event_bus = EventBus()


def get_event_bus() -> EventBus:
    """Get the global EventBus singleton instance."""
    return _event_bus


def reset_event_bus() -> None:
    """Reset the global event bus (useful for testing)."""
    global _event_bus
    _event_bus = EventBus()