"""
Athena Event Bus

Provides a simple synchronous event bus for publishing events
and subscribing/unsubscribing to receive events of specific types.

The bus owns no registry of its own. Every method here forwards to the
Dispatcher, which is the single place a subscription can live.

That indirection is deliberate, and it was previously a bug: the bus kept its
own ``_subscribers`` dict while ``publish()`` routed through the Dispatcher's
separate registry, so a callback registered on the bus was never called. It
failed silently — subscribing appeared to work, and the events simply never
arrived. Two registries with one reader is not a design worth preserving, so
the bus keeps none.
"""

from typing import Callable

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

    Subscribing through the Dispatcher directly is equivalent; this class is
    the more convenient face of it, since publishers already hold a bus.
    """

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

        from athena.events.dispatcher import get_dispatcher

        get_dispatcher().subscribe(event_type, callback)

    def unsubscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        """
        Remove a previously registered callback for the given event type.

        Args:
            event_type: The event type string to unsubscribe from.
            callback: The callback that was previously registered.
        """
        from athena.events.dispatcher import get_dispatcher

        get_dispatcher().unsubscribe(event_type, callback)

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
        from athena.events.dispatcher import get_dispatcher

        return get_dispatcher().get_route_count(event_type)


# Module-level singleton instance
_event_bus = EventBus()


def get_event_bus() -> EventBus:
    """Get the global EventBus singleton instance."""
    return _event_bus


def reset_event_bus() -> None:
    """Drop every subscription (useful for testing).

    Replacing the bus alone would clear nothing, since the bus holds no
    state — the subscriptions live in the Dispatcher, so that is what has to
    be reset for this to mean anything.
    """
    global _event_bus

    from athena.events.dispatcher import reset_dispatcher

    reset_dispatcher()
    _event_bus = EventBus()