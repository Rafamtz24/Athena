"""
Athena Event Dispatcher

Routes published events to subscribed callbacks.
Keeps implementation minimal - simply iterates over registered subscribers
and invokes each callback with the event.
"""

from typing import Callable, Dict, List

from athena.events.models import Event


class Dispatcher:
    """
    Routes published events to subscribers based on event type and source.

    The dispatcher maintains two levels of routing:
        1. By event type (e.g., 'ThoughtCreated')
        2. By source (e.g., 'thought_pipeline')

    When an event is routed, all callbacks registered for that event's type
    are invoked synchronously in the calling thread.
    """

    def __init__(self) -> None:
        """Initialize the Dispatcher with empty routing registries."""
        # Maps event_type string -> list of (callback, source_filter) tuples
        self._routes: Dict[str, List[tuple]] = {}

    def subscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        """
        Register a callback to receive events of the given type.

        Args:
            event_type: The event type string to subscribe to.
            callback: A callable that accepts an Event instance.
        """
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must be a non-empty string")

        if not callable(callback):
            raise TypeError("callback must be callable")

        if event_type not in self._routes:
            self._routes[event_type] = []

        # Avoid duplicate subscriptions
        route_entry = (callback, None)  # No source filter by default
        if route_entry not in self._routes[event_type]:
            self._routes[event_type].append(route_entry)

    def unsubscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        """
        Remove a previously registered callback for the given event type.

        Args:
            event_type: The event type string to unsubscribe from.
            callback: The callback that was previously registered.
        """
        if event_type in self._routes:
            try:
                self._routes[event_type].remove((callback, None))
                # Clean up empty lists
                if not self._routes[event_type]:
                    del self._routes[event_type]
            except ValueError:
                pass

    def route(self, event: Event) -> None:
        """
        Route an event to all matching subscribers.

        Iterates over registered callbacks for the event's type and invokes
        each one with the event. Callbacks are invoked in registration order.

        Args:
            event: The Event instance to route.
        """
        if not isinstance(event, Event):
            raise TypeError("event must be an instance of Event")

        # Route by event type
        callbacks = self._routes.get(event.type, [])
        for callback, source_filter in callbacks:
            try:
                callback(event)
            except Exception:
                # Swallow exceptions to prevent one failing callback from breaking others
                pass

        # Also route by source if there are source-specific subscriptions
        if event.source and event.source != event.type:
            source_callbacks = self._routes.get(event.source, [])
            for callback, source_filter in source_callbacks:
                try:
                    callback(event)
                except Exception:
                    pass

    def get_route_count(self, event_type: str) -> int:
        """Return the number of subscribers for a given event type."""
        return len(self._routes.get(event_type, []))


# Module-level singleton dispatcher instance
_dispatcher = Dispatcher()


def get_dispatcher():
    """Get the global Dispatcher singleton instance."""
    return _dispatcher


def reset_dispatcher() -> None:
    """Reset the global dispatcher (useful for testing)."""
    global _dispatcher
    _dispatcher = Dispatcher()