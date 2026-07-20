"""
Tests for the EventBus / Dispatcher pair.

The bug these exist to prevent: the bus used to keep its own `_subscribers`
dict while `publish()` routed through the Dispatcher's separate registry. A
callback registered on the bus was therefore never invoked — and the failure
was silent, because subscribing succeeded and the events simply never arrived.

Anything asserting only on a subscriber *count* would have passed throughout.
So the assertions here are on delivery: a subscriber must actually be called.
"""
import pytest

from athena.events.bus import EventBus, get_event_bus, reset_event_bus
from athena.events.dispatcher import get_dispatcher
from athena.events.models import Event


@pytest.fixture(autouse=True)
def _clean_registry():
    """Subscriptions are global; keep each test from leaking into the next."""
    reset_event_bus()
    yield
    reset_event_bus()


def _event(event_type="ThoughtCreated", **payload):
    return Event(
        type=event_type,
        source="thought_pipeline",
        payload=payload,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------

def test_a_bus_subscriber_actually_receives_events():
    received = []
    bus = get_event_bus()
    bus.subscribe("ThoughtCreated", received.append)

    bus.publish(_event(user_input="hello"))

    assert len(received) == 1
    assert received[0].payload["user_input"] == "hello"


def test_subscribing_on_either_face_reaches_the_same_registry():
    """The bus forwards to the Dispatcher, so the two are interchangeable."""
    received = []
    get_dispatcher().subscribe("ThoughtCreated", received.append)

    get_event_bus().publish(_event())

    assert len(received) == 1


def test_unsubscribe_stops_delivery():
    received = []
    bus = get_event_bus()
    bus.subscribe("ThoughtCreated", received.append)
    bus.unsubscribe("ThoughtCreated", received.append)

    bus.publish(_event())

    assert received == []


def test_only_the_subscribed_type_is_delivered():
    received = []
    bus = get_event_bus()
    bus.subscribe("ThoughtCreated", received.append)

    bus.publish(_event("ThoughtCompleted"))

    assert received == []


def test_a_failing_subscriber_does_not_stop_the_others():
    received = []

    def _explode(event):
        raise RuntimeError("subscriber is broken")

    bus = get_event_bus()
    bus.subscribe("ThoughtCreated", _explode)
    bus.subscribe("ThoughtCreated", received.append)

    bus.publish(_event())

    assert len(received) == 1


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------

def test_subscriber_count_reports_the_registry_that_is_read():
    """A count taken from a registry nothing publishes to is worse than no
    count at all — it reads as proof the subscription works."""
    bus = get_event_bus()
    assert bus.get_subscriber_count("ThoughtCreated") == 0

    bus.subscribe("ThoughtCreated", lambda event: None)

    assert bus.get_subscriber_count("ThoughtCreated") == 1
    assert get_dispatcher().get_route_count("ThoughtCreated") == 1


def test_the_same_callback_subscribes_once():
    received = []
    bus = get_event_bus()
    bus.subscribe("ThoughtCreated", received.append)
    bus.subscribe("ThoughtCreated", received.append)

    bus.publish(_event())

    assert len(received) == 1


def test_reset_clears_subscriptions():
    bus = get_event_bus()
    bus.subscribe("ThoughtCreated", lambda event: None)

    reset_event_bus()

    # Replacing the bus alone would clear nothing — it holds no registry.
    assert get_event_bus().get_subscriber_count("ThoughtCreated") == 0
    assert get_dispatcher().get_route_count("ThoughtCreated") == 0


def test_the_bus_holds_no_registry_of_its_own():
    """Two registries with one reader is what caused the original bug."""
    assert not hasattr(EventBus(), "_subscribers")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_invalid_subscriptions_are_rejected():
    bus = get_event_bus()

    with pytest.raises(ValueError):
        bus.subscribe("", lambda event: None)
    with pytest.raises(TypeError):
        bus.subscribe("ThoughtCreated", "not callable")


def test_publishing_a_non_event_is_rejected():
    with pytest.raises(TypeError):
        get_event_bus().publish({"type": "ThoughtCreated"})


# ---------------------------------------------------------------------------
# The logger subscription stays opt-in
# ---------------------------------------------------------------------------

def test_importing_the_logger_does_not_subscribe_it():
    """Enabling it puts ~8 INFO lines per turn into the console, mid-answer.
    It never ran while the bus was broken, so switching it on as a side effect
    of fixing the bus would be a regression, not a fix."""
    import athena.logging.logger  # noqa: F401

    assert get_event_bus().get_subscriber_count("ThoughtCreated") == 0
