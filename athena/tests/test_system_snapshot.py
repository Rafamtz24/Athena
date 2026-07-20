"""
Tests for the /system snapshot's assembly.

Not about the hardware values — those come from the machine and cannot be
asserted portably. These cover how the sections are gathered, which is where
the bugs live.

The snapshot is a dozen PowerShell and wmic processes, ~0.7s each to start,
and it ran them one after another: ~15s of every /system turn, before the
model saw a token. They are now gathered concurrently, and two things about
that need holding in place — the CPU reading must not measure Athena's own
load, and one broken section must not take down the other nine.
"""
import time

import pytest

import athena.tools.system_snapshot as snapshot


def _generate(**kwargs):
    """Call the real assembler.

    conftest stubs generate_system_snapshot for the whole session so no test
    shells out to read real hardware. These tests are about that assembler, so
    they reach past the stub — the gatherers underneath are stubbed per test
    instead, which keeps them fast without skipping the code under test.
    """
    return snapshot._real_generate_system_snapshot(**kwargs)


@pytest.fixture(autouse=True)
def fast_cpu_lookup(monkeypatch):
    """Keep cpuinfo out of it.

    cpuinfo.get_cpu_info() probes the processor and takes well over a second —
    the same kind of cost this whole change exists to remove, and none of
    these tests care what the CPU is called.
    """
    monkeypatch.setattr(
        snapshot.cpuinfo,
        "get_cpu_info",
        lambda: {"brand_raw": "Test CPU", "hz_advertised_friendly": "3.0000 GHz"},
    )


@pytest.fixture
def stub_gatherers(monkeypatch):
    """Replace every gatherer with a slow, predictable stand-in.

    Each sleeps, so a sequential implementation takes the sum and a
    concurrent one takes roughly the longest.
    """

    def _slow(name, seconds=0.2):
        def _gather(*args, **kwargs):
            time.sleep(seconds)
            return f"  {name} ok"

        return _gather

    for attribute in (
        "_get_os_info",
        "_get_cpu_info",
        "_get_ram_info",
        "_get_gpu_info",
        "_get_storage_info",
        "_get_display_info",
        "_get_motherboard_info",
        "_get_power_info",
        "_get_network_info",
    ):
        monkeypatch.setattr(snapshot, attribute, _slow(attribute))
    monkeypatch.setattr(snapshot, "_get_athena_runtime_info", _slow("runtime"))
    monkeypatch.setattr(snapshot, "_sample_cpu_utilization", lambda: 1.0)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_sections_are_gathered_concurrently(stub_gatherers):
    """Ten gatherers sleeping 0.2s each: ~2s in sequence, ~0.2s overlapped."""
    started = time.time()
    _generate()
    elapsed = time.time() - started

    assert elapsed < 1.0, f"sections look sequential ({elapsed:.2f}s)"


def test_section_order_is_stable(stub_gatherers):
    """Whichever section finishes first, the snapshot reads the same."""
    text = _generate()

    order = [line for line in text.splitlines() if line.startswith("-- ")]
    assert order == [
        "-- Operating System --",
        "-- CPU --",
        "-- RAM --",
        "-- GPU --",
        "-- Storage --",
        "-- Displays --",
        "-- Motherboard --",
        "-- Power --",
        "-- Network --",
        "-- Athena Runtime --",
    ]


def test_the_query_is_rendered_first(stub_gatherers):
    text = _generate(tool_prompt="how much RAM?")

    assert text.splitlines()[0] == "-- Query --"
    assert "how much RAM?" in text


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------

def test_one_failing_section_does_not_lose_the_others(stub_gatherers, monkeypatch):
    """Gatherers run off the main thread now, where an unexpected exception
    would otherwise surface only when its result is read — and take the whole
    snapshot with it."""

    def _explode():
        raise RuntimeError("wmic went missing")

    monkeypatch.setattr(snapshot, "_get_gpu_info", _explode)

    text = _generate()

    assert "Unavailable" in text
    assert "wmic went missing" in text
    # The other nine still reported.
    assert text.count(" ok") == 9


# ---------------------------------------------------------------------------
# CPU load must describe the machine, not the snapshot
# ---------------------------------------------------------------------------

def test_cpu_utilization_is_sampled_before_the_other_sections(monkeypatch):
    """Sampling runs for half a second. Doing it alongside the other sections
    measured Athena spawning a dozen processes — 50% on a machine sitting at
    1% — so the reading is taken first, while nothing else is running.
    """
    events = []

    def _sample():
        events.append("sampled")
        return 1.0

    def _gather(*args, **kwargs):
        events.append("gathered")
        return "  ok"

    monkeypatch.setattr(snapshot, "_sample_cpu_utilization", _sample)
    for attribute in (
        "_get_os_info",
        "_get_ram_info",
        "_get_gpu_info",
        "_get_storage_info",
        "_get_display_info",
        "_get_motherboard_info",
        "_get_power_info",
        "_get_network_info",
    ):
        monkeypatch.setattr(snapshot, attribute, _gather)
    monkeypatch.setattr(snapshot, "_get_athena_runtime_info", _gather)

    _generate()

    assert events[0] == "sampled", "load was sampled while other sections ran"


def test_cpu_section_uses_the_supplied_reading(monkeypatch):
    """The sampled value has to reach the output, or the early sample is
    pointless and the section silently takes its own."""
    monkeypatch.setattr(
        snapshot.psutil, "cpu_percent", lambda *a, **k: pytest.fail("resampled")
    )

    assert "Current utilization: 12.5%" in snapshot._get_cpu_info(12.5)


def test_cpu_section_samples_for_itself_when_called_alone(monkeypatch):
    monkeypatch.setattr(snapshot, "_sample_cpu_utilization", lambda: 7.5)

    assert "Current utilization: 7.5%" in snapshot._get_cpu_info()
