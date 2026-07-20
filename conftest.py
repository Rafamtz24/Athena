"""Shared pytest configuration for the Athena test suite.

The critical job here is storage isolation. Athena's memory is plain JSON in
data/, and several tests drive a real AthenaBrain end to end — which runs the
post-answer learning phase and writes whatever it extracted. Without isolation
that lands in the developer's actual memory file: running the suite once
appended thirty-odd junk entries, including MagicMock repr strings from the
mocked-provider tests and a stray "My name is Qwen, created by Alibaba Cloud"
that directly contradicts Athena's identity rules.

The `isolate_storage` fixture is autouse and session-scoped, so every test in
the suite writes to a temporary directory instead. It relies on the storage
paths being read from settings at call time rather than captured at import —
see athena/memory/semantic.py and athena/brain/brain.py.

`stub_system_snapshot` is the same idea applied to the machine rather than the
disk: it keeps tests from shelling out to PowerShell to read real hardware.
"""

import pytest

from athena.config.settings import get_settings

# A snapshot shaped like the real one, minus the fifteen seconds.
_STUB_SNAPSHOT = """=== SYSTEM SNAPSHOT ===
OS: Test OS 1.0
CPU: Test CPU, 8 cores / 16 threads, 5.0% utilization
RAM: 32.0 GB total, 16.0 GB available
GPU: Test GPU, 8 GB VRAM
Storage: 500.0 GB total, 250.0 GB free
Network: connected
"""


@pytest.fixture(scope="session", autouse=True)
def isolate_storage(tmp_path_factory):
    """Point every persistent store at a temporary directory for the session.

    Autouse so no test has to remember to ask for it — the failure mode of
    forgetting is silent corruption of real user data, which is exactly the
    kind of thing that should not be opt-in.
    """
    data_dir = tmp_path_factory.mktemp("athena-data")
    storage = get_settings().storage

    original = {
        "working_memory_path": storage.working_memory_path,
        "chat_history_path": storage.chat_history_path,
        "semantic_memory_path": storage.semantic_memory_path,
        "user_prefs_path": storage.user_prefs_path,
    }

    storage.working_memory_path = str(data_dir / "working_memory.json")
    storage.chat_history_path = str(data_dir / "chat_history.json")
    storage.semantic_memory_path = str(data_dir / "semantic_memory.json")
    storage.user_prefs_path = str(data_dir / "user_prefs.json")

    yield data_dir

    for attribute, value in original.items():
        setattr(storage, attribute, value)


@pytest.fixture(scope="session", autouse=True)
def stub_system_snapshot():
    """Stop tests reading real hardware, which costs ~15s every time.

    generate_system_snapshot() spawns ten PowerShell processes and a couple of
    wmic calls to read GPU, storage, motherboard and display details. Each
    spawn is ~0.7s on Windows, and the whole snapshot measured 14.7s.

    Nine tests in test_regression_planner_router.py route through /system to
    prove the planner and router hold no state between requests — one of them
    five times over. That was ~220 seconds of the suite's ~223, spent
    interrogating the developer's actual graphics card to check that a mocked
    provider still returns "System OK."

    No test reads the snapshot's contents; the ones that care about tool
    content build a ToolContext directly. So this returns a fixed string, and
    real hardware detection is left to the diagnose_* scripts and the app.

    Autouse for the same reason as isolate_storage: a test that forgets is
    slow and machine-dependent, which is not a property worth opting into.
    """
    import athena.tools.system_snapshot as module

    original = module.generate_system_snapshot
    # Kept reachable for the tests that are ABOUT the snapshot: they stub the
    # individual gatherers themselves and need the real assembly around them,
    # which this fixture would otherwise have replaced before they ran.
    module._real_generate_system_snapshot = original
    module.generate_system_snapshot = lambda *args, **kwargs: _STUB_SNAPSHOT

    yield _STUB_SNAPSHOT

    module.generate_system_snapshot = original
    del module._real_generate_system_snapshot
