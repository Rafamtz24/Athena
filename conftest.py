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
"""

import pytest

from athena.config.settings import get_settings


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
