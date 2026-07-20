"""User preference persistence.

A tiny JSON store for settings the user changes at runtime and expects to
survive restarts — currently the `/think` and `/learn` toggles. Kept separate
from settings.py so the values there remain pure, static defaults.

Only keys listed in ``_PERSISTED_KEYS`` are read from and written to disk;
anything else in the file is ignored. Failures (missing/corrupt file, I/O
errors) are non-fatal and fall back to the defaults.
"""

import json
from pathlib import Path

# Preference key -> (settings section, attribute on that section).
_PERSISTED_KEYS = {
    "thinking_enabled": ("provider", "thinking_enabled"),
    "show_thinking": ("provider", "show_thinking"),
    "learning_enabled": ("learning", "enabled"),
}


def _load_raw(path: str) -> dict:
    """Read the prefs file, returning {} if missing or unreadable."""
    prefs_path = Path(path)
    if not prefs_path.exists():
        return {}
    try:
        with open(prefs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_raw(path: str, prefs: dict) -> None:
    """Write the prefs file atomically."""
    prefs_path = Path(path)
    prefs_path.parent.mkdir(parents=True, exist_ok=True)
    temp = prefs_path.with_suffix(".json.tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=2)
    temp.replace(prefs_path)


def apply_to_settings(settings) -> None:
    """Overlay persisted preferences onto a freshly built settings object.

    Called once at startup. Only recognised keys with a type matching the
    current default are applied, so a stale or hand-edited file can never
    crash startup.
    """
    prefs = _load_raw(settings.storage.user_prefs_path)
    for key, (section, attr) in _PERSISTED_KEYS.items():
        if key not in prefs:
            continue
        section_obj = getattr(settings, section, None)
        if section_obj is None:
            continue
        current = getattr(section_obj, attr)
        if isinstance(prefs[key], type(current)):
            setattr(section_obj, attr, prefs[key])


def set_pref(key: str, value) -> None:
    """Persist a preference and apply it to the live settings immediately.

    Args:
        key: One of the keys in ``_PERSISTED_KEYS``.
        value: The value to store.

    Raises:
        KeyError: If ``key`` is not a managed preference.
    """
    if key not in _PERSISTED_KEYS:
        raise KeyError(f"Unknown preference: {key!r}")

    from athena.config.settings import get_settings

    settings = get_settings()
    path = settings.storage.user_prefs_path
    prefs = _load_raw(path)
    prefs[key] = value
    _save_raw(path, prefs)

    # Reflect the change in the live settings too.
    section, attr = _PERSISTED_KEYS[key]
    setattr(getattr(settings, section), attr, value)
