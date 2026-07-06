"""
Athena Prompt Loader

Loads prompt configurations from JSON files in the athena/prompts/ directory.

Responsibilities:
    - Load prompt JSON files from the configured prompts directory
    - Validate required fields exist in loaded prompts
    - Cache immutable prompts (loaded once, cached forever)
    - Provide prompt objects to existing components

Usage:
    >>> from athena.prompt.loader import PromptLoader
    >>> profile = PromptLoader.load("reasoning")
    >>> profile.system_prompt
    'You are Athena...'

Prompt files are JSON and must contain at minimum a "system_prompt" field.
This is validated at load time and raises a clear error if missing.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional


# ── Default prompts directory ─────────────────────────────────────
# Relative to the project root (where athena/ lives).
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# ── Required fields per prompt type ───────────────────────────────
# Every prompt profile MUST contain these fields.
# Validation catches missing fields at load time.
_REQUIRED_FIELDS = {
    "system_prompt",
}

# ── Optional fields that may appear in any profile ────────────────
_OPTIONAL_FIELDS = {
    "instructions",
    "response_format",
    "tool_context_rules",
    "rules",
    "examples",
    "description",
    "version",
}


class PromptValidationError(Exception):
    """Raised when a prompt file fails validation."""


class PromptProfile:
    """Immutable prompt configuration loaded from a JSON file.
    
    Attributes:
        name: The prompt name (e.g., "reasoning", "extraction").
        data: The raw dictionary loaded from the JSON file.
    
    The profile provides attribute-style access for convenience:
        profile.system_prompt
        profile.instructions
        profile.response_format
    """

    __slots__ = ("name", "data")

    def __init__(self, name: str, data: dict) -> None:
        self.name: str = name
        self.data: dict = data

    def __getattr__(self, attr: str) -> Any:
        """Allow attribute-style access to prompt fields.
        
        Falls back to getattr default behavior for non-field attributes.
        """
        if attr.startswith("_") or attr not in self.data:
            raise AttributeError(
                f"'PromptProfile' object has no attribute '{attr}'"
            )
        return self.data[attr]

    def get(self, key: str, default: Any = None) -> Any:
        """Get a prompt field by key with an optional default."""
        return self.data.get(key, default)

    def has(self, key: str) -> bool:
        """Check if a field exists in this prompt profile."""
        return key in self.data

    def __repr__(self) -> str:
        fields = list(self.data.keys())
        return f"PromptProfile(name='{self.name}', fields={fields})"


class PromptLoader:
    """Loads, validates, and caches prompt configurations.
    
    Prompts are loaded from JSON files in the prompts directory.
    Once loaded, they are cached indefinitely (immutable by design).
    
    To reload after editing a prompt file:
        PromptLoader.clear_cache()     # Clear all cached prompts
        PromptLoader.clear_cache("reasoning")  # Clear specific prompt
    """

    # ── Class-level cache ──────────────────────────────────────────
    # Maps prompt name -> PromptProfile instance.
    # Loaded once, cached forever. Immutable prompt design.
    _cache: dict[str, "PromptProfile"] = {}

    @classmethod
    def load(cls, name: str) -> PromptProfile:
        """Load a prompt profile by name.
        
        Args:
            name: The prompt name (e.g., "reasoning"), which maps to
                  athena/prompts/{name}.json.
        
        Returns:
            A PromptProfile instance with the loaded prompt data.
        
        Raises:
            FileNotFoundError: If the prompt file does not exist.
            json.JSONDecodeError: If the prompt file contains invalid JSON.
            PromptValidationError: If required fields are missing.
        """
        # ── Check cache first ──
        cached = cls._cache.get(name)
        if cached is not None:
            return cached

        # ── Resolve file path ──
        filepath = _PROMPTS_DIR / f"{name}.json"

        if not filepath.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {filepath}\n"
                f"Ensure athena/prompts/{name}.json exists."
            )

        # ── Load and parse JSON ──
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in prompt file '{filepath}': {e.msg}",
                e.doc,
                e.pos,
            )

        # ── Validate required fields ──
        missing = _REQUIRED_FIELDS - set(data.keys())
        if missing:
            raise PromptValidationError(
                f"Prompt file '{filepath}' is missing required field(s): "
                f"{', '.join(sorted(missing))}\n"
                f"Every prompt profile must include at least: "
                f"{', '.join(sorted(_REQUIRED_FIELDS))}"
            )

        # ── Create and cache ──
        profile = PromptProfile(name=name, data=data)
        cls._cache[name] = profile
        return profile

    @classmethod
    def get_system_prompt(cls, name: str) -> str:
        """Convenience: load a prompt and return its system_prompt field.
        
        Args:
            name: The prompt name (e.g., "reasoning").
        
        Returns:
            The system_prompt string from the loaded profile.
        """
        return cls.load(name).system_prompt

    @classmethod
    def get_instructions(cls, name: str) -> Optional[str]:
        """Convenience: load a prompt and return its instructions field.
        
        Args:
            name: The prompt name (e.g., "extraction").
        
        Returns:
            The instructions string, or None if not present.
        """
        return cls.load(name).get("instructions")

    @classmethod
    def get_response_format(cls, name: str) -> Optional[str]:
        """Convenience: load a prompt and return its response_format field.
        
        Args:
            name: The prompt name.
        
        Returns:
            The response_format string, or None if not present.
        """
        return cls.load(name).get("response_format")

    @classmethod
    def clear_cache(cls, name: Optional[str] = None) -> None:
        """Clear the prompt cache.
        
        Args:
            name: Optional specific prompt name to clear.
                  If None, clears ALL cached prompts.
        
        Usage:
            PromptLoader.clear_cache()           # Clear all
            PromptLoader.clear_cache("reasoning")  # Clear specific
        """
        if name is not None:
            cls._cache.pop(name, None)
        else:
            cls._cache.clear()

    @classmethod
    def get_cached_names(cls) -> list[str]:
        """Return the names of currently cached prompts."""
        return list(cls._cache.keys())

    @classmethod
    def reload(cls, name: Optional[str] = None) -> PromptProfile:
        """Reload a prompt profile, bypassing the cache.
        
        Args:
            name: The prompt name to reload. If None, clears all cache
                  but does not return anything.
        
        Returns:
            The reloaded PromptProfile (if name provided).
        
        Raises:
            Same as load().
        """
        cls.clear_cache(name)
        if name is not None:
            return cls.load(name)
        return None