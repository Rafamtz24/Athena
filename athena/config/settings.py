"""
Athena Configuration Module

Provides centralized application configuration with sensible defaults.
Future expansion will support external config sources (env vars, .env files, YAML).
"""

from dataclasses import dataclass, field


@dataclass(frozen=False)
class AppSettings:
    """
    Centralized application settings for Athena AI platform.

    Attributes:
        app_name: The name of the application.
        version: Application version string.
        debug: Enable debug mode (more verbose logging, auto-reload).
        llm_provider: Provider identifier (e.g., 'lm-studio', 'openai').
                      Used to select which provider implementation to use.
    """

    app_name: str = "Athena"
    version: str = "0.2.0"
    debug: bool = False
    llm_provider: str = "default"

    def __post_init__(self):
        """Validate settings after initialization."""
        if not self.app_name:
            raise ValueError("app_name cannot be empty")
        if not self.version:
            raise ValueError("version cannot be empty")


# Global configuration instance (singleton pattern)
settings = AppSettings()


def get_settings() -> AppSettings:
    """
    Retrieve the global application settings.

    Returns:
        The singleton AppSettings instance.
    """
    return settings