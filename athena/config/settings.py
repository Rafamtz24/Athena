"""
Athena Configuration Module

Provides centralized application configuration with sensible defaults.
Future expansion will support external config sources (env vars, .env files, YAML).
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=False)
class ProviderSettings:
    """Settings for the LLM provider."""

    provider: str = "llamacpp"
    base_url: str = "http://127.0.0.1:1234"
    model: str = "qwen2.5-7b-instruct"
    temperature: float = 0.7

    # Local GGUF model configuration
    model_directory: str = "models"
    reasoning_model: str = "qwen2.5-7b-instruct-q4_k_m.gguf"


@dataclass(frozen=False)
class StorageSettings:
    """Paths for persistent storage."""

    working_memory_path: str = "data/working_memory.json"
    chat_history_path: str = "data/chat_history.json"
    semantic_memory_path: str = "data/semantic_memory.json"


@dataclass(frozen=False)
class RetrievalSettings:
    """Settings for semantic memory retrieval."""

    # Maximum number of semantic memory entries to retrieve per query
    max_results: int = 10


@dataclass(frozen=False)
class LearningSettings:
    """Settings for the learning pipeline."""

    enabled: bool = True


@dataclass(frozen=False)
class WebSearchSettings:
    """Settings for the Web Search tool."""

    enabled: bool = True
    provider: str = "duckduckgo"
    max_results: int = 5
    timeout: int = 10
    user_agent: str = "Athena"


@dataclass(frozen=False)
class PromptSettings:
    """Settings for prompt construction."""

    csize: int = 4000


@dataclass(frozen=False)
class PerformanceSettings:
    """Settings for inference performance tuning."""

    # Performance mode:
    #   "auto"     – automatically configure for best balance (default)
    #   "balanced" – same as auto (future: more conservative)
    #   "maximum"  – prioritise throughput / context size (future)
    #   "cpu_only" – force CPU inference, no GPU offloading (future)
    performance_mode: str = "auto"


@dataclass(frozen=False)
class AppSettings:
    """
    Centralized application settings for Athena AI platform.

    Attributes:
        app_name: The name of the application.
        version: Application version string.
        debug: Enable debug mode (more verbose logging, auto-reload).
        provider: LLM provider configuration.
        performance: Inference performance configuration.
        storage: Persistent storage path configuration.
        retrieval: Semantic memory retrieval configuration.
        learning: Learning pipeline configuration.
        prompt: Prompt construction configuration.
    """

    app_name: str = "Athena"
    version: str = "0.2.0"
    debug: bool = False
    provider: ProviderSettings = field(default_factory=ProviderSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    learning: LearningSettings = field(default_factory=LearningSettings)
    web_search: WebSearchSettings = field(default_factory=WebSearchSettings)
    prompt: PromptSettings = field(default_factory=PromptSettings)

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
