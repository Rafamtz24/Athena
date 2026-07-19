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

    # "llamaserver" is the default because it needs no compiler: setup
    # downloads a prebuilt llama.cpp binary (~32 MB) and Athena runs it as a
    # child process. "llamacpp" uses the in-process llama-cpp-python bindings
    # instead, which have to be compiled from source on Windows.
    provider: str = "llamaserver"
    base_url: str = "http://127.0.0.1:1234"
    model: str = "qwen2.5-7b-instruct"
    temperature: float = 0.7

    # Where setup places the prebuilt llama.cpp binaries (llamaserver provider).
    runtime_directory: str = "runtime/llama"

    # Local GGUF model configuration.
    # Reasoning and learning models live in separate sub-folders so a small,
    # fast, non-thinking model can handle the learning pipeline while a larger
    # model does the reasoning. If the learning folder holds no model, learning
    # falls back to the reasoning model.
    model_directory: str = "models"
    reason_model_directory: str = "models/reason"
    learning_model_directory: str = "models/learning"
    reasoning_model: str = "qwen2.5-7b-instruct-q4_k_m.gguf"

    # Hard cap on tokens generated per call. Prevents "thinking" models
    # (which emit long <think>…</think> traces) from generating unbounded
    # output and appearing to hang — including on the silent learning-phase
    # calls that run after each response.
    max_tokens: int = 2048

    # Whether "thinking" models are allowed to emit their <think>…</think>
    # reasoning trace. When False, the Qwen3 `/no_think` soft-switch is sent
    # so the model answers directly (faster). Toggle at runtime with
    # `/think on` and `/think off`. Reasoning is stripped from output either
    # way; this only controls whether the model spends tokens producing it.
    thinking_enabled: bool = True


@dataclass(frozen=False)
class StorageSettings:
    """Paths for persistent storage."""

    working_memory_path: str = "data/working_memory.json"
    chat_history_path: str = "data/chat_history.json"
    semantic_memory_path: str = "data/semantic_memory.json"
    books_path: str = "books"
    # Runtime-adjustable preferences that persist across restarts
    # (e.g. the /think toggle). Static defaults live in this module.
    user_prefs_path: str = "data/user_prefs.json"


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
class BudgetSettings:
    """Settings for the Context Budget Manager."""

    # Fraction of the context window reserved for model generation output.
    # The prompt will use (1 - generation_reserve_ratio) of the window.
    generation_reserve_ratio: float = 0.25


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
    version: str = "4.1.0"
    debug: bool = False
    provider: ProviderSettings = field(default_factory=ProviderSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    learning: LearningSettings = field(default_factory=LearningSettings)
    web_search: WebSearchSettings = field(default_factory=WebSearchSettings)
    prompt: PromptSettings = field(default_factory=PromptSettings)
    budget: BudgetSettings = field(default_factory=BudgetSettings)

    def __post_init__(self):
        """Validate settings after initialization."""
        if not self.app_name:
            raise ValueError("app_name cannot be empty")
        if not self.version:
            raise ValueError("version cannot be empty")


# Global configuration instance (singleton pattern)
settings = AppSettings()

# Overlay any persisted runtime preferences (e.g. the /think toggle) so they
# survive restarts. Kept out of the dataclass defaults on purpose: those stay
# static, this reflects the user's last choice.
from athena.config.persistence import apply_to_settings  # noqa: E402

apply_to_settings(settings)


def get_settings() -> AppSettings:
    """
    Retrieve the global application settings.

    Returns:
        The singleton AppSettings instance.
    """
    return settings
