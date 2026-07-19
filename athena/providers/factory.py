"""
Provider Factory

Central factory for creating provider instances.
The Brain must never instantiate providers directly.

Changing only the `provider` setting in settings.py must be sufficient
to switch inference backends.

Future providers only require modifications inside this factory.
"""

from athena.config.settings import get_settings


# Hardware detection + inference tuning is identical for every local model on
# this host, so compute it once and share it between the reasoning and learning
# providers instead of probing the hardware twice.
_INFERENCE_CONFIG = None


def _local_provider_class(provider_name: str):
    """Return the provider class backing a local GGUF model.

    Both local providers run the same models with the same inference config and
    expose the same interface; they differ only in how llama.cpp is reached —
    a bundled ``llama-server`` subprocess, or the in-process llama-cpp-python
    bindings.

    Args:
        provider_name: Either "llamaserver" or "llamacpp".

    Returns:
        The provider class to instantiate.
    """
    if provider_name == "llamaserver":
        from athena.providers.llamaserver import LlamaServerProvider

        return LlamaServerProvider

    from athena.providers.llamacpp import LlamaCppProvider

    return LlamaCppProvider


def _get_inference_config():
    """Detect hardware and compute the inference config once, then cache it."""
    global _INFERENCE_CONFIG
    if _INFERENCE_CONFIG is None:
        from athena.hardware import HardwareDetector
        from athena.config.inference import AutoConfigurator

        hardware = HardwareDetector().detect()
        _INFERENCE_CONFIG = AutoConfigurator().configure(hardware)
    return _INFERENCE_CONFIG


class ProviderFactory:
    """
    Single location responsible for creating providers.

    Usage:
        provider = ProviderFactory.create()

    The factory reads the configured provider name from settings
    and returns the appropriate provider instance.

    Supported providers:
        - "llamaserver" -> LlamaServerProvider (default; prebuilt binary)
        - "llamacpp"  -> LlamaCppProvider (in-process; needs compilation)
        - "lmstudio"  -> LMStudioProvider
        - "ollama"    -> OllamaProvider (future)
        - "openrouter" -> OpenRouterProvider (future)

    To switch providers, change only:
        settings.provider.provider = "llamacpp"
        # or
        settings.provider.provider = "lmstudio"

    No Brain changes are required.
    """

    @staticmethod
    def create():
        """
        Create and return the REASONING provider based on the configured name.

        Returns:
            A provider instance implementing generate(prompt) and call(prompt).

        Raises:
            ValueError: If the configured provider is unknown or unsupported.
        """
        provider_name = get_settings().provider.provider

        if provider_name == "lmstudio":
            from athena.providers.lmstudio import LMStudioProvider

            return LMStudioProvider()

        elif provider_name in ("llamacpp", "llamaserver"):
            from athena.providers.model_selector import resolve_model_path

            config = _get_inference_config()

            _print_startup_banner()

            reason_path = resolve_model_path(
                get_settings().provider.reason_model_directory
            )

            provider_class = _local_provider_class(provider_name)
            return provider_class(
                model_path=reason_path,
                inference_config=config,
                label="reasoning",
            )

        else:
            raise ValueError(
                f"Unknown provider: '{provider_name}'. Supported providers: "
                f"llamaserver, llamacpp, lmstudio, ollama, openrouter"
            )

    @staticmethod
    def create_learning(reasoning_provider):
        """
        Create the LEARNING provider — a small, fast model dedicated to the
        knowledge-extraction / memory-reconciliation pipeline.

        If no dedicated learning model is present (the learning folder is
        missing or empty), the reasoning provider is reused so learning still
        works out of the box.

        Args:
            reasoning_provider: The already-created reasoning provider, used as
                the fallback.

        Returns:
            A provider instance for the learning pipeline.
        """
        provider_name = get_settings().provider.provider

        if provider_name in ("llamacpp", "llamaserver"):
            from athena.providers.model_selector import resolve_model_path_optional

            learning_path = resolve_model_path_optional(
                get_settings().provider.learning_model_directory
            )
            if learning_path is None:
                # No dedicated learning model — fall back to the reasoning model.
                print(
                    "No dedicated learning model found — using the reasoning "
                    "model for learning as well.\n"
                    "      (Drop a small GGUF into "
                    f"{get_settings().provider.learning_model_directory} to "
                    "speed up the learning phase.)\n"
                )
                return reasoning_provider

            try:
                provider_class = _local_provider_class(provider_name)
                return provider_class(
                    model_path=learning_path,
                    inference_config=_get_inference_config(),
                    label="learning",
                )
            except Exception as exc:
                # A dedicated learning model exists but could not be loaded
                # alongside the reasoning model. The common cause is simply not
                # enough spare GPU memory to hold a second model (expected on
                # smaller GPUs); a corrupt or incompatible file is rarer. Either
                # way, degrade gracefully to the reasoning model for learning
                # rather than crashing the app — everything keeps working.
                print(
                    "Note: the separate learning model couldn't be loaded next "
                    "to the reasoning model — this is normal when GPU memory is "
                    "limited. Athena will use the reasoning model for learning "
                    "instead, so everything still works."
                )
                # Keep the technical detail on its own line for troubleshooting.
                print(f"      (details: {exc})")
                return reasoning_provider

        # Other providers serve a single model/endpoint — reuse it for learning.
        return reasoning_provider


# Startup logo shown at launch. Hardware detection and inference tuning still
# run (they configure the model); they are just no longer printed.
_ATHENA_BANNER = r"""
    _  _____ _   _ _____ _   _    _
   / \|_   _| | | | ____| \ | |  / \
  / _ \ | | | |_| |  _| |  \| | / _ \
 / ___ \| | |  _  | |___| |\  |/ ___ \
/_/   \_\_| |_| |_|_____|_| \_/_/   \_\
"""


def _print_startup_banner() -> None:
    """Print the Athena startup logo."""
    print(_ATHENA_BANNER)