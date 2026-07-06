"""
Provider Factory

Central factory for creating provider instances.
The Brain must never instantiate providers directly.

Changing only the `provider` setting in settings.py must be sufficient
to switch inference backends.

Future providers only require modifications inside this factory.
"""

from athena.config.settings import get_settings


class ProviderFactory:
    """
    Single location responsible for creating providers.

    Usage:
        provider = ProviderFactory.create()

    The factory reads the configured provider name from settings
    and returns the appropriate provider instance.

    Supported providers:
        - "lmstudio"  -> LMStudioProvider
        - "llamacpp"  -> LlamaCppProvider
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
        Create and return a provider instance based on the configured provider name.

        Returns:
            A provider instance implementing generate(prompt) and call(prompt).

        Raises:
            ValueError: If the configured provider is unknown or unsupported.
        """
        provider_name = get_settings().provider.provider

        if provider_name == "lmstudio":
            from athena.providers.lmstudio import LMStudioProvider

            return LMStudioProvider()

        elif provider_name == "llamacpp":
            from athena.hardware import HardwareDetector
            from athena.config.inference import AutoConfigurator, InferenceConfiguration
            from athena.providers.llamacpp import LlamaCppProvider

            # Auto-detect hardware and compute optimal inference config
            hardware = HardwareDetector().detect()
            config = AutoConfigurator().configure(hardware)

            _print_startup_banner()

            return LlamaCppProvider(inference_config=config)

        else:
            raise ValueError(
                f"Unknown provider: '{provider_name}'. "
                f"Supported providers: lmstudio, llamacpp, ollama, openrouter"
            )


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