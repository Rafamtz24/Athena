"""
LM Studio Provider

Provides a placeholder implementation for LM Studio integration.
"""

import requests

from athena.config.settings import get_settings


class LMStudioProvider:
    """
    A provider for LM Studio local LLM serving.

    Constructor accepts:
        base_url (str): The base URL of the LM Studio server.
            Defaults to value from settings.provider.base_url
    """

    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or get_settings().provider.base_url

    def generate(self, prompt: str, system: str | None = None) -> str:
        """
        Generate a response from LM Studio.

        Args:
            prompt: The input prompt string (user content).
            system: Optional system prompt. Delivered in the `system` role so
                the model treats it as its own identity/instructions rather
                than as a user claim (otherwise the base model's built-in
                identity, e.g. "You are Qwen", stays in force).

        Returns:
            The LLM response text.

        Raises:
            RuntimeError: If the LM Studio provider is unavailable or returns an error.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": get_settings().provider.model,
            "messages": messages,
            "temperature": get_settings().provider.temperature,
            "stream": False
        }
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"LM Studio returned unexpected response format: {data}") from e

    def call(self, prompt: str) -> str:
        """Alias for generate() to match KnowledgeManager expectations."""
        return self.generate(prompt)

    def count_tokens(self, text: str) -> int:
        """Count tokens using a character-based heuristic.

        LM Studio does not expose a tokenizer endpoint, so we use the
        standard heuristic: len(text) // 4.

        Args:
            text: The text to estimate token count for.

        Returns:
            Estimated token count.
        """
        return len(text) // 4

    def get_context_window(self) -> int:
        """Get the default context window size.

        LM Studio does not expose context window via API.
        Returns a conservative default. Override in settings if needed.

        Returns:
            The default context window in tokens (4096).
        """
        from athena.config.settings import get_settings
        # Check if a custom context window is configured via inference config
        try:
            from athena.hardware import HardwareDetector
            from athena.config.inference import AutoConfigurator
            hardware = HardwareDetector().detect()
            config = AutoConfigurator().configure(hardware)
            return config.n_ctx
        except Exception:
            return 4096
