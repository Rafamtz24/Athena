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

    def generate(self, prompt: str) -> str:
        """
        Generate a response from LM Studio.

        Args:
            prompt: The input prompt string.

        Returns:
            The LLM response text.

        Raises:
            RuntimeError: If the LM Studio provider is unavailable or returns an error.
        """
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": get_settings().provider.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
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
