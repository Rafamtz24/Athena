"""
Athena Providers - Base Module

Defines the abstract base class for all LLM providers in the Athena AI platform.
"""

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.

    All concrete provider implementations must implement these methods:
        - generate(): Generate a response from the LLM
        - health_check(): Check if the provider is healthy/available

    This interface allows swapping providers without changing the brain's logic.
    """

    @abstractmethod
    async def generate(self, message: str, **kwargs: Any) -> str:
        """
        Generate a response from the LLM for the given message.

        Args:
            message: The input message to send to the LLM.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The generated response string.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and available.

        Returns:
            True if the provider is healthy, False otherwise.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError  # pragma: no cover