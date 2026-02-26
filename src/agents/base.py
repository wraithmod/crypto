"""Abstract LLM provider interface.

All LLM integrations must implement this interface so providers can be
swapped without changing any call sites.
"""
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Minimal async interface for text completion."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name, e.g. 'gemini-flash'."""
        ...

    @abstractmethod
    async def complete(self, system: str, user: str, max_tokens: int = 150) -> str:
        """Send a prompt to the LLM and return its text response.

        Args:
            system: System/instruction prompt.
            user:   User message.
            max_tokens: Maximum tokens to generate.

        Returns:
            The model's text response as a plain string.

        Raises:
            LLMError: On any API or network failure (callers catch this).
        """
        ...


class LLMError(Exception):
    """Raised by providers on API failures."""
