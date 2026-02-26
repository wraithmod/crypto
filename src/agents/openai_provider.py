"""OpenAI provider (GPT-4o-mini) for cost-effective completions."""
import logging

from openai import AsyncOpenAI
from openai import OpenAIError

from src.agents.base import LLMProvider, LLMError

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """Calls OpenAI GPT-4o-mini."""

    MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)

    @property
    def name(self) -> str:
        return f"openai/{self.MODEL}"

    async def complete(self, system: str, user: str, max_tokens: int = 150) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self.MODEL,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content.strip()
        except OpenAIError as exc:
            raise LLMError(f"OpenAI API error: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"Unexpected OpenAI error: {exc}") from exc
