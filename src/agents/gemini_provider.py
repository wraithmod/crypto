"""Google Gemini provider using the google-genai SDK."""
import logging

from google import genai
from google.genai import errors as genai_errors

from src.agents.base import LLMProvider, LLMError

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Calls Google Gemini Flash via the native async google-genai SDK."""

    MODEL = "gemini-2.5-flash-lite"

    def __init__(self, api_key: str) -> None:
        self._client = genai.Client(api_key=api_key)

    @property
    def name(self) -> str:
        return f"gemini/{self.MODEL}"

    async def complete(self, system: str, user: str, max_tokens: int = 150) -> str:
        prompt = f"{system}\n\n{user}"
        try:
            response = await self._client.aio.models.generate_content(
                model=self.MODEL,
                contents=prompt,
                config=genai.types.GenerateContentConfig(max_output_tokens=max_tokens),
            )
            return response.text.strip()
        except genai_errors.APIError as exc:
            raise LLMError(f"Gemini API error: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"Unexpected Gemini error: {exc}") from exc
