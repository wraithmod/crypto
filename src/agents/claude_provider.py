"""Claude (Anthropic) LLM provider — uses claude-haiku for fast, cheap sentiment analysis."""
import anthropic
from src.agents.base import LLMProvider, LLMError


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider using the async client."""

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return "claude/claude-haiku-4-5-20251001"

    async def complete(self, system: str, user: str, max_tokens: int = 150) -> str:
        try:
            msg = await self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return msg.content[0].text.strip()
        except anthropic.APIError as exc:
            raise LLMError(f"Claude API error: {exc}") from exc
