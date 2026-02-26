"""LLM provider factory — creates the configured provider at startup."""
import logging
from config import AppConfig
from src.agents.base import LLMProvider

logger = logging.getLogger(__name__)


def create_provider(cfg: AppConfig) -> LLMProvider:
    """Instantiate and return the LLM provider specified in config.

    Supported values of cfg.llm_provider:
        "gemini"  — Google Gemini Flash (default)
        "openai"  — OpenAI GPT-4o-mini
        "claude"  — Anthropic Claude (disabled stub)

    Raises:
        ValueError: If the provider name is unrecognised.
    """
    provider = cfg.llm_provider.lower()
    logger.info("Creating LLM provider: %s", provider)

    if provider == "gemini":
        from src.agents.gemini_provider import GeminiProvider
        return GeminiProvider(api_key=cfg.gemini_api_key)

    if provider == "openai":
        from src.agents.openai_provider import OpenAIProvider
        return OpenAIProvider(api_key=cfg.openai_api_key)

    if provider == "claude":
        from src.agents.claude_provider import ClaudeProvider
        return ClaudeProvider(api_key=cfg.claude_api_key)

    raise ValueError(
        f"Unknown llm_provider {provider!r}. "
        "Choose 'gemini', 'openai', or 'claude'."
    )
