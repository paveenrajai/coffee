"""Provider registry – model-to-provider resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .anthropic import AnthropicMessagesClient
from .google import GoogleTextClient
from .openai import OpenAIResponsesClient
from .protocol import ProviderProtocol

if TYPE_CHECKING:
    from ..config import Config


def get_provider(
    model: str,
    config: "Config",
    *,
    request_timeout: Optional[float] = None,
    google_explicit_cache: bool = True,
    google_inline_citations: bool = True,
) -> ProviderProtocol:
    """Return the appropriate provider client for the given model name."""
    model_lower = model.lower()
    kwargs: dict = {"config": config}
    if request_timeout is not None:
        kwargs["request_timeout"] = request_timeout
    if model_lower.startswith("claude") or model_lower.startswith("anthropic"):
        return AnthropicMessagesClient(**kwargs)
    if model_lower.startswith("gemini") or model_lower.startswith("google"):
        return GoogleTextClient(
            **kwargs,
            google_explicit_cache=google_explicit_cache,
            google_inline_citations=google_inline_citations,
        )
    return OpenAIResponsesClient(**kwargs)
