"""Tests for provider registry."""

import sys
from unittest.mock import MagicMock, patch

from coffee_with_llm import Config
from coffee_with_llm.providers.anthropic import AnthropicMessagesClient
from coffee_with_llm.providers.google import GoogleTextClient
from coffee_with_llm.providers.openai import OpenAIResponsesClient
from coffee_with_llm.providers.protocol import ProviderProtocol
from coffee_with_llm.providers.registry import get_provider


def _config():
    return Config.from_env()


class TestGetProvider:
    """Tests for get_provider model resolution."""

    def test_openai_model_returns_openai_client(self):
        """gpt-* returns OpenAIResponsesClient."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
            with patch("openai.AsyncOpenAI"):
                client = get_provider("gpt-4o-mini", _config())
                assert isinstance(client, OpenAIResponsesClient)

    def test_claude_model_returns_anthropic_client(self):
        """claude-* returns AnthropicMessagesClient."""
        fake_anthropic = MagicMock()
        with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
                client = get_provider("claude-sonnet-4-6", _config())
                assert isinstance(client, AnthropicMessagesClient)

    def test_gemini_model_returns_google_client(self):
        """gemini-* returns GoogleTextClient."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test"}):
            with patch("coffee_with_llm.providers.google.text_client.genai.Client"):
                client = get_provider("gemini-2.0-flash", _config())
                assert isinstance(client, GoogleTextClient)

    def test_google_prefix_returns_google_client(self):
        """google-* returns GoogleTextClient."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test"}):
            with patch("coffee_with_llm.providers.google.text_client.genai.Client"):
                client = get_provider("google-gemini-pro", _config())
                assert isinstance(client, GoogleTextClient)

    def test_provider_implements_protocol(self):
        """All returned providers implement ProviderProtocol."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
            with patch("openai.AsyncOpenAI"):
                client = get_provider("gpt-4o-mini", _config())
                assert isinstance(client, ProviderProtocol)
