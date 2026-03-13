"""Tests for Anthropic provider."""

from unittest.mock import MagicMock, patch

import pytest

from coffee_with_llm import Config
from coffee_with_llm.exceptions import ConfigurationError
from coffee_with_llm.providers.anthropic import AnthropicMessagesClient
from coffee_with_llm.providers.anthropic.messages_client import _convert_tools_to_anthropic
from coffee_with_llm.providers.tool_utils import normalize_tool_result


def _config(anthropic_api_key="test-key"):
    return Config(
        openai_api_key=None,
        anthropic_api_key=anthropic_api_key,
        google_api_key=None,
        request_timeout=60.0,
    )


class TestAnthropicMessagesClientInitialization:
    """Tests for AnthropicMessagesClient initialization."""

    def test_init_without_api_key(self):
        """Test that missing API key raises ConfigurationError."""
        cfg = Config(
            openai_api_key=None, anthropic_api_key=None, google_api_key=None, request_timeout=60.0
        )
        with pytest.raises(ConfigurationError, match="Anthropic.*not configured"):
            AnthropicMessagesClient(config=cfg)

    def test_init_with_api_key(self):
        """Test successful initialization with API key."""
        fake_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            client = AnthropicMessagesClient(config=_config())
            assert client._api_key == "test-key"

    def test_init_with_missing_anthropic_package(self):
        """Test that missing Anthropic package raises ConfigurationError."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("No module named 'anthropic'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ConfigurationError, match="Anthropic package not installed"):
                AnthropicMessagesClient(config=_config())


class TestConvertToolsToAnthropic:
    """Tests for _convert_tools_to_anthropic."""

    def test_convert_openai_style_function(self):
        """Test converting OpenAI-style function tool."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]
        result = _convert_tools_to_anthropic(tools)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get weather"
        assert result[0]["input_schema"]["type"] == "object"
        assert "location" in result[0]["input_schema"]["properties"]

    def test_convert_already_anthropic_format(self):
        """Test passing through already-Anthropic format."""
        tools = [
            {
                "name": "get_time",
                "description": "Get time",
                "input_schema": {"type": "object", "properties": {"tz": {"type": "string"}}},
            }
        ]
        result = _convert_tools_to_anthropic(tools)
        assert len(result) == 1
        assert result[0]["name"] == "get_time"
        assert result[0]["input_schema"]["type"] == "object"

    def test_convert_empty_list(self):
        """Test empty tools list."""
        assert _convert_tools_to_anthropic([]) == []


class TestAnthropicMessagesClientNormalizeToolResult:
    """Tests for normalize_tool_result (shared tool_utils)."""

    def test_normalize_with_ok_attribute(self):
        """Test normalization with object having ok attribute."""
        mock_result = MagicMock()
        mock_result.ok = True
        mock_result.result = {"data": "test"}
        mock_result.error = None

        normalized = normalize_tool_result(mock_result)
        assert normalized == {"ok": True, "result": {"data": "test"}, "error": None}

    def test_normalize_with_dict(self):
        """Test normalization with dict."""
        result_dict = {"ok": True, "result": {"data": "test"}, "error": None}
        normalized = normalize_tool_result(result_dict)
        assert normalized == result_dict

    def test_normalize_with_invalid_input(self):
        """Test normalization with invalid input."""
        normalized = normalize_tool_result("invalid")
        assert normalized == {"ok": False, "result": {}, "error": None}


class TestAnthropicMessagesClientGenerate:
    """Tests for generate method."""

    @pytest.mark.asyncio
    async def test_generate_basic(self):
        """Test basic generation."""
        fake_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [{"type": "text", "text": "Test response"}]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

        async def mock_create(*args, **kwargs):
            return mock_response

        mock_client.messages.create = mock_create
        fake_anthropic.AsyncAnthropic = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            client = AnthropicMessagesClient(config=_config())
            text, usage = await client.generate(prompt="What is Python?", model="claude-sonnet-4-6")
            assert text == "Test response"
            assert usage is not None
            assert usage.input_tokens == 10
            assert usage.output_tokens == 5

    @pytest.mark.asyncio
    async def test_generate_with_system_instruction(self):
        """Test generation with system instruction."""
        fake_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [{"type": "text", "text": "Hello"}]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=5, output_tokens=3)

        async def mock_create(*args, **kwargs):
            assert kwargs.get("system") == "You are helpful."
            return mock_response

        mock_client.messages.create = mock_create
        fake_anthropic.AsyncAnthropic = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            client = AnthropicMessagesClient(config=_config())
            text, usage = await client.generate(
                prompt="Hi",
                model="claude-sonnet-4-6",
                instructions="You are helpful.",
            )
            assert text == "Hello"
