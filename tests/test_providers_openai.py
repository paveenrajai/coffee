"""Tests for OpenAI provider."""

from unittest.mock import MagicMock, patch

import pytest

from coffee_with_llm import Config
from coffee_with_llm.exceptions import ConfigurationError
from coffee_with_llm.providers.openai import OpenAIResponsesClient
from coffee_with_llm.providers.tool_utils import normalize_tool_result


def _config(openai_api_key="test-key"):
    return Config(
        openai_api_key=openai_api_key,
        anthropic_api_key=None,
        google_api_key=None,
        request_timeout=60.0,
    )


class TestOpenAIResponsesClientInitialization:
    """Tests for OpenAIResponsesClient initialization."""

    def test_init_without_api_key(self):
        """Test that missing API key raises ConfigurationError."""
        cfg = Config(
            openai_api_key=None, anthropic_api_key=None, google_api_key=None, request_timeout=60.0
        )
        with pytest.raises(ConfigurationError, match="OpenAI.*not configured"):
            OpenAIResponsesClient(config=cfg)

    def test_init_with_api_key(self):
        """Test successful initialization with API key."""
        with patch("openai.AsyncOpenAI"):
            client = OpenAIResponsesClient(config=_config())
            assert client._api_key == "test-key"

    def test_init_with_missing_openai_package(self):
        """Test that missing OpenAI package raises ConfigurationError."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named 'openai'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ConfigurationError, match="OpenAI package not installed"):
                OpenAIResponsesClient(config=_config())


class TestOpenAIResponsesClientNormalizeToolResult:
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


class TestOpenAIResponsesClientParseResponseFormat:
    """Tests for _parse_response_format method."""

    def test_parse_none(self):
        """Test parsing None."""
        result = OpenAIResponsesClient._parse_response_format(None)
        assert result is None

    def test_parse_json_schema_dict(self):
        """Test parsing JSON schema dict."""
        response_format = {
            "type": "json_schema",
            "json_schema": {"type": "object", "properties": {"name": {"type": "string"}}},
        }
        result = OpenAIResponsesClient._parse_response_format(response_format)
        assert result is not None
        assert "format" in result

    def test_parse_json_string(self):
        """Test parsing JSON string."""
        result = OpenAIResponsesClient._parse_response_format("json")
        assert result is not None
        assert result["format"]["type"] == "json_object"

    def test_parse_markdown_string(self):
        """Test parsing markdown string."""
        result = OpenAIResponsesClient._parse_response_format("markdown")
        assert result is not None
        assert result["format"]["type"] == "markdown"

    def test_parse_text_string(self):
        """Test parsing text string."""
        result = OpenAIResponsesClient._parse_response_format("text")
        assert result is not None
        assert result["format"]["type"] == "text"


class TestOpenAIResponsesClientGenerate:
    """Tests for generate method."""

    @pytest.mark.asyncio
    async def test_generate_basic(self):
        """Test basic generation."""
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.output_text = "Test response"
            mock_response.usage = MagicMock()
            mock_response.usage.input_tokens = 10
            mock_response.usage.output_tokens = 1
            mock_response.usage.cached_tokens = 0
            mock_response.usage.prompt_tokens = 10
            mock_response.required_action = None

            async def mock_create(*args, **kwargs):
                return mock_response

            mock_client_instance.responses.create = mock_create
            mock_openai.return_value = mock_client_instance

            client = OpenAIResponsesClient(config=_config())
            text, usage = await client.generate(prompt="What is Python?", model="gpt-4o-mini")
            assert text == "Test response"
            assert usage is not None
            assert usage.input_tokens == 10

    @pytest.mark.asyncio
    async def test_generate_with_instructions(self):
        """Test generation with instructions."""
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.output_text = "Test response"
            mock_response.usage = MagicMock()
            mock_response.usage.cached_tokens = 0
            mock_response.usage.prompt_tokens = 10
            mock_response.required_action = None

            async def mock_create(*args, **kwargs):
                return mock_response

            mock_client_instance.responses.create = mock_create
            mock_openai.return_value = mock_client_instance

            client = OpenAIResponsesClient(config=_config())
            text, usage = await client.generate(
                prompt="What is Python?",
                model="gpt-4o-mini",
                instructions="You are a helpful assistant.",
            )
            assert text == "Test response"
            assert usage is not None

    @pytest.mark.asyncio
    async def test_generate_with_max_tokens(self):
        """Test generation with max_tokens."""
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.output_text = "Test response"
            mock_response.usage = MagicMock()
            mock_response.usage.cached_tokens = 0
            mock_response.usage.prompt_tokens = 10
            mock_response.required_action = None

            async def mock_create(*args, **kwargs):
                return mock_response

            mock_client_instance.responses.create = mock_create
            mock_openai.return_value = mock_client_instance

            client = OpenAIResponsesClient(config=_config())
            text, usage = await client.generate(
                prompt="What is Python?", model="gpt-4o-mini", max_tokens=100
            )
            assert text == "Test response"
            assert usage is not None
