"""Tests for AskLLM main class."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coffee_with_llm import AskLLM, TokenUsage
from coffee_with_llm.exceptions import APIError, ConfigurationError, ValidationError


class TestAskLLMInitialization:
    """Tests for AskLLM initialization."""

    def test_init_without_model_raises_error(self, mock_both_api_keys):
        """Test that initialization without model raises ValidationError."""
        with pytest.raises(ValidationError, match="Model name is required"):
            AskLLM(model=None)

    def test_init_with_empty_model_raises_error(self, mock_both_api_keys):
        """Test that initialization with empty model raises ValidationError."""
        with pytest.raises(ValidationError, match="Model name is required"):
            AskLLM(model="")

    def test_init_with_openai_model(self, mock_openai_api_key):
        """Test initialization with OpenAI model."""
        with patch("openai.AsyncOpenAI"):
            llm = AskLLM(model="gpt-4o-mini")
            assert llm._model == "gpt-4o-mini"

    def test_init_with_gemini_model(self, mock_google_api_key):
        """Test initialization with Gemini model."""
        with patch("coffee_with_llm.providers.google.text_client.genai.Client"):
            llm = AskLLM(model="gemini-2.0-flash-exp")
            assert llm._model == "gemini-2.0-flash-exp"

    def test_init_with_google_prefix(self, mock_google_api_key):
        """Test initialization with 'google' prefix."""
        with patch("coffee_with_llm.providers.google.text_client.genai.Client"):
            llm = AskLLM(model="google-gemini-pro")
            assert llm._model == "google-gemini-pro"

    def test_init_with_missing_openai_key(self):
        """Test initialization fails when OpenAI API key is missing."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError):
                AskLLM(model="gpt-4o-mini")

    def test_init_with_missing_google_key(self):
        """Test initialization fails when Google API key is missing."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError):
                AskLLM(model="gemini-2.0-flash-exp")

    def test_init_client_initialization_error(self, mock_openai_api_key):
        """Test that client initialization errors are wrapped."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "openai":
                raise Exception("Connection failed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ConfigurationError, match="Failed to initialize client"):
                AskLLM(model="gpt-4o-mini")


class TestAskLLMAskMethod:
    """Tests for AskLLM.ask method."""

    @pytest.mark.asyncio
    async def test_ask_with_empty_prompt(self, mock_openai_api_key):
        """Test that empty prompt raises ValidationError."""
        with patch("openai.AsyncOpenAI"):
            llm = AskLLM(model="gpt-4o-mini")
            with pytest.raises(ValidationError, match="Prompt cannot be empty"):
                await llm.ask(prompt="")

    @pytest.mark.asyncio
    async def test_ask_with_whitespace_only_prompt(self, mock_openai_api_key):
        """Test that whitespace-only prompt raises ValidationError."""
        with patch("openai.AsyncOpenAI"):
            llm = AskLLM(model="gpt-4o-mini")
            with pytest.raises(ValidationError, match="Prompt cannot be empty"):
                await llm.ask(prompt="   ")

    @pytest.mark.asyncio
    async def test_ask_with_invalid_max_tokens(self, mock_openai_api_key):
        """Test that negative max_tokens raises ValidationError."""
        with patch("openai.AsyncOpenAI"):
            llm = AskLLM(model="gpt-4o-mini")
            with pytest.raises(ValidationError, match="max_tokens must be positive"):
                await llm.ask(prompt="test", max_tokens=-1)

    @pytest.mark.asyncio
    async def test_ask_with_zero_max_tokens(self, mock_openai_api_key):
        """Test that zero max_tokens raises ValidationError."""
        with patch("openai.AsyncOpenAI"):
            llm = AskLLM(model="gpt-4o-mini")
            with pytest.raises(ValidationError, match="max_tokens must be positive"):
                await llm.ask(prompt="test", max_tokens=0)

    @pytest.mark.asyncio
    async def test_ask_with_invalid_temperature(self, mock_openai_api_key):
        """Test that invalid temperature raises ValidationError."""
        with patch("openai.AsyncOpenAI"):
            llm = AskLLM(model="gpt-4o-mini")
            with pytest.raises(ValidationError, match="temperature must be between 0 and 2"):
                await llm.ask(prompt="test", temperature=-1)

            with pytest.raises(ValidationError, match="temperature must be between 0 and 2"):
                await llm.ask(prompt="test", temperature=3)

    @pytest.mark.asyncio
    async def test_ask_with_invalid_top_p(self, mock_openai_api_key):
        """Test that invalid top_p raises ValidationError."""
        with patch("openai.AsyncOpenAI"):
            llm = AskLLM(model="gpt-4o-mini")
            with pytest.raises(ValidationError, match="top_p must be between 0 and 1"):
                await llm.ask(prompt="test", top_p=-1)

            with pytest.raises(ValidationError, match="top_p must be between 0 and 1"):
                await llm.ask(prompt="test", top_p=2)

    @pytest.mark.asyncio
    async def test_ask_with_invalid_max_steps(self, mock_openai_api_key):
        """Test that invalid max_steps raises ValidationError."""
        with patch("openai.AsyncOpenAI"):
            llm = AskLLM(model="gpt-4o-mini")
            with pytest.raises(ValidationError, match="max_steps must be positive"):
                await llm.ask(prompt="test", max_steps=0)

    @pytest.mark.asyncio
    async def test_ask_with_invalid_max_effective_tool_steps(self, mock_openai_api_key):
        """Test that invalid max_effective_tool_steps raises ValidationError."""
        with patch("openai.AsyncOpenAI"):
            llm = AskLLM(model="gpt-4o-mini")
            with pytest.raises(ValidationError, match="max_effective_tool_steps must be positive"):
                await llm.ask(prompt="test", max_effective_tool_steps=0)

    @pytest.mark.asyncio
    async def test_ask_with_openai_success(self, mock_openai_api_key):
        """Test successful OpenAI API call."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.cached_tokens = 0
        mock_response.usage.prompt_tokens = 10

        async def mock_generate(*args, **kwargs):
            return mock_response

        mock_client.responses.create = mock_generate

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            llm = AskLLM(model="gpt-4o-mini")
            mock_client_instance = MagicMock()
            mock_client_instance.generate = AsyncMock(
                return_value=(
                    "Test response",
                    TokenUsage(
                        input_tokens=10, output_tokens=5, total_tokens=15, cached_tokens=None
                    ),
                )
            )
            llm._client.generate = mock_client_instance.generate

            result = await llm.ask(prompt="What is Python?")
            assert result.text == "Test response"
            assert result.usage.input_tokens == 10
            assert result.usage.output_tokens == 5

    @pytest.mark.asyncio
    async def test_ask_with_google_success(self, mock_google_api_key):
        """Test successful Google API call."""
        with patch("coffee_with_llm.providers.google.text_client.genai.Client"):
            llm = AskLLM(model="gemini-2.0-flash-exp")
            mock_client_instance = MagicMock()
            mock_client_instance.generate = AsyncMock(return_value="Test response")
            llm._client.generate = mock_client_instance.generate

            result = await llm.ask(prompt="What is Python?")
            assert result.text == "Test response"

    @pytest.mark.asyncio
    async def test_ask_with_system_instruct(self, mock_openai_api_key):
        """Test ask with system instruction."""
        with patch("openai.AsyncOpenAI"):
            llm = AskLLM(model="gpt-4o-mini")
            mock_client_instance = MagicMock()
            mock_client_instance.generate = AsyncMock(return_value="Test response")
            llm._client.generate = mock_client_instance.generate

            result = await llm.ask(
                prompt="What is Python?", system_instruct="You are a helpful assistant."
            )
            assert result.text == "Test response"

    @pytest.mark.asyncio
    async def test_ask_api_error_handling(self, mock_openai_api_key):
        """Test that API errors are properly wrapped."""
        with patch("openai.AsyncOpenAI"):
            llm = AskLLM(model="gpt-4o-mini")
            mock_client_instance = MagicMock()
            mock_client_instance.generate = AsyncMock(side_effect=Exception("API error"))
            llm._client.generate = mock_client_instance.generate

            with pytest.raises(APIError, match="Failed to generate response"):
                await llm.ask(prompt="test")

    @pytest.mark.asyncio
    async def test_ask_preserves_validation_errors(self, mock_openai_api_key):
        """Test that ValidationError is not wrapped."""
        with patch("openai.AsyncOpenAI"):
            llm = AskLLM(model="gpt-4o-mini")
            mock_client_instance = MagicMock()
            mock_client_instance.generate = AsyncMock(side_effect=ValidationError("Invalid"))
            llm._client.generate = mock_client_instance.generate

            with pytest.raises(ValidationError, match="Invalid"):
                await llm.ask(prompt="test")

    @pytest.mark.asyncio
    async def test_ask_preserves_configuration_errors(self, mock_openai_api_key):
        """Test that ConfigurationError is not wrapped."""
        with patch("openai.AsyncOpenAI"):
            llm = AskLLM(model="gpt-4o-mini")
            mock_client_instance = MagicMock()
            mock_client_instance.generate = AsyncMock(
                side_effect=ConfigurationError("Config error")
            )
            llm._client.generate = mock_client_instance.generate

            with pytest.raises(ConfigurationError, match="Config error"):
                await llm.ask(prompt="test")


class TestAskLLMStreaming:
    """Tests for AskLLM streaming."""

    @pytest.mark.asyncio
    async def test_stream_returns_stream_result(self, mock_openai_api_key):
        """stream=True returns StreamResult with chunks and usage."""

        async def mock_stream(*args, **kwargs):
            yield "Hello "
            yield "world!"
            yield TokenUsage(10, 5, 15, None)

        with patch("openai.AsyncOpenAI"):
            llm = AskLLM(model="gpt-4o-mini")
            llm._client.generate_stream = mock_stream

        result = await llm.ask(prompt="hi", stream=True)
        chunks = []
        async for c in result:
            chunks.append(c)
        assert chunks == ["Hello ", "world!"]
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    @pytest.mark.asyncio
    async def test_stream_with_tools_raises(self, mock_openai_api_key):
        """stream=True with tools_schema raises ValidationError."""
        with patch("openai.AsyncOpenAI"):
            llm = AskLLM(model="gpt-4o-mini")
            with pytest.raises(ValidationError, match="stream=True is not supported"):
                await llm.ask(
                    prompt="hi",
                    stream=True,
                    tools_schema=[
                        {"type": "function", "function": {"name": "x", "parameters": {}}}
                    ],
                )

    @pytest.mark.asyncio
    async def test_stream_with_response_format_raises(self, mock_openai_api_key):
        """stream=True with response_format raises ValidationError."""
        with patch("openai.AsyncOpenAI"):
            llm = AskLLM(model="gpt-4o-mini")
            with pytest.raises(ValidationError, match="stream=True is not supported"):
                await llm.ask(
                    prompt="hi",
                    stream=True,
                    response_format={"type": "json_object"},
                )
