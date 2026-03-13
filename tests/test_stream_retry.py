"""Tests for stream retry behavior."""

from unittest.mock import patch

import pytest

from coffee_with_llm import AskLLM, TokenUsage


class TestStreamRetry:
    """Tests for StreamResult retry on rate limit."""

    @pytest.mark.asyncio
    async def test_stream_retries_on_rate_limit_before_first_chunk(self, mock_openai_api_key):
        """Stream retries when rate limit hits before first chunk."""
        call_count = 0

        async def mock_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("429 rate limit exceeded")
            yield "Hello "
            yield "world!"
            yield TokenUsage(10, 5, 15, None)

        with patch("openai.AsyncOpenAI"):
            llm = AskLLM(model="gpt-4o-mini", max_retries=3)
            llm._client.generate_stream = mock_stream

        result = await llm.ask(prompt="hi", stream=True)
        chunks = []
        async for c in result:
            chunks.append(c)
        assert chunks == ["Hello ", "world!"]
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_stream_raises_runtime_error_if_anext_without_aiter(self, mock_openai_api_key):
        """StreamResult.__anext__ raises if called without __aiter__ first."""

        async def mock_stream(*args, **kwargs):
            yield "x"
            yield TokenUsage(0, 0, 0, None)

        with patch("openai.AsyncOpenAI"):
            llm = AskLLM(model="gpt-4o-mini")
            llm._client.generate_stream = mock_stream

        result = await llm.ask(prompt="hi", stream=True)
        with pytest.raises(RuntimeError, match="__aiter__ was not called"):
            await result.__anext__()
