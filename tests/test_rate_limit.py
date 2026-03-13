"""Tests for rate_limit module."""

import pytest

from coffee_with_llm.rate_limit import (
    RATE_LIMIT_BACKOFF_BASE,
    is_rate_limit_error,
    retry_stream,
    with_retry,
)


class TestIsRateLimitError:
    """Tests for is_rate_limit_error."""

    def test_429_string(self):
        assert is_rate_limit_error(Exception("Error 429: Too many requests"))

    def test_rate_limit_string(self):
        assert is_rate_limit_error(Exception("Rate limit exceeded"))

    def test_quota_exceeded(self):
        assert is_rate_limit_error(Exception("quota exceeded"))

    def test_normal_error(self):
        assert not is_rate_limit_error(Exception("Connection refused"))

    def test_http_status_error_429(self):
        try:
            from httpx import HTTPStatusError, Request, Response

            req = Request("GET", "https://api.example.com")
            resp = Response(429)
            err = HTTPStatusError("429", request=req, response=resp)
            assert is_rate_limit_error(err)
        except ImportError:
            pytest.skip("httpx not installed")


class TestWithRetry:
    """Tests for with_retry."""

    @pytest.mark.asyncio
    async def test_success_first_try(self):
        call_count = 0

        async def op():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await with_retry(op, max_retries=3)
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        call_count = 0

        async def op():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("429 rate limit")
            return "ok"

        result = await with_retry(op, max_retries=3)
        assert result == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_raises_immediately(self):
        call_count = 0

        async def op():
            nonlocal call_count
            call_count += 1
            raise ValueError("bad request")

        with pytest.raises(ValueError, match="bad request"):
            await with_retry(op, max_retries=3)
        assert call_count == 1


class TestRetryStream:
    """Tests for retry_stream."""

    @pytest.mark.asyncio
    async def test_yields_all_items(self):
        from coffee_with_llm.types import TokenUsage

        async def stream():
            yield "a"
            yield "b"
            yield TokenUsage(1, 2, 3, None)

        def factory():
            return stream()

        chunks = []
        async for item in retry_stream(factory, max_retries=3):
            chunks.append(item)
        assert chunks == ["a", "b", TokenUsage(1, 2, 3, None)]

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit(self):
        from coffee_with_llm.types import TokenUsage

        call_count = 0

        async def stream():
            nonlocal call_count
            call_count += 1
            yield "a"
            if call_count == 1:
                raise Exception("429 rate limit")
            yield "b"
            yield TokenUsage(0, 0, 0, None)

        def factory():
            return stream()

        chunks = []
        async for item in retry_stream(factory, max_retries=3):
            chunks.append(item)
        assert chunks == ["a", "a", "b", TokenUsage(0, 0, 0, None)]
        assert call_count == 2


class TestConstants:
    def test_backoff_base(self):
        assert RATE_LIMIT_BACKOFF_BASE == 2
