"""Rate limit detection and retry with exponential backoff."""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Awaitable, Callable, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

RATE_LIMIT_BACKOFF_BASE = 2  # seconds; backoff = BASE ** attempt

# Known rate limit exception types from provider SDKs
_RATE_LIMIT_EXCEPTIONS: Tuple[Type[Exception], ...] = ()

try:
    from openai import RateLimitError as OpenAIRateLimitError

    _RATE_LIMIT_EXCEPTIONS = (*_RATE_LIMIT_EXCEPTIONS, OpenAIRateLimitError)
except ImportError:
    pass

try:
    from anthropic import RateLimitError as AnthropicRateLimitError

    _RATE_LIMIT_EXCEPTIONS = (*_RATE_LIMIT_EXCEPTIONS, AnthropicRateLimitError)
except ImportError:
    pass

try:
    from httpx import HTTPStatusError

    _HTTP_STATUS_ERROR: Type[HTTPStatusError] | None = HTTPStatusError
except ImportError:
    _HTTP_STATUS_ERROR = None

_STRING_INDICATORS = (
    "429",
    "rate limit",
    "too many requests",
    "ratelimit",
    "quota exceeded",
    "quota",
)


def is_rate_limit_error(error: Exception) -> bool:
    """Check if exception is a rate limit error."""
    # 1. Known SDK exception types
    if isinstance(error, _RATE_LIMIT_EXCEPTIONS):
        return True

    # 2. HTTPStatusError with 429 (used by httpx-based clients)
    if _HTTP_STATUS_ERROR is not None and isinstance(error, _HTTP_STATUS_ERROR):
        if hasattr(error, "response") and getattr(error.response, "status_code", None) == 429:
            return True

    # 3. Fallback: string matching
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    return any(ind in error_str for ind in _STRING_INDICATORS) or any(
        ind in error_type for ind in _STRING_INDICATORS
    )


T = TypeVar("T")


async def with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    is_retryable: Callable[[Exception], bool] = is_rate_limit_error,
) -> T:
    """Run async operation with exponential backoff on retryable errors."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return await operation()
        except Exception as e:
            last_exc = e
            if is_retryable(e) and attempt < max_retries - 1:
                backoff = RATE_LIMIT_BACKOFF_BASE**attempt
                logger.warning(
                    "Retryable error (attempt %d/%d), backing off %ds: %s",
                    attempt + 1,
                    max_retries,
                    backoff,
                    e,
                )
                await asyncio.sleep(backoff)
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("with_retry: unreachable")


async def retry_stream(
    stream_factory: Callable[[], AsyncIterator[object]],
    *,
    max_retries: int = 3,
) -> AsyncIterator[object]:
    """Wrap async stream, retrying on rate limit before each chunk. Yields items from stream."""
    stream = stream_factory()
    it = stream.__aiter__()
    while True:
        for attempt in range(max_retries):
            try:
                item = await it.__anext__()
                yield item
                break
            except StopAsyncIteration:
                return
            except Exception as e:
                if is_rate_limit_error(e) and attempt < max_retries - 1:
                    backoff = RATE_LIMIT_BACKOFF_BASE**attempt
                    logger.warning(
                        "Stream rate limit (attempt %d/%d), backing off %ds: %s",
                        attempt + 1,
                        max_retries,
                        backoff,
                        e,
                    )
                    await asyncio.sleep(backoff)
                    stream = stream_factory()
                    it = stream.__aiter__()
                    continue
                raise
