"""Shared types for coffee."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Callable, Optional, Union

from .rate_limit import retry_stream


@dataclass(frozen=True)
class TokenUsage:
    """Token usage for a generation session (aggregated across multi-step/tool loops)."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: Optional[int] = None
    cost_usd: Optional[float] = None


@dataclass
class AskResult:
    """Result of an LLM ask with token usage."""

    text: str
    usage: TokenUsage

    def __str__(self) -> str:
        return self.text


class StreamResult:
    """
    Result of streaming. Iterate to get text chunks; usage is populated after iteration completes.

    Note: Must be iterated via ``async for`` (which calls ``__aiter__`` before ``__anext__``).
    Re-iteration creates a fresh stream; ``usage`` reflects the most recent completed iteration.
    """

    def __init__(
        self,
        stream_factory: Callable[[], AsyncIterator[Union[str, TokenUsage]]],
        usage_callback: Optional[Callable[[TokenUsage], TokenUsage]] = None,
        max_retries: int = 3,
    ) -> None:
        self._stream_factory = stream_factory
        self._usage_callback = usage_callback
        self._max_retries = max_retries
        self._usage: Optional[TokenUsage] = None
        self._iter: Optional[AsyncIterator[Union[str, TokenUsage]]] = None

    def __aiter__(self) -> StreamResult:
        self._iter = retry_stream(
            self._stream_factory,
            max_retries=self._max_retries,
        ).__aiter__()
        return self

    async def __anext__(self) -> str:
        if self._iter is None:
            raise RuntimeError(
                "StreamResult must be iterated via async for; __aiter__ was not called"
            )
        item = await self._iter.__anext__()
        if isinstance(item, TokenUsage):
            self._usage = self._usage_callback(item) if self._usage_callback else item
            raise StopAsyncIteration
        return item

    @property
    def usage(self) -> Optional[TokenUsage]:
        """Token usage, populated after stream iteration completes."""
        return self._usage
