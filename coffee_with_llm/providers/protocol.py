"""Provider protocol – common interface for all LLM providers."""

from __future__ import annotations

from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from ..types import TokenUsage


@runtime_checkable
class ProviderProtocol(Protocol):
    """Protocol for LLM providers. Implement generate() and generate_stream()."""

    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        instructions: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        execute_tool_cb: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        tool_error_callback: Optional[
            Callable[[str, Optional[str], Dict[str, Any]], Optional[str]]
        ] = None,
        max_steps: int = 24,
        max_effective_tool_steps: int = 12,
        force_tool_use: bool = False,
        temperature: Optional[float] = None,
        system_instruct: str = "",
    ) -> tuple[str, TokenUsage]:
        """Generate text. Returns (text, usage)."""
        ...

    async def generate_stream(
        self,
        *,
        prompt: str,
        model: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        instructions: Optional[str] = None,
        system_instruct: str = "",
    ) -> AsyncIterator[Union[str, TokenUsage]]:
        """Stream text chunks, then TokenUsage as final yield. No tools or response_format."""
        ...
