from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

from .config import Config
from .cost import estimate_cost
from .exceptions import APIError, ConfigurationError, RateLimitError, ValidationError
from .providers.registry import get_provider
from .rate_limit import is_rate_limit_error, with_retry
from .types import AskResult, StreamResult, TokenUsage

logger = logging.getLogger(__name__)


class AskLLM:
    """
    Model-agnostic LLM interface supporting OpenAI, Anthropic Claude, and Google Gemini.

    Automatically selects the appropriate provider based on the model name.
    Provides a unified API for both providers with parameter normalization.

    Example:
        ```python
        from coffee import AskLLM

        llm = AskLLM(model="gpt-5.4")
        result = await llm.ask(
            prompt="What is Python?",
            system_instruct="You are a helpful assistant."
        )
        print(result.text)
        print(result.usage.input_tokens, result.usage.output_tokens)
        ```
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        config: Optional[Config] = None,
        min_delay_between_calls: float = 1.0,
        max_retries: int = 3,
        request_timeout: Optional[float] = None,
        google_explicit_cache: bool = True,
        google_inline_citations: bool = True,
    ) -> None:
        """
        Initialize AskLLM with a model.

        Args:
            model: Model name (e.g., "gpt-5.4", "claude-sonnet-4-6", "gemini-3.1-pro-preview")
                 Provider is auto-detected based on model prefix.
                 Must be provided.
            config: Config instance. If None, uses Config.from_env() (API keys from env).
            min_delay_between_calls: Min delay in seconds between API calls (default: 1.0)
            max_retries: Maximum number of retries for rate limit errors (default: 3)
            request_timeout: Request timeout in seconds (default: 60)
            google_explicit_cache: Enable Google context caching for Gemini (default: True)
            google_inline_citations: Inject [cite: url] for Gemini grounding (default: True)

        Raises:
            ValidationError: If model is not provided.
            ConfigurationError: If API keys are missing or client initialization fails.
        """
        if not model:
            raise ValidationError("Model name is required")

        self._model = model
        self._min_delay = min_delay_between_calls
        self._max_retries = max_retries
        self._last_call_time: Optional[float] = None

        cfg = (config or Config.from_env()).with_request_timeout(request_timeout)
        self._request_timeout = cfg.request_timeout

        try:
            self._client = get_provider(
                model,
                config=cfg,
                request_timeout=self._request_timeout,
                google_explicit_cache=google_explicit_cache,
                google_inline_citations=google_inline_citations,
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize client for model '{model}': {e}") from e

    async def ask(
        self,
        *,
        prompt: str,
        system_instruct: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
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
        stream: bool = False,
    ) -> Union[AskResult, StreamResult]:
        """
        Ask the LLM a question.

        Args:
            prompt: User prompt/question (appended to messages if provided)
            system_instruct: System instruction/prompt
            messages: Optional history (list of {"role": "user"|"assistant", "content": "..."})
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            presence_penalty: Presence penalty (OpenAI only)
            reasoning_effort: Reasoning effort level (OpenAI only, e.g., "low", "medium", "high")
            tools_schema: Tool/function calling schema (OpenAI, Anthropic, Google)
            response_format: Response format specification (JSON schema, etc.)
            execute_tool_cb: Callback for executing tools (OpenAI, Anthropic, Google)
            tool_error_callback: When tool returns ok=False, (tool_name, error_code, payload).
                Return str to start new session; None to feed error back.
            max_steps: Maximum tool-calling steps (OpenAI, Anthropic, Google)
            max_effective_tool_steps: Maximum effective tool steps (OpenAI, Anthropic, Google)
            force_tool_use: When True, force at least one tool call (Anthropic tool_choice=any).
                Prevents text-only responses when tools are provided.
            stream: When True, return StreamResult (async iterable; usage after iteration).
                Not supported with tools_schema or response_format.

        Returns:
            AskResult with text and token usage, or StreamResult when stream=True.

        Raises:
            ValidationError: If prompt is empty or invalid parameters provided.
            APIError: If the API call fails.
        """
        if not prompt or not prompt.strip():
            raise ValidationError("Prompt cannot be empty")

        if max_tokens is not None and max_tokens <= 0:
            raise ValidationError("max_tokens must be positive")

        if temperature is not None and (temperature < 0 or temperature > 2):
            raise ValidationError("temperature must be between 0 and 2")

        if top_p is not None and (top_p < 0 or top_p > 1):
            raise ValidationError("top_p must be between 0 and 1")

        if max_steps <= 0:
            raise ValidationError("max_steps must be positive")

        if max_effective_tool_steps <= 0:
            raise ValidationError("max_effective_tool_steps must be positive")

        if stream and (tools_schema or response_format):
            raise ValidationError(
                "stream=True is not supported with tools_schema or response_format"
            )

        # Rate limiting: wait if needed before making API call
        await self._wait_if_needed()

        if stream:
            return self._ask_stream(
                prompt=prompt,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                system_instruct=system_instruct,
            )

        async def _generate() -> AskResult:
            result = await self._client.generate(
                prompt=prompt,
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                top_p=top_p,
                presence_penalty=presence_penalty,
                instructions=system_instruct if system_instruct else None,
                reasoning_effort=reasoning_effort,
                tools_schema=tools_schema,
                response_format=response_format,
                execute_tool_cb=execute_tool_cb,
                tool_error_callback=tool_error_callback,
                max_steps=max_steps,
                max_effective_tool_steps=max_effective_tool_steps,
                force_tool_use=force_tool_use,
                temperature=temperature,
                system_instruct=system_instruct or "",
            )
            text, usage = (
                result if isinstance(result, tuple) else (result, TokenUsage(0, 0, 0, None))
            )
            return AskResult(text=text, usage=self._usage_with_cost(usage))

        try:
            return await with_retry(
                _generate,
                max_retries=self._max_retries,
            )
        except Exception as e:
            if isinstance(e, (ValidationError, ConfigurationError, RateLimitError)):
                raise
            if is_rate_limit_error(e):
                logger.error(
                    "Rate limit exceeded for model '%s' after %d attempts",
                    self._model,
                    self._max_retries,
                )
                raise RateLimitError(
                    f"Rate limit exceeded after {self._max_retries} retries: {e}"
                ) from e
            logger.error(f"API call failed for model '{self._model}': {e}")
            raise APIError(f"Failed to generate response: {e}") from e

    def _ask_stream(
        self,
        *,
        prompt: str,
        messages: Optional[List[Dict[str, Any]]],
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        system_instruct: str,
    ) -> StreamResult:
        """Stream text chunks with usage and rate-limit retry."""

        def create_stream() -> AsyncIterator[Union[str, TokenUsage]]:
            return self._client.generate_stream(
                prompt=prompt,
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature,
                instructions=system_instruct if system_instruct else None,
                system_instruct=system_instruct or "",
            )

        return StreamResult(
            stream_factory=create_stream,
            usage_callback=lambda u: self._usage_with_cost(u),
            max_retries=self._max_retries,
        )

    def _usage_with_cost(self, usage: TokenUsage) -> TokenUsage:
        """Add cost_usd to usage."""
        cost = estimate_cost(usage, self._model)
        return TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_tokens=usage.cached_tokens,
            cost_usd=cost,
        )

    async def _wait_if_needed(self) -> None:
        """Wait if needed to maintain minimum delay between calls."""
        if self._last_call_time is not None:
            elapsed = time.perf_counter() - self._last_call_time
            if elapsed < self._min_delay:
                wait_time = self._min_delay - elapsed
                await asyncio.sleep(wait_time)
        self._last_call_time = time.perf_counter()
