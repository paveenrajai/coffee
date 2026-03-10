from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .exceptions import ConfigurationError, ValidationError, APIError, RateLimitError
from .providers.anthropic import AnthropicMessagesClient
from .providers.google import GoogleTextClient
from .providers.openai import OpenAIResponsesClient

logger = logging.getLogger(__name__)


class AskLLM:
    """
    Model-agnostic LLM interface supporting OpenAI, Anthropic Claude, and Google Gemini.

    Automatically selects the appropriate provider based on the model name.
    Provides a unified API for both providers with parameter normalization.

    Example:
        ```python
        from ask_llm import AskLLM

        llm = AskLLM(model="gpt-4o-mini")
        response = await llm.ask(
            prompt="What is Python?",
            system_instruct="You are a helpful assistant."
        )
        ```
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        min_delay_between_calls: float = 1.0,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize AskLLM with a model.

        Args:
            model: Model name (e.g., "gpt-4o-mini", "claude-sonnet-4-6", "gemini-2.0-flash-exp")
                 Provider is auto-detected based on model prefix.
                 Must be provided.
            min_delay_between_calls: Minimum delay in seconds between consecutive API calls (default: 1.0)
            max_retries: Maximum number of retries for rate limit errors (default: 3)

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
        model_lower = model.lower()

        try:
            if model_lower.startswith("claude") or model_lower.startswith("anthropic"):
                self._client = AnthropicMessagesClient()
            elif model_lower.startswith("gemini") or model_lower.startswith("google"):
                self._client = GoogleTextClient()
            else:
                self._client = OpenAIResponsesClient()
        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize client for model '{model}': {e}"
            ) from e

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
        tool_error_callback: Optional[Callable[[str, Optional[str], Dict[str, Any]], Optional[str]]] = None,
        max_steps: int = 24,
        max_effective_tool_steps: int = 12,
        force_tool_use: bool = False,
    ) -> str:
        """
        Ask the LLM a question.

        Args:
            prompt: User prompt/question (appended to messages if provided)
            system_instruct: System instruction/prompt
            messages: Optional conversation history (list of {"role": "user"|"assistant", "content": "..."})
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            presence_penalty: Presence penalty (OpenAI only)
            reasoning_effort: Reasoning effort level (OpenAI only, e.g., "low", "medium", "high")
            tools_schema: Tool/function calling schema (OpenAI only)
            response_format: Response format specification (JSON schema, etc.)
            execute_tool_cb: Callback for executing tools (OpenAI only)
            tool_error_callback: When tool returns ok=False, called with (tool_name, error_code, payload).
                error_code from payload (tool-defined). Return str to start new session with that message;
                None to feed error back. (OpenAI only)
            max_steps: Maximum tool-calling steps (OpenAI only)
            max_effective_tool_steps: Maximum effective tool steps (OpenAI only)
            force_tool_use: When True, force at least one tool call (Anthropic tool_choice=any).
                Prevents text-only responses when tools are provided.

        Returns:
            Generated text response

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

        # Rate limiting: wait if needed before making API call
        await self._wait_if_needed()

        # Retry logic for rate limit errors
        last_exception: Optional[Exception] = None
        
        for attempt in range(self._max_retries):
            try:
                if isinstance(
                    self._client,
                    (OpenAIResponsesClient, AnthropicMessagesClient, GoogleTextClient),
                ):
                    result = self._client.generate(
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
                    )
                    if hasattr(result, "__await__"):
                        result = await result
                    return result

                return await self._client.generate(
                    prompt=prompt,
                    model=self._model,
                    system_instruct=system_instruct or "",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    response_format=response_format,
                )
            except Exception as e:
                last_exception = e
                
                # Check if it's a rate limit error
                if self._is_rate_limit_error(e):
                    if attempt < self._max_retries - 1:
                        # Exponential backoff: 2^attempt seconds
                        backoff = 2 ** attempt
                        logger.warning(
                            f"Rate limit hit for model '{self._model}', "
                            f"retrying in {backoff}s (attempt {attempt + 1}/{self._max_retries})"
                        )
                        await asyncio.sleep(backoff)
                        # Reset last call time to allow retry
                        self._last_call_time = None
                        continue
                    else:
                        logger.error(
                            f"Rate limit exceeded for model '{self._model}' after {self._max_retries} attempts"
                        )
                        raise RateLimitError(
                            f"Rate limit exceeded after {self._max_retries} retries: {e}"
                        ) from e
                
                # Not a rate limit error or validation/configuration error
                if isinstance(e, (ValidationError, ConfigurationError)):
                    raise
                
                # For other errors, raise immediately (don't retry)
                logger.error(f"API call failed for model '{self._model}': {e}")
                raise APIError(f"Failed to generate response: {e}") from e
        
        # Should never reach here, but just in case
        if last_exception:
            raise APIError(f"Failed to generate response after {self._max_retries} attempts: {last_exception}") from last_exception
        raise APIError("Failed to generate response: unknown error")
    
    async def _wait_if_needed(self) -> None:
        """Wait if needed to maintain minimum delay between calls."""
        if self._last_call_time is not None:
            elapsed = time.perf_counter() - self._last_call_time
            if elapsed < self._min_delay:
                wait_time = self._min_delay - elapsed
                await asyncio.sleep(wait_time)
        self._last_call_time = time.perf_counter()
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an exception is a rate limit error."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Check for common rate limit indicators
        rate_limit_indicators = [
            "429",
            "rate limit",
            "too many requests",
            "ratelimit",
            "quota exceeded",
            "quota",
        ]
        
        return any(indicator in error_str for indicator in rate_limit_indicators) or any(
            indicator in error_type for indicator in rate_limit_indicators
        )
