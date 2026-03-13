from __future__ import annotations

import inspect
import json
import logging
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

from ...config import Config
from ...exceptions import APIError, ConfigurationError
from ...rate_limit import is_rate_limit_error
from ...types import TokenUsage
from ..tool_utils import (
    extract_error_code,
    normalize_tool_result,
    should_break_loop,
    update_step_tracking,
)

logger = logging.getLogger(__name__)

REASONING_LOG_TOOL_NAME = "reasoning_log"
REASONING_PREVIEW_LENGTH = 200


class OpenAIResponsesClient:
    def __init__(self, config: Config, request_timeout: Optional[float] = None) -> None:
        self._api_key = config.require_openai_key()
        self._request_timeout = request_timeout

        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ConfigurationError(
                "OpenAI package not installed. Install with: pip install openai"
            ) from e
        except Exception as e:
            raise ConfigurationError(f"Failed to import OpenAI client: {e}") from e

        self._AsyncOpenAI = AsyncOpenAI

    @staticmethod
    def _parse_response_format(response_format: Any) -> Optional[Dict[str, Any]]:
        """Parse response format parameter into OpenAI Responses API format.

        The Responses API expects text.format structure:
        - For json_schema: {"format": {"type": "json_schema", "name": "...", "schema": {...}}}
        - For json_object: {"format": {"type": "json_object"}}

        This method converts from Chat Completions API format if needed.
        """
        if not response_format:
            return None

        if isinstance(response_format, dict):
            # Check if it's Chat Completions API format with nested json_schema
            if response_format.get("type") == "json_schema":
                json_schema = response_format.get("json_schema", {})
                if json_schema:
                    # Transform to Responses API format
                    # name and schema go directly under format, not nested in json_schema
                    return {
                        "format": {
                            "type": "json_schema",
                            "name": json_schema.get("name", "response_schema"),
                            "schema": json_schema.get("schema", {}),
                            "strict": json_schema.get("strict", True),
                        }
                    }
                return {"format": response_format}

            # Already in correct format or other type
            return {"format": response_format}

        if isinstance(response_format, str):
            fmt = response_format.strip().lower()
            if fmt in ("json", "json_object"):
                return {"format": {"type": "json_object"}}
            if fmt in ("markdown", "md"):
                return {"format": {"type": "markdown"}}
            return {"format": {"type": "text"}}

        return None

    @staticmethod
    def _extract_usage(resp: Any) -> Optional[TokenUsage]:
        """Extract token usage from OpenAI response."""
        try:
            usage = getattr(resp, "usage", None)
            if not usage:
                return None
            inp = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0)
            out = getattr(usage, "output_tokens", 0)
            total = getattr(usage, "total_tokens", None) or (inp + out)
            cached = getattr(usage, "cached_tokens", None)
            return TokenUsage(
                input_tokens=int(inp),
                output_tokens=int(out),
                total_tokens=int(total),
                cached_tokens=int(cached) if cached is not None else None,
            )
        except Exception:
            return None

    def _log_cache_usage(self, resp: Any) -> None:
        """Log OpenAI cache usage if available."""
        try:
            usage = getattr(resp, "usage", None)
            if not usage:
                return

            cached_tokens = getattr(usage, "cached_tokens", 0)
            prompt_tokens = getattr(usage, "prompt_tokens", 0)

            if cached_tokens and cached_tokens > 0 and prompt_tokens > 0:
                cache_hit_rate = (cached_tokens / prompt_tokens) * 100
                logger.debug(
                    f"OpenAI cache hit: {cached_tokens}/{prompt_tokens} tokens cached "
                    f"({cache_hit_rate:.1f}%)"
                )
        except Exception as e:
            logger.debug(f"Failed to log cache usage: {e}")

    def _extract_and_log_reasoning(
        self, resp: Any, execute_tool_cb: Optional[Callable[[str, Dict[str, Any]], Any]]
    ) -> str:
        """Extract reasoning text from response and log it."""
        reasoning_text = getattr(resp, "output_text", "") or ""

        if reasoning_text.strip():
            preview = reasoning_text[:REASONING_PREVIEW_LENGTH]
            suffix = "..." if len(reasoning_text) > REASONING_PREVIEW_LENGTH else ""
            logger.debug(
                f"LLM reasoning extracted ({len(reasoning_text)} chars): {preview}{suffix}"
            )
        else:
            logger.debug(
                "No reasoning text found in response output_text "
                "(model might not generate reasoning without reasoning_effort)"
            )

        if execute_tool_cb and hasattr(execute_tool_cb, "_executor"):
            executor = getattr(execute_tool_cb, "_executor", None)
            if executor and hasattr(executor, "set_reasoning"):
                executor.set_reasoning(reasoning_text)

        return reasoning_text

    def _parse_tool_call(self, tc: Any) -> tuple[Optional[str], Optional[str], Dict[str, Any]]:
        """Parse tool call to (id, name, args)."""
        try:
            tc_id = getattr(tc, "id", None)
            name = getattr(tc, "name", None) or getattr(
                getattr(tc, "function", object()), "name", None
            )
            args_raw = getattr(tc, "arguments", None) or getattr(
                getattr(tc, "function", object()), "arguments", "{}"
            )
            args = json.loads(args_raw or "{}")
            return tc_id, name, args
        except Exception as e:
            logger.debug(f"Failed to parse tool call: {e}")
            return getattr(tc, "id", None), getattr(tc, "name", None), {}

    def _get_tool_error_retry_message(
        self,
        tool_calls: List[Any],
        outputs: List[Dict[str, Any]],
        tool_error_callback: Optional[Callable],
    ) -> Optional[str]:
        """Check tool outputs for errors; return retry message if callback provides one."""
        if not tool_error_callback:
            return None
        for tc, out in zip(tool_calls, outputs):
            try:
                payload = json.loads(out["output"]) if isinstance(out.get("output"), str) else {}
            except json.JSONDecodeError:
                payload = {}
            if payload.get("ok"):
                continue
            name = getattr(tc, "name", None) or getattr(
                getattr(tc, "function", object()), "name", None
            )
            msg = tool_error_callback(name or "", extract_error_code(payload), payload)
            if msg:
                return msg
        return None

    def _get_fc_error_retry_message(
        self,
        fc_outputs: List[Dict[str, Any]],
        tool_error_callback: Optional[Callable],
    ) -> Optional[str]:
        """Check function call outputs for errors; return retry message if callback provides one."""
        if not tool_error_callback:
            return None
        for fco in fc_outputs:
            payload = fco.get("payload", {})
            if payload.get("ok"):
                continue
            msg = tool_error_callback(fco.get("name") or "", extract_error_code(payload), payload)
            if msg:
                return msg
        return None

    async def _finalize_empty_response(
        self,
        client: Any,
        params: Dict[str, Any],
        input_list: List[Dict[str, Any]],
    ) -> tuple[str, Optional[TokenUsage]]:
        """Finalize when response is empty; returns (final_text, usage_delta)."""
        finalize_params = dict(params)
        finalize_params.pop("tools", None)
        finalize_params_input = list(input_list) + [
            {
                "role": "system",
                "content": (
                    "Finalize now. Return the final answer in the requested format. "
                    "No further tool calls."
                ),
            }
        ]
        finalize_params["input"] = finalize_params_input
        finalize_resp = await client.responses.create(**finalize_params)
        text = getattr(finalize_resp, "output_text", "") or ""
        usage = self._extract_usage(finalize_resp)
        return text, usage

    async def _execute_tool_with_context(
        self,
        name: Optional[str],
        args: Dict[str, Any],
        reasoning_text: str,
        execute_tool_cb: Optional[Callable[[str, Dict[str, Any]], Any]],
    ) -> Dict[str, Any]:
        """Execute a tool call with optional reasoning context."""
        if execute_tool_cb is None:
            return {
                "ok": False,
                "result": {},
                "error": "no executor provided",
            }

        context = {"reasoning": reasoning_text} if reasoning_text.strip() else None

        try:
            if context and hasattr(execute_tool_cb, "__call__"):
                try:
                    maybe = execute_tool_cb(name, args, context)
                except TypeError:
                    maybe = execute_tool_cb(name, args)
            else:
                maybe = execute_tool_cb(name, args)

            if inspect.isawaitable(maybe) or hasattr(maybe, "__await__"):
                result = await maybe
            else:
                result = maybe

            return normalize_tool_result(result)
        except Exception as e:
            logger.error(f"Tool execution failed for {name}: {e}")
            return {"ok": False, "result": {}, "error": str(e)}

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
        response_format: Optional[Any] = None,
        execute_tool_cb: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        tool_error_callback: Optional[
            Callable[[str, Optional[str], Dict[str, Any]], Optional[str]]
        ] = None,
        max_steps: int = 16,
        max_effective_tool_steps: int = 8,
        force_tool_use: bool = False,
        temperature: Optional[float] = None,
        system_instruct: str = "",
    ) -> tuple[str, TokenUsage]:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if not model or not model.strip():
            raise ValueError("Model name cannot be empty")

        try:
            client_kwargs: Dict[str, Any] = {"api_key": self._api_key}
            if self._request_timeout is not None:
                client_kwargs["timeout"] = self._request_timeout
            client = self._AsyncOpenAI(**client_kwargs)
        except Exception as e:
            raise APIError(f"Failed to initialize OpenAI client: {e}") from e

        # Build input list from conversation history + current prompt
        input_list: List[Dict[str, Any]] = []

        # Add conversation history if provided
        if messages:
            for msg in messages:
                input_list.append(
                    {
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                    }
                )

        # Add current prompt as the latest user message
        input_list.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        base_input: List[Dict[str, Any]] = [dict(m) for m in input_list]

        params: Dict[str, Any] = {
            "model": model,
            "input": input_list,
        }

        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens
        if top_p is not None:
            params["top_p"] = top_p
        if presence_penalty is not None:
            params["presence_penalty"] = presence_penalty
        if instructions:
            params["instructions"] = instructions
        if reasoning_effort:
            params["reasoning"] = {"effort": reasoning_effort}

        if tools_schema:
            params["tools"] = tools_schema

        text_format = self._parse_response_format(response_format)
        if text_format:
            params["text"] = text_format

        last_resp: Optional[Any] = None
        last_nonempty_output: Optional[str] = None
        effective_steps = 0
        consecutive_reasoning_only = 0
        pending_resp: Optional[Any] = None
        total_input = 0
        total_output = 0
        total_cached: Optional[int] = 0

        for step in range(max_steps):
            try:
                if pending_resp is not None:
                    resp = pending_resp
                    pending_resp = None
                else:
                    resp = await client.responses.create(**params)
                last_resp = resp
            except Exception as e:
                if is_rate_limit_error(e):
                    logger.warning(f"OpenAI API rate limit hit at step {step + 1}: {e}")
                    raise
                logger.error(f"OpenAI API call failed at step {step + 1}: {e}")
                raise APIError(f"OpenAI API request failed: {e}") from e

            self._log_cache_usage(resp)
            step_usage = self._extract_usage(resp)
            if step_usage:
                total_input += step_usage.input_tokens
                total_output += step_usage.output_tokens
                if step_usage.cached_tokens is not None:
                    total_cached = (total_cached or 0) + step_usage.cached_tokens

            try:
                txt = getattr(resp, "output_text", "") or ""
                if txt.strip():
                    last_nonempty_output = txt
            except Exception as e:
                logger.debug(f"Failed to extract output text: {e}")

            required_action = getattr(resp, "required_action", None)
            if required_action and getattr(required_action, "type", None) == "submit_tool_outputs":
                submit = getattr(required_action, "submit_tool_outputs", None)
                tool_calls = getattr(submit, "tool_calls", []) if submit else []

                reasoning_text = self._extract_and_log_reasoning(resp, execute_tool_cb)

                outputs = []
                had_non_reasoning_tool = False

                for tc in tool_calls:
                    tc_id, name, args = self._parse_tool_call(tc)
                    result_payload = await self._execute_tool_with_context(
                        name, args, reasoning_text, execute_tool_cb
                    )

                    if (name or "") != REASONING_LOG_TOOL_NAME:
                        had_non_reasoning_tool = True

                    outputs.append({"tool_call_id": tc_id, "output": json.dumps(result_payload)})

                retry_message = self._get_tool_error_retry_message(
                    tool_calls, outputs, tool_error_callback
                )
                if retry_message is not None:
                    logger.info("[TOOL_ERROR] New session (callback returned retry message)")
                    new_input = base_input + [{"role": "user", "content": retry_message}]
                    params["input"] = new_input
                    pending_resp = await client.responses.create(**params)
                    continue

                if outputs:
                    try:
                        resp = await client.responses.submit_tool_outputs(
                            response_id=getattr(resp, "id"),
                            tool_outputs=outputs,
                        )
                        last_resp = resp
                        pending_resp = resp
                    except Exception as e:
                        if is_rate_limit_error(e):
                            logger.warning(
                                f"OpenAI API rate limit hit when submitting tool outputs: {e}"
                            )
                            raise
                        logger.error(f"Failed to submit tool outputs: {e}")
                        raise APIError(f"Failed to submit tool outputs: {e}") from e
                    try:
                        for out in outputs:
                            input_list.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": out.get("tool_call_id"),
                                    "output": out.get("output"),
                                }
                            )
                        params["input"] = input_list
                    except Exception as e:
                        logger.warning(f"Failed to update input list: {e}")

                if getattr(resp, "output_text", ""):
                    break

                effective_steps, consecutive_reasoning_only = update_step_tracking(
                    had_non_reasoning_tool,
                    effective_steps,
                    consecutive_reasoning_only,
                    max_effective_tool_steps,
                )

                if should_break_loop(
                    effective_steps, consecutive_reasoning_only, max_effective_tool_steps
                ):
                    break
            else:
                function_calls = []
                for item in getattr(resp, "output", []) or []:
                    if getattr(item, "type", None) == "function_call":
                        function_calls.append(item)

                if not function_calls:
                    break

                reasoning_text = self._extract_and_log_reasoning(resp, execute_tool_cb)

                output_items = getattr(resp, "output", []) or []
                input_list += output_items
                had_non_reasoning_tool = False
                fc_outputs: List[Dict[str, Any]] = []

                for fc in function_calls:
                    name = getattr(fc, "name", None)
                    try:
                        args = json.loads(getattr(fc, "arguments", "") or "{}")
                    except Exception as e:
                        logger.debug(f"Failed to parse function call arguments: {e}")
                        args = {}

                    if execute_tool_cb is None:
                        continue

                    result_payload = await self._execute_tool_with_context(
                        name, args, reasoning_text, execute_tool_cb
                    )

                    fc_outputs.append({"name": name, "payload": result_payload})

                    if (name or "") != REASONING_LOG_TOOL_NAME:
                        had_non_reasoning_tool = True

                retry_message_fc = self._get_fc_error_retry_message(fc_outputs, tool_error_callback)
                if retry_message_fc is not None:
                    logger.info(
                        "[TOOL_ERROR] New session (callback retry, output-array path)"
                    )
                    # Revert output append to avoid orphaned function_calls
                    for _ in output_items:
                        input_list.pop()
                    new_input = base_input + [{"role": "user", "content": retry_message_fc}]
                    params["input"] = new_input
                    pending_resp = await client.responses.create(**params)
                    continue

                for fc, fco in zip(function_calls, fc_outputs):
                    call_id = getattr(fc, "call_id", None) or getattr(fc, "id", None)
                    if not call_id:
                        logger.warning(
                            "[TOOL_OUTPUT] function_call missing call_id: name=%s",
                            getattr(fc, "name", "?"),
                        )
                    input_list.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps(fco["payload"]),
                        }
                    )

                params["input"] = input_list

                effective_steps, consecutive_reasoning_only = update_step_tracking(
                    had_non_reasoning_tool,
                    effective_steps,
                    consecutive_reasoning_only,
                    max_effective_tool_steps,
                )

                if should_break_loop(
                    effective_steps, consecutive_reasoning_only, max_effective_tool_steps
                ):
                    break

        final_text = getattr(last_resp, "output_text", "") or "" if last_resp else ""
        if not final_text.strip():
            final_text = last_nonempty_output or ""

        if not final_text.strip():
            try:
                final_text, final_usage = await self._finalize_empty_response(
                    client, params, input_list
                )
                if final_usage:
                    total_input += final_usage.input_tokens
                    total_output += final_usage.output_tokens
                    if final_usage.cached_tokens is not None:
                        total_cached = (total_cached or 0) + final_usage.cached_tokens
            except Exception as e:
                if is_rate_limit_error(e):
                    logger.warning(f"OpenAI API rate limit hit during finalization: {e}")
                    raise
                logger.warning(f"Failed to finalize response: {e}")
                if not final_text:
                    raise APIError(f"Failed to generate final response: {e}") from e

        if not final_text.strip():
            raise APIError("Empty response received from OpenAI API")

        usage = TokenUsage(
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output,
            cached_tokens=total_cached if total_cached else None,
        )
        return final_text, usage

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
        """Stream text chunks, then TokenUsage. No tools or response_format support."""
        input_list: List[Dict[str, Any]] = []
        if messages:
            for msg in messages:
                input_list.append(
                    {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                )
        input_list.append({"role": "user", "content": prompt})

        params: Dict[str, Any] = {"model": model, "input": input_list}
        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens
        if top_p is not None:
            params["top_p"] = top_p
        if instructions or system_instruct:
            params["instructions"] = (instructions or system_instruct or "").strip() or None
        if params.get("instructions") is None:
            params.pop("instructions", None)

        try:
            client_kwargs: Dict[str, Any] = {"api_key": self._api_key}
            if self._request_timeout is not None:
                client_kwargs["timeout"] = self._request_timeout
            client = self._AsyncOpenAI(**client_kwargs)
        except Exception as e:
            raise APIError(f"Failed to initialize OpenAI client: {e}") from e

        try:
            stream = client.responses.stream(**params)
            async with stream as s:
                async for event in s:
                    event_type = getattr(event, "type", "") or ""
                    if "output_text" in event_type and "delta" in event_type:
                        delta = getattr(event, "delta", None) or getattr(event, "text", "")
                        if delta:
                            yield delta
                final = s.get_final_response()
                if hasattr(final, "__await__"):
                    final = await final
                usage = self._extract_usage(final)
                if usage:
                    yield usage
                else:
                    yield TokenUsage(0, 0, 0, None)
        except Exception as e:
            if is_rate_limit_error(e):
                raise
            raise APIError(f"OpenAI streaming failed: {e}") from e
