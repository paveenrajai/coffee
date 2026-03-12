"""Anthropic Claude provider using Messages API with tool use support."""

from __future__ import annotations

import inspect
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from ...exceptions import ConfigurationError, APIError
from ...types import TokenUsage

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_REASONING_ONLY = 3


def _convert_tools_to_anthropic(tools_schema: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style tools to Anthropic format.

    Supports both:
    - OpenAI nested: {"type": "function", "function": {"name", "description", "parameters"}}
    - Flat (tools registry): {"type": "function", "name", "description", "parameters"}
    Anthropic: {"name", "description", "input_schema"} where input_schema is JSON Schema.
    """
    if not tools_schema:
        return []

    anthropic_tools: List[Dict[str, Any]] = []
    for t in tools_schema:
        if t.get("type") == "function":
            fn = t.get("function") or t
            if not isinstance(fn, dict) or "name" not in fn:
                continue
            params = fn.get("parameters", {})
            anthropic_tools.append({
                "name": fn.get("name", "unknown"),
                "description": fn.get("description", ""),
                "input_schema": params if isinstance(params, dict) else {},
            })
        elif "name" in t and "input_schema" in t:
            anthropic_tools.append(t)
        elif "name" in t and "description" in t:
            anthropic_tools.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("input_schema", t.get("parameters", {})),
            })
    return anthropic_tools


class AnthropicMessagesClient:
    """Anthropic Claude client using Messages API with tool use support."""

    def __init__(self) -> None:
        self._api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ConfigurationError("ANTHROPIC_API_KEY environment variable is not set")

        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise ConfigurationError(
                "Anthropic package not installed. Install with: pip install anthropic"
            ) from e
        except Exception as e:
            raise ConfigurationError(f"Failed to import Anthropic client: {e}") from e

        self._AsyncAnthropic = AsyncAnthropic

    @staticmethod
    def _normalize_tool_result(result: Any) -> Dict[str, Any]:
        """Normalize tool execution result to standard format."""
        try:
            if hasattr(result, "ok"):
                return {
                    "ok": bool(getattr(result, "ok", False)),
                    "result": getattr(result, "result", {}),
                    "error": getattr(result, "error", None),
                }
            if isinstance(result, dict):
                return {
                    "ok": bool(result.get("ok", False)),
                    "result": result.get("result", {}),
                    "error": result.get("error", None),
                }
        except Exception as e:
            logger.warning(f"Failed to normalize tool result: {e}")
        return {"ok": False, "result": {}, "error": None}

    @staticmethod
    def _extract_error_code(payload: Dict[str, Any]) -> Optional[str]:
        """Extract tool-defined error_code from payload."""
        result = payload.get("result") or {}
        if isinstance(result, dict):
            code = result.get("error_code")
            if isinstance(code, str) and code:
                return code
        return payload.get("error_code") if isinstance(payload.get("error_code"), str) else None

    async def _execute_tool(
        self,
        name: Optional[str],
        args: Dict[str, Any],
        execute_tool_cb: Optional[Callable[[str, Dict[str, Any]], Any]],
    ) -> Dict[str, Any]:
        """Execute a tool call via callback."""
        if execute_tool_cb is None:
            return {"ok": False, "result": {}, "error": "no executor provided"}

        try:
            maybe = execute_tool_cb(name, args)
            if inspect.isawaitable(maybe) or hasattr(maybe, "__await__"):
                result = await maybe
            else:
                result = maybe
            return self._normalize_tool_result(result)
        except Exception as e:
            logger.error(f"Tool execution failed for {name}: {e}")
            return {"ok": False, "result": {}, "error": str(e)}

    def _update_step_tracking(
        self,
        had_non_reasoning_tool: bool,
        effective_steps: int,
        consecutive_reasoning_only: int,
        max_effective_tool_steps: int,
    ) -> tuple[int, int]:
        """Update step tracking counters."""
        if had_non_reasoning_tool:
            return effective_steps + 1, 0
        return effective_steps, consecutive_reasoning_only + 1

    def _should_break_loop(
        self,
        effective_steps: int,
        consecutive_reasoning_only: int,
        max_effective_tool_steps: int,
    ) -> bool:
        """Check if the generation loop should break."""
        return (
            effective_steps >= max_effective_tool_steps
            or consecutive_reasoning_only >= MAX_CONSECUTIVE_REASONING_ONLY
        )

    def _build_messages(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Build message list from history + prompt."""
        out: List[Dict[str, Any]] = []
        if messages:
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "assistant":
                    out.append({"role": "assistant", "content": content})
                else:
                    out.append({"role": "user", "content": content})
        out.append({"role": "user", "content": prompt})
        return out

    def _content_to_text(self, content: Any) -> str:
        """Extract text from Anthropic response content blocks."""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        texts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
            elif hasattr(block, "type") and getattr(block, "type") == "text":
                texts.append(getattr(block, "text", "") or "")
        return "".join(texts)

    def _blocks_to_api_format(self, content: Any) -> List[Dict[str, Any]]:
        """Convert response content blocks to API request format."""
        if not isinstance(content, list):
            return []
        out: List[Dict[str, Any]] = []
        for block in content:
            if isinstance(block, dict):
                out.append(block)
            elif hasattr(block, "model_dump"):
                out.append(block.model_dump(exclude_none=True))
            else:
                btype = getattr(block, "type", "text")
                if btype == "text":
                    out.append({"type": "text", "text": getattr(block, "text", "") or ""})
                elif btype == "tool_use":
                    inp = getattr(block, "input", {})
                    out.append({
                        "type": "tool_use",
                        "id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                        "input": inp if isinstance(inp, dict) else {},
                    })
        return out

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
        tool_error_callback: Optional[Callable[[str, Optional[str], Dict[str, Any]], Optional[str]]] = None,
        max_steps: int = 16,
        max_effective_tool_steps: int = 8,
        force_tool_use: bool = False,
    ) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if not model or not model.strip():
            raise ValueError("Model name cannot be empty")

        try:
            client = self._AsyncAnthropic(api_key=self._api_key)
        except Exception as e:
            raise APIError(f"Failed to initialize Anthropic client: {e}") from e

        anthropic_tools = _convert_tools_to_anthropic(tools_schema or [])
        base_messages = self._build_messages(prompt, messages)

        params: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or 4096,
            "messages": base_messages,
        }
        if instructions:
            params["system"] = instructions
        if top_p is not None:
            params["top_p"] = top_p
        if anthropic_tools:
            params["tools"] = anthropic_tools
            if force_tool_use:
                params["tool_choice"] = {"type": "any"}
                logger.info("[ANTHROPIC] tool_choice=any (force tool use)")

        logger.info(
            "[ANTHROPIC] Request params keys: %s (tool_choice=%s)",
            list(params.keys()),
            params.get("tool_choice"),
        )

        last_resp: Optional[Any] = None
        last_nonempty_output = ""
        effective_steps = 0
        consecutive_reasoning_only = 0
        pending_resp: Optional[Any] = None
        total_input = 0
        total_output = 0

        for step in range(max_steps):
            try:
                if pending_resp is not None:
                    resp = pending_resp
                    pending_resp = None
                else:
                    resp = await client.messages.create(**params)
                last_resp = resp
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                    logger.warning(f"Anthropic API rate limit hit at step {step + 1}: {e}")
                    raise
                logger.error(f"Anthropic API call failed at step {step + 1}: {e}")
                raise APIError(f"Anthropic API request failed: {e}") from e

            content = getattr(resp, "content", []) or []
            text = self._content_to_text(content)
            if text.strip():
                last_nonempty_output = text

            stop_reason = getattr(resp, "stop_reason", None) or "end_turn"
            block_types = [
                b.get("type", getattr(b, "type", "?")) if isinstance(b, dict) else getattr(b, "type", "?")
                for b in content
            ]
            usage = getattr(resp, "usage", None)
            usage_str = ""
            if usage:
                inp = getattr(usage, "input_tokens", None)
                out = getattr(usage, "output_tokens", None)
                if inp is not None:
                    total_input += inp
                if out is not None:
                    total_output += out
                if inp is not None or out is not None:
                    usage_str = f", usage=input={inp or 0} output={out or 0}"

            logger.info(
                "[ANTHROPIC] step=%d stop_reason=%s block_types=%s%s",
                step + 1,
                stop_reason,
                block_types,
                usage_str,
            )

            if stop_reason != "tool_use" and anthropic_tools and force_tool_use:
                logger.warning(
                    "Expected tool_use but got stop_reason=%s (force_tool_use=True). "
                    "Model may have hit max_tokens or returned text-only.",
                    stop_reason,
                )

            if stop_reason == "tool_use":
                tool_uses = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            tool_uses.append(block)
                    elif hasattr(block, "type") and getattr(block, "type") == "tool_use":
                        tool_uses.append({
                            "id": getattr(block, "id", ""),
                            "name": getattr(block, "name", ""),
                            "input": getattr(block, "input", {}),
                        })

                if not tool_uses:
                    logger.warning(
                        "[ANTHROPIC] stop_reason=tool_use but no tool_use blocks in content "
                        "(block_types=%s). Breaking.",
                        block_types,
                    )
                    break

                tool_names = [
                    t.get("name", "") if isinstance(t, dict) else getattr(t, "name", "")
                    for t in tool_uses
                ]
                logger.info("[ANTHROPIC] Executing tools: %s", tool_names)

                output_payloads: List[Dict[str, Any]] = []
                had_non_reasoning_tool = False

                for tu in tool_uses:
                    tu_id = tu.get("id", "") if isinstance(tu, dict) else getattr(tu, "id", "")
                    name = tu.get("name", "") if isinstance(tu, dict) else getattr(tu, "name", "")
                    inp = tu.get("input", {}) if isinstance(tu, dict) else getattr(tu, "input", {})
                    if isinstance(inp, str):
                        try:
                            inp = json.loads(inp) if inp else {}
                        except json.JSONDecodeError:
                            inp = {}

                    result_payload = await self._execute_tool(name, inp, execute_tool_cb)
                    output_payloads.append({"tool_use_id": tu_id, "name": name, "payload": result_payload})
                    if name and "reasoning" not in name.lower():
                        had_non_reasoning_tool = True

                retry_message: Optional[str] = None
                if tool_error_callback:
                    for out in output_payloads:
                        if out["payload"].get("ok"):
                            continue
                        msg = tool_error_callback(
                            out["name"], self._extract_error_code(out["payload"]), out["payload"]
                        )
                        if msg:
                            retry_message = msg
                            break

                if retry_message is not None:
                    logger.info("[TOOL_ERROR] New session (callback returned retry message)")
                    new_messages = base_messages + [{"role": "user", "content": retry_message}]
                    params["messages"] = new_messages
                    pending_resp = await client.messages.create(**params)
                    continue

                tool_results: List[Dict[str, Any]] = []
                for out in output_payloads:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": out["tool_use_id"],
                        "content": json.dumps(out["payload"]),
                        "is_error": not out["payload"].get("ok", False),
                    })

                new_messages = list(base_messages)
                assistant_content = self._blocks_to_api_format(content)
                new_messages.append({"role": "assistant", "content": assistant_content})
                new_messages.append({"role": "user", "content": tool_results})
                params["messages"] = new_messages
                base_messages = new_messages

                effective_steps, consecutive_reasoning_only = self._update_step_tracking(
                    had_non_reasoning_tool,
                    effective_steps,
                    consecutive_reasoning_only,
                    max_effective_tool_steps,
                )

                if self._should_break_loop(
                    effective_steps, consecutive_reasoning_only, max_effective_tool_steps
                ):
                    logger.info(
                        "[ANTHROPIC] Breaking loop: effective_steps=%d consecutive_reasoning=%d",
                        effective_steps,
                        consecutive_reasoning_only,
                    )
                    break
            else:
                logger.info(
                    "[ANTHROPIC] Breaking: stop_reason=%s (not tool_use)",
                    stop_reason,
                )
                break

        final_text = self._content_to_text(getattr(last_resp, "content", []) or []) if last_resp else ""
        if not final_text.strip():
            final_text = last_nonempty_output or ""

        if not final_text.strip():
            try:
                finalize_params = dict(params)
                finalize_params.pop("tools", None)
                finalize_params["messages"] = base_messages + [
                    {"role": "user", "content": "Finalize now. Return the final answer. No further tool calls."}
                ]
                finalize_resp = await client.messages.create(**finalize_params)
                final_text = self._content_to_text(getattr(finalize_resp, "content", []) or [])
                fu = getattr(finalize_resp, "usage", None)
                if fu:
                    total_input += getattr(fu, "input_tokens", 0) or 0
                    total_output += getattr(fu, "output_tokens", 0) or 0
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                    raise
                logger.warning(f"Failed to finalize response: {e}")
                if not final_text:
                    raise APIError(f"Failed to generate final response: {e}") from e

        if not final_text.strip():
            raise APIError("Empty response received from Anthropic API")

        usage = TokenUsage(
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output,
            cached_tokens=None,
        )
        return final_text, usage
