from __future__ import annotations

import inspect
import json
import logging
import os
import hashlib
import time
from typing import Any, Callable, Dict, List, Optional
from collections import OrderedDict

from google import genai
from google.genai import types
from .utils.citations import (
    inject_inline_citations,
    collect_grounding_urls,
    async_resolve_urls,
)
import httpx

from ...exceptions import ConfigurationError, APIError
from ...types import TokenUsage

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_REASONING_ONLY = 3


# Gemini API rejects these JSON Schema keys; strip them when converting.
_GEMINI_REJECTED_KEYS = frozenset({
    "$defs",
    "$ref",
    "additionalProperties",
    "additional_properties",
})


def _inline_json_schema_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Inline $ref, remove $defs and unsupported keys for Gemini compatibility.

    Gemini's API rejects: $ref, $defs, additionalProperties. This recursively
    resolves #/$defs/X references and strips unsupported fields.
    """
    if not isinstance(schema, dict):
        return schema

    defs_map: Dict[str, Any] = dict(schema.get("$defs", {}) or {})

    def resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj and len(obj) == 1:
                ref = obj["$ref"]
                if isinstance(ref, str) and ref.startswith("#/$defs/"):
                    key = ref.split("/")[-1]
                    if key in defs_map:
                        return resolve(defs_map[key])
                return obj
            return {
                k: resolve(v)
                for k, v in obj.items()
                if k not in _GEMINI_REJECTED_KEYS
            }
        if isinstance(obj, list):
            return [resolve(v) for v in obj]
        return obj

    return resolve(schema)


def _convert_tools_to_gemini(tools_schema: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style tools to Gemini function_declarations format.

    OpenAI: {"type": "function", "function": {"name", "description", "parameters"}}
    Gemini: {"name", "description", "parameters"} (JSON Schema)
    """
    if not tools_schema:
        return []

    gemini_tools: List[Dict[str, Any]] = []
    for t in tools_schema:
        if t.get("type") == "function":
            fn = t.get("function") or t
            if not isinstance(fn, dict) or "name" not in fn:
                continue
            params = fn.get("parameters", {})
            params = _inline_json_schema_refs(params) if isinstance(params, dict) else {}
            gemini_tools.append({
                "name": fn.get("name", "unknown"),
                "description": fn.get("description", ""),
                "parameters": params,
            })
        elif "name" in t and "parameters" in t:
            params = _inline_json_schema_refs(t["parameters"]) if isinstance(t.get("parameters"), dict) else t["parameters"]
            gemini_tools.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": params,
            })
        elif "name" in t and "description" in t:
            raw_params = t.get("parameters", t.get("input_schema", {}))
            params = _inline_json_schema_refs(raw_params) if isinstance(raw_params, dict) else {}
            gemini_tools.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": params,
            })
    return gemini_tools


class GoogleTextClient:
    def __init__(self) -> None:
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ConfigurationError("GOOGLE_API_KEY environment variable is not set")

        try:
            self._client = genai.Client(api_key=api_key)
        except ImportError as e:
            raise ConfigurationError(
                "Google GenAI package not installed. Install with: pip install google-genai"
            ) from e
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Google client: {e}") from e

        self._cached_contexts: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._max_cached_contexts = 10
        self._context_ttl_seconds = 3600

    def _build_config_dict(
        self,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools_schema: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build generation config as dict for Google Gemini API."""
        config_dict: Dict[str, Any] = {}

        is_json_response = (
            response_format
            and isinstance(response_format, dict)
            and response_format.get("type") == "json_schema"
            and response_format.get("json_schema")
        )

        if not is_json_response:
            gemini_decls = _convert_tools_to_gemini(tools_schema or [])
            if gemini_decls:
                config_dict["tools"] = [
                    types.Tool(function_declarations=gemini_decls)
                ]
            elif not tools_schema:
                config_dict["tools"] = [{"google_search": {}}]

        if max_tokens is not None:
            config_dict["max_output_tokens"] = max_tokens
        if temperature is not None:
            config_dict["temperature"] = temperature
        if top_p is not None:
            config_dict["top_p"] = top_p

        if response_format and isinstance(response_format, dict):
            if response_format.get("type") == "json_schema":
                json_schema = response_format.get("json_schema")
                if json_schema:
                    config_dict["response_mime_type"] = "application/json"
                    config_dict["response_json_schema"] = json_schema

        return config_dict

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

    def _get_system_prompt_hash(self, system_instruct: str) -> str:
        """Generate hash for system prompt to use as cache key."""
        return hashlib.sha256(system_instruct.encode()).hexdigest()

    async def _get_or_create_cached_context(
        self,
        system_instruct: str,
        model: str,
    ) -> Optional[str]:
        """Get or create cached context for static system prompt."""
        if not system_instruct or not system_instruct.strip():
            return None

        enable_explicit_cache = (
            os.getenv("GOOGLE_EXPLICIT_CACHE", "1") or "1"
        ).strip() not in ("0", "false", "False")

        if not enable_explicit_cache:
            return None

        model_lower = model.lower()
        if not ("2.5" in model_lower or "gemini-2" in model_lower):
            return None

        prompt_hash = self._get_system_prompt_hash(system_instruct)
        now = time.time()

        if prompt_hash in self._cached_contexts:
            context_name, created_at = self._cached_contexts[prompt_hash]
            if now - created_at < self._context_ttl_seconds:
                self._cached_contexts.move_to_end(prompt_hash)
                return context_name
            else:
                del self._cached_contexts[prompt_hash]

        try:
            static_content = system_instruct

            try:
                cached_context = await self._client.aio.cached_contents.create(
                    model=model,
                    contents=[static_content],
                    ttl=self._context_ttl_seconds,
                )
            except AttributeError:
                try:
                    cached_context = (
                        await self._client.aio.models.cached_contents.create(
                            model=model,
                            contents=[static_content],
                            ttl=self._context_ttl_seconds,
                        )
                    )
                except (AttributeError, Exception):
                    return None

            context_name = getattr(cached_context, "name", None)
            if context_name:
                if len(self._cached_contexts) >= self._max_cached_contexts:
                    self._cached_contexts.popitem(last=False)

                self._cached_contexts[prompt_hash] = (context_name, now)
                self._cached_contexts.move_to_end(prompt_hash)

                logger.debug(f"Google cache created: {context_name[:50]}...")
                return context_name
        except Exception as e:
            logger.debug(f"Google explicit caching unavailable: {e}")
            return None

        return None

    def _extract_function_calls(self, resp: Any) -> List[Dict[str, Any]]:
        """Extract function calls from Gemini response."""
        calls: List[Dict[str, Any]] = []
        candidates = getattr(resp, "candidates", []) or []
        if not candidates:
            return calls
        content = getattr(candidates[0], "content", None)
        if not content:
            return calls
        parts = getattr(content, "parts", []) or []
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc:
                name = getattr(fc, "name", None)
                args = getattr(fc, "args", None) or {}
                if isinstance(args, dict):
                    calls.append({"name": name, "args": args, "part": part})
                else:
                    calls.append({"name": name, "args": {}, "part": part})
        return calls

    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_instruct: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        execute_tool_cb: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        tool_error_callback: Optional[Callable[[str, Optional[str], Dict[str, Any]], Optional[str]]] = None,
        max_steps: int = 16,
        max_effective_tool_steps: int = 8,
        instructions: Optional[str] = None,
        presence_penalty: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        force_tool_use: Optional[bool] = None,
    ) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if not model or not model.strip():
            raise ValueError("Model name cannot be empty")

        system_instruct = system_instruct or (instructions or "")
        use_tools = bool(tools_schema and execute_tool_cb)
        cached_context_name = await self._get_or_create_cached_context(
            system_instruct, model
        )

        def build_initial_contents() -> List[Any]:
            out: List[Any] = []
            if cached_context_name:
                if messages:
                    for msg in messages:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        google_role = "model" if role == "assistant" else "user"
                        out.append({"role": google_role, "parts": [{"text": content}]})
                out.append(prompt)
            else:
                if messages:
                    for i, msg in enumerate(messages):
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        google_role = "model" if role == "assistant" else "user"
                        if i == 0 and role == "user" and system_instruct:
                            content = f"{system_instruct}\n\n{content}"
                        out.append({"role": google_role, "parts": [{"text": content}]})
                    out.append({"role": "user", "parts": [{"text": prompt}]})
                else:
                    merged = (
                        f"{system_instruct}\n\n{prompt}"
                        if (system_instruct or "").strip()
                        else prompt
                    )
                    out.append(merged)
            return out

        contents = build_initial_contents()
        config = self._build_config_dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            response_format=response_format,
            tools_schema=tools_schema if use_tools else None,
        )

        request_kwargs: Dict[str, Any] = {
            "model": model,
            "contents": contents,
            "config": config,
        }
        if cached_context_name:
            request_kwargs["cached_content"] = cached_context_name

        last_resp: Optional[Any] = None
        last_nonempty_output = ""
        effective_steps = 0
        consecutive_reasoning_only = 0
        total_input = 0
        total_output = 0
        total_cached: Optional[int] = None

        for step in range(max_steps):
            try:
                resp = await self._client.aio.models.generate_content(**request_kwargs)
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                    raise
                logger.error(f"Google API call failed: {e}")
                raise APIError(f"Google API request failed: {e}") from e

            last_resp = resp
            text = str(getattr(resp, "text", None) or getattr(resp, "output_text", "") or "")
            if text.strip():
                last_nonempty_output = text

            um = getattr(resp, "usage_metadata", None)
            if um:
                total_input += getattr(um, "prompt_token_count", 0) or 0
                total_output += getattr(um, "candidates_token_count", 0) or 0
                cc = getattr(um, "cached_content_token_count", None)
                if cc is not None:
                    total_cached = (total_cached or 0) + cc

            if not use_tools:
                break

            function_calls = self._extract_function_calls(resp)
            if not function_calls:
                break

            output_payloads: List[Dict[str, Any]] = []
            had_non_reasoning_tool = False

            for fc in function_calls:
                name = fc.get("name")
                args = fc.get("args") or {}
                result_payload = await self._execute_tool(name, args, execute_tool_cb)
                output_payloads.append({"name": name, "payload": result_payload})
                if name and "reasoning" not in (name or "").lower():
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
                contents = build_initial_contents() + [{"role": "user", "parts": [{"text": retry_message}]}]
                request_kwargs["contents"] = contents
                continue

            model_content = getattr(last_resp.candidates[0], "content", None) if last_resp and getattr(last_resp, "candidates", None) else None
            if model_content is not None:
                contents.append(model_content)

            response_parts: List[Any] = []
            for out in output_payloads:
                part = types.Part.from_function_response(
                    name=out["name"],
                    response=out["payload"],
                )
                response_parts.append(part)

            contents.append(types.Content(role="user", parts=response_parts))
            request_kwargs["contents"] = contents

            effective_steps = effective_steps + 1 if had_non_reasoning_tool else effective_steps
            consecutive_reasoning_only = 0 if had_non_reasoning_tool else consecutive_reasoning_only + 1

            if effective_steps >= max_effective_tool_steps or consecutive_reasoning_only >= MAX_CONSECUTIVE_REASONING_ONLY:
                break

        text = str(getattr(last_resp, "text", None) or getattr(last_resp, "output_text", "") or "") if last_resp else ""
        if not text.strip():
            text = last_nonempty_output or ""

        if not text.strip():
            raise APIError("Empty response received from Google API")

        try:
            enable_inline = (
                os.getenv("GOOGLE_INLINE_CITATIONS", "1") or "1"
            ).strip() not in ("0", "false", "False")
            if enable_inline and last_resp:
                urls = collect_grounding_urls(last_resp)
                if urls:
                    async with httpx.AsyncClient(
                        follow_redirects=True, timeout=2
                    ) as http:
                        resolved = await async_resolve_urls(
                            urls, http, max_concurrency=4
                        )
                    text = inject_inline_citations(
                        text,
                        last_resp,
                        resolve_url=lambda u: resolved.get(u, u),
                    )
        except Exception as e:
            logger.debug(f"Failed to inject citations: {e}")

        usage = TokenUsage(
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output,
            cached_tokens=total_cached if total_cached else None,
        )
        return text, usage
