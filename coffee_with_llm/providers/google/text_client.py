from __future__ import annotations

import hashlib
import inspect
import logging
import time
from collections import OrderedDict
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

import httpx
from google import genai
from google.genai import types

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
from .utils.citations import (
    async_resolve_urls,
    collect_grounding_urls,
    inject_inline_citations,
)

logger = logging.getLogger(__name__)

MAX_CACHED_CONTEXTS = 10
CONTEXT_TTL_SECONDS = 3600  # 1 hour

# Gemini API rejects these JSON Schema keys; strip them when converting.
_GEMINI_REJECTED_KEYS = frozenset(
    {
        "$defs",
        "$ref",
        "additionalProperties",
        "additional_properties",
    }
)


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
            return {k: resolve(v) for k, v in obj.items() if k not in _GEMINI_REJECTED_KEYS}
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
            gemini_tools.append(
                {
                    "name": fn.get("name", "unknown"),
                    "description": fn.get("description", ""),
                    "parameters": params,
                }
            )
        elif "name" in t and "parameters" in t:
            params = (
                _inline_json_schema_refs(t["parameters"])
                if isinstance(t.get("parameters"), dict)
                else t["parameters"]
            )
            gemini_tools.append(
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": params,
                }
            )
        elif "name" in t and "description" in t:
            raw_params = t.get("parameters", t.get("input_schema", {}))
            params = _inline_json_schema_refs(raw_params) if isinstance(raw_params, dict) else {}
            gemini_tools.append(
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": params,
                }
            )
    return gemini_tools


class GoogleTextClient:
    def __init__(
        self,
        config: Config,
        request_timeout: Optional[float] = None,
        google_explicit_cache: bool = True,
        google_inline_citations: bool = True,
    ) -> None:
        self._api_key = config.require_google_key()
        self._google_explicit_cache = google_explicit_cache
        self._google_inline_citations = google_inline_citations

        try:
            self._client = genai.Client(api_key=self._api_key)
        except ImportError as e:
            raise ConfigurationError(
                "Google GenAI package not installed. Install with: pip install google-genai"
            ) from e
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Google client: {e}") from e

        self._request_timeout = request_timeout
        self._cached_contexts: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._max_cached_contexts = MAX_CACHED_CONTEXTS
        self._context_ttl_seconds = CONTEXT_TTL_SECONDS

    def _build_config_dict(
        self,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        include_google_search: bool = True,
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
                config_dict["tools"] = [types.Tool(function_declarations=gemini_decls)]
            elif include_google_search and not tools_schema:
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
            return normalize_tool_result(result)
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

        if not self._google_explicit_cache:
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
                    cached_context = await self._client.aio.models.cached_contents.create(
                        model=model,
                        contents=[static_content],
                        ttl=self._context_ttl_seconds,
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

    def _get_tool_error_retry_message(
        self,
        output_payloads: List[Dict[str, Any]],
        tool_error_callback: Optional[
            Callable[[str, Optional[str], Dict[str, Any]], Optional[str]]
        ],
    ) -> Optional[str]:
        """Check tool outputs for errors; return retry message if callback provides one."""
        if not tool_error_callback:
            return None
        for out in output_payloads:
            if out["payload"].get("ok"):
                continue
            msg = tool_error_callback(
                out["name"], extract_error_code(out["payload"]), out["payload"]
            )
            if msg:
                return msg
        return None

    def _build_initial_contents(
        self,
        cached_context_name: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
        prompt: str,
        system_instruct: str,
    ) -> List[Any]:
        """Build initial contents for Gemini API request."""
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
                    f"{system_instruct}\n\n{prompt}" if (system_instruct or "").strip() else prompt
                )
                out.append(merged)
        return out

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

        system_instruct = system_instruct or (instructions or "")
        use_tools = bool(tools_schema and execute_tool_cb)
        cached_context_name = await self._get_or_create_cached_context(system_instruct, model)

        def build_initial_contents() -> List[Any]:
            return self._build_initial_contents(
                cached_context_name, messages, prompt, system_instruct
            )

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
                if is_rate_limit_error(e):
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

            retry_message = self._get_tool_error_retry_message(output_payloads, tool_error_callback)
            if retry_message is not None:
                contents = build_initial_contents() + [
                    {"role": "user", "parts": [{"text": retry_message}]}
                ]
                request_kwargs["contents"] = contents
                continue

            model_content = (
                getattr(last_resp.candidates[0], "content", None)
                if last_resp and getattr(last_resp, "candidates", None)
                else None
            )
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

            effective_steps, consecutive_reasoning_only = update_step_tracking(
                had_non_reasoning_tool,
                effective_steps,
                consecutive_reasoning_only,
                max_effective_tool_steps,
            )

            if should_break_loop(
                effective_steps,
                consecutive_reasoning_only,
                max_effective_tool_steps,
            ):
                break

        text = (
            str(getattr(last_resp, "text", None) or getattr(last_resp, "output_text", "") or "")
            if last_resp
            else ""
        )
        if not text.strip():
            text = last_nonempty_output or ""

        if not text.strip():
            raise APIError("Empty response received from Google API")

        try:
            if self._google_inline_citations and last_resp:
                urls = collect_grounding_urls(last_resp)
                if urls:
                    async with httpx.AsyncClient(follow_redirects=True, timeout=2) as http:
                        resolved = await async_resolve_urls(urls, http, max_concurrency=4)
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
        system_instruct = system_instruct or (instructions or "")
        contents: List[Any] = []
        if messages:
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                google_role = "model" if role == "assistant" else "user"
                contents.append({"role": google_role, "parts": [{"text": content}]})
            contents.append({"role": "user", "parts": [{"text": prompt}]})
        else:
            merged = f"{system_instruct}\n\n{prompt}".strip() if system_instruct.strip() else prompt
            contents.append(merged)

        config = self._build_config_dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            response_format=None,
            tools_schema=None,
            include_google_search=False,
        )

        try:
            stream = self._client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )
            last_chunk = None
            async for chunk in stream:
                last_chunk = chunk
                text = getattr(chunk, "text", None) or ""
                if text:
                    yield text
            um = getattr(last_chunk, "usage_metadata", None) if last_chunk else None
            if um:
                inp = getattr(um, "prompt_token_count", 0) or 0
                out = getattr(um, "candidates_token_count", 0) or 0
                cached = getattr(um, "cached_content_token_count", None)
                yield TokenUsage(inp, out, inp + out, int(cached) if cached is not None else None)
            else:
                yield TokenUsage(0, 0, 0, None)
        except Exception as e:
            if is_rate_limit_error(e):
                raise
            raise APIError(f"Google streaming failed: {e}") from e
