"""Shared types for ask_llm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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
