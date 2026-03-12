"""
Token usage cost estimation.

Prices are per 1M tokens in USD. Cached tokens use discounted rate when available.
Sources: openai.com/api/pricing, ai.google.dev/pricing, anthropic.com/pricing
"""

from __future__ import annotations

from typing import Optional, Tuple

from .types import TokenUsage

# (input_per_1m, output_per_1m, cached_per_1m or None) USD
# Order: most specific prefix first
_MODEL_PRICING: list[Tuple[str, float, float, Optional[float]]] = [
    # OpenAI
    ("gpt-5.4", 2.50, 15.00, 0.25),
    ("gpt-5.2", 1.75, 14.00, None),
    ("gpt-5-mini", 0.25, 2.00, 0.025),
    ("gpt-5-nano", 0.05, 0.40, None),
    ("gpt-4o-mini", 0.15, 0.60, None),
    ("gpt-4o", 2.50, 10.00, 1.25),
    # Anthropic
    ("claude-sonnet-4-6", 3.00, 15.00, None),
    ("claude-opus-4", 5.00, 25.00, None),
    ("claude-haiku", 1.00, 5.00, None),
    ("claude-3-5-sonnet", 3.00, 15.00, None),
    # Google - most specific first
    ("gemini-3.1-pro-preview", 2.00, 12.00, 0.20),
    ("gemini-3.1-flash-lite-preview", 0.25, 1.50, 0.025),
    ("gemini-3.1-flash", 0.50, 3.00, 0.05),
    ("gemini-3-flash", 0.50, 3.00, 0.05),
    ("gemini-2.5-pro", 1.25, 10.00, 0.125),
    ("gemini-2.5-flash", 0.30, 2.50, 0.03),
    ("gemini-2.5-flash-lite", 0.10, 0.40, 0.01),
    ("gemini-2.0-flash", 0.15, 0.60, None),
    ("gemini-flash-lite-latest", 0.10, 0.40, 0.01),  # alias for 2.5-flash-lite
    ("gemini-flash-lite", 0.10, 0.40, 0.01),
    ("gemini-flash", 0.30, 2.50, 0.03),
    ("gemini-pro", 1.25, 10.00, 0.125),
]


def _get_pricing(model: str) -> Optional[Tuple[float, float, Optional[float]]]:
    """Return (input_per_1m, output_per_1m, cached_per_1m) for model or None."""
    m = (model or "").lower()
    for prefix, inp, out, cached in _MODEL_PRICING:
        if m.startswith(prefix):
            return (inp, out, cached)
    return None


def estimate_cost(usage: TokenUsage, model: str) -> Optional[float]:
    """
    Estimate cost in USD from token usage.

    Args:
        usage: TokenUsage from AskResult
        model: Model name (e.g. gpt-5-nano, gemini-3.1-pro-preview)

    Returns:
        Estimated cost in USD, or None if model pricing unknown
    """
    pricing = _get_pricing(model)
    if not pricing:
        return None

    inp_per_1m, out_per_1m, cached_per_1m = pricing

    # Non-cached input: input_tokens - cached_tokens
    cached = usage.cached_tokens or 0
    uncached_input = max(0, usage.input_tokens - cached)

    cost = 0.0
    cost += (uncached_input / 1_000_000) * inp_per_1m
    cost += (usage.output_tokens / 1_000_000) * out_per_1m
    if cached > 0 and cached_per_1m is not None:
        cost += (cached / 1_000_000) * cached_per_1m
    elif cached > 0:
        # Fallback: cached at input rate (conservative)
        cost += (cached / 1_000_000) * inp_per_1m

    return round(cost, 6)
