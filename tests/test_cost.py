"""Tests for cost module."""

from coffee_with_llm.cost import estimate_cost
from coffee_with_llm.types import TokenUsage


class TestEstimateCost:
    """Tests for estimate_cost."""

    def test_known_model_returns_cost(self):
        usage = TokenUsage(1000, 500, 1500, None)
        cost = estimate_cost(usage, "gpt-4o-mini")
        assert cost is not None
        assert cost > 0

    def test_unknown_model_returns_none(self):
        usage = TokenUsage(1000, 500, 1500, None)
        cost = estimate_cost(usage, "unknown-model-xyz")
        assert cost is None

    def test_cached_tokens_use_discounted_rate(self):
        usage_uncached = TokenUsage(1000, 100, 1100, None)
        usage_cached = TokenUsage(1000, 100, 1100, 500)
        cost_uncached = estimate_cost(usage_uncached, "gpt-4o")
        cost_cached = estimate_cost(usage_cached, "gpt-4o")
        assert cost_cached is not None
        assert cost_uncached is not None
        assert cost_cached < cost_uncached

    def test_zero_usage(self):
        usage = TokenUsage(0, 0, 0, None)
        cost = estimate_cost(usage, "gpt-4o-mini")
        assert cost == 0.0

    def test_model_prefix_match(self):
        usage = TokenUsage(1000, 100, 1100, None)
        cost = estimate_cost(usage, "gpt-4o-mini-2024-01")
        assert cost is not None
