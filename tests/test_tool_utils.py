"""Tests for shared tool_utils."""

from unittest.mock import MagicMock

from coffee_with_llm.providers.tool_utils import (
    extract_error_code,
    normalize_tool_result,
    should_break_loop,
    update_step_tracking,
)


class TestNormalizeToolResult:
    """Tests for normalize_tool_result."""

    def test_object_with_ok_attribute(self):
        """Object with ok, result, error attributes."""
        obj = MagicMock()
        obj.ok = True
        obj.result = {"x": 1}
        obj.error = None
        assert normalize_tool_result(obj) == {"ok": True, "result": {"x": 1}, "error": None}

    def test_dict_input(self):
        """Dict input is normalized."""
        assert normalize_tool_result({"ok": False, "result": {}, "error": "err"}) == {
            "ok": False,
            "result": {},
            "error": "err",
        }

    def test_invalid_input_returns_default(self):
        """Invalid input returns default error format."""
        assert normalize_tool_result("invalid") == {"ok": False, "result": {}, "error": None}


class TestExtractErrorCode:
    """Tests for extract_error_code."""

    def test_from_result_nested(self):
        """Extract from payload.result.error_code."""
        assert extract_error_code({"result": {"error_code": "E1"}}) == "E1"

    def test_from_top_level(self):
        """Extract from payload.error_code."""
        assert extract_error_code({"error_code": "E2"}) == "E2"

    def test_missing_returns_none(self):
        """Missing error_code returns None."""
        assert extract_error_code({}) is None
        assert extract_error_code({"result": {}}) is None


class TestUpdateStepTracking:
    """Tests for update_step_tracking."""

    def test_had_non_reasoning_tool_increments_effective(self):
        """Non-reasoning tool increments effective_steps, resets consecutive."""
        e, c = update_step_tracking(True, 1, 2, 8)
        assert e == 2
        assert c == 0

    def test_reasoning_only_increments_consecutive(self):
        """Reasoning-only keeps effective, increments consecutive."""
        e, c = update_step_tracking(False, 1, 2, 8)
        assert e == 1
        assert c == 3


class TestShouldBreakLoop:
    """Tests for should_break_loop."""

    def test_effective_steps_reached(self):
        """Break when effective_steps >= max."""
        assert should_break_loop(8, 0, 8) is True
        assert should_break_loop(9, 0, 8) is True

    def test_consecutive_reasoning_reached(self):
        """Break when consecutive_reasoning_only >= 3."""
        assert should_break_loop(0, 3, 8) is True

    def test_continue_when_under_limits(self):
        """Continue when under both limits."""
        assert should_break_loop(2, 1, 8) is False
