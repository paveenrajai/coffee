"""Tests for Config."""

from unittest.mock import patch

import pytest

from coffee_with_llm import Config
from coffee_with_llm.exceptions import ConfigurationError


class TestConfigFromEnv:
    """Tests for Config.from_env."""

    def test_loads_api_keys_from_env(self):
        """Config loads API keys from environment."""
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "sk-test",
                "ANTHROPIC_API_KEY": "sk-ant-test",
                "GOOGLE_API_KEY": "test-google-key",
            },
            clear=False,
        ):
            cfg = Config.from_env()
            assert cfg.openai_api_key == "sk-test"
            assert cfg.anthropic_api_key == "sk-ant-test"
            assert cfg.google_api_key == "test-google-key"

    def test_request_timeout_from_env(self):
        """COFFEE_REQUEST_TIMEOUT env is parsed."""
        with patch.dict("os.environ", {"COFFEE_REQUEST_TIMEOUT": "120"}, clear=False):
            cfg = Config.from_env()
            assert cfg.request_timeout == 120.0

    def test_request_timeout_invalid_falls_back_to_60(self):
        """Invalid COFFEE_REQUEST_TIMEOUT falls back to 60."""
        with patch.dict("os.environ", {"COFFEE_REQUEST_TIMEOUT": "invalid"}, clear=False):
            cfg = Config.from_env()
            assert cfg.request_timeout == 60.0

    def test_request_timeout_zero_falls_back_to_60(self):
        """COFFEE_REQUEST_TIMEOUT=0 falls back to 60."""
        with patch.dict("os.environ", {"COFFEE_REQUEST_TIMEOUT": "0"}, clear=False):
            cfg = Config.from_env()
            assert cfg.request_timeout == 60.0

    def test_request_timeout_over_600_falls_back_to_60(self):
        """COFFEE_REQUEST_TIMEOUT > 600 falls back to 60."""
        with patch.dict("os.environ", {"COFFEE_REQUEST_TIMEOUT": "1000"}, clear=False):
            cfg = Config.from_env()
            assert cfg.request_timeout == 60.0


class TestConfigWithRequestTimeout:
    """Tests for Config.with_request_timeout."""

    def test_overrides_timeout(self):
        """with_request_timeout overrides timeout."""
        cfg = Config(openai_api_key="x", request_timeout=60.0)
        new_cfg = cfg.with_request_timeout(120.0)
        assert new_cfg.request_timeout == 120.0
        assert cfg.request_timeout == 60.0

    def test_none_returns_self(self):
        """with_request_timeout(None) returns same config."""
        cfg = Config(openai_api_key="x", request_timeout=60.0)
        new_cfg = cfg.with_request_timeout(None)
        assert new_cfg is cfg


class TestConfigRequireKeys:
    """Tests for Config require_* methods."""

    def test_require_openai_key_raises_when_missing(self):
        """require_openai_key raises when missing."""
        cfg = Config(openai_api_key=None)
        with pytest.raises(ConfigurationError, match="OpenAI.*not configured"):
            cfg.require_openai_key()

    def test_require_openai_key_returns_when_present(self):
        """require_openai_key returns key when present."""
        cfg = Config(openai_api_key="sk-test")
        assert cfg.require_openai_key() == "sk-test"

    def test_require_anthropic_key_raises_when_missing(self):
        """require_anthropic_key raises when missing."""
        cfg = Config(anthropic_api_key=None)
        with pytest.raises(ConfigurationError, match="Anthropic.*not configured"):
            cfg.require_anthropic_key()

    def test_require_anthropic_key_returns_when_present(self):
        """require_anthropic_key returns key when present."""
        cfg = Config(anthropic_api_key="sk-ant-test")
        assert cfg.require_anthropic_key() == "sk-ant-test"

    def test_require_google_key_raises_when_missing(self):
        """require_google_key raises when missing."""
        cfg = Config(google_api_key=None)
        with pytest.raises(ConfigurationError, match="Google.*not configured"):
            cfg.require_google_key()

    def test_require_google_key_returns_when_present(self):
        """require_google_key returns key when present."""
        cfg = Config(google_api_key="test-google-key")
        assert cfg.require_google_key() == "test-google-key"
