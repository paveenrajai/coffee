"""Tests for exception classes."""

import pytest

from coffee_with_llm.exceptions import (
    APIError,
    AskLLMError,
    ConfigurationError,
    ValidationError,
)


class TestAskLLMError:
    """Tests for base exception class."""

    def test_base_exception(self):
        """Test that base exception can be raised."""
        with pytest.raises(AskLLMError):
            raise AskLLMError("Test error")

    def test_exception_message(self):
        """Test exception message."""
        error = AskLLMError("Test message")
        assert str(error) == "Test message"


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_inherits_from_base(self):
        """Test that ConfigurationError inherits from AskLLMError."""
        error = ConfigurationError("Config error")
        assert isinstance(error, AskLLMError)

    def test_can_be_raised(self):
        """Test that ConfigurationError can be raised."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Configuration error")


class TestAPIError:
    """Tests for APIError."""

    def test_inherits_from_base(self):
        """Test that APIError inherits from AskLLMError."""
        error = APIError("API error")
        assert isinstance(error, AskLLMError)

    def test_can_be_raised(self):
        """Test that APIError can be raised."""
        with pytest.raises(APIError):
            raise APIError("API error")


class TestValidationError:
    """Tests for ValidationError."""

    def test_inherits_from_base(self):
        """Test that ValidationError inherits from AskLLMError."""
        error = ValidationError("Validation error")
        assert isinstance(error, AskLLMError)

    def test_can_be_raised(self):
        """Test that ValidationError can be raised."""
        with pytest.raises(ValidationError):
            raise ValidationError("Validation error")
