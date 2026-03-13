"""Pytest configuration and shared fixtures."""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_openai_api_key():
    """Mock OpenAI API key."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}):
        yield


@pytest.fixture
def mock_google_api_key():
    """Mock Google API key."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-google-key"}):
        yield


@pytest.fixture
def mock_both_api_keys(mock_openai_api_key, mock_google_api_key):
    """Mock both API keys."""
    yield


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI AsyncOpenAI client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Test response"
    mock_response.usage = MagicMock()
    mock_response.usage.cached_tokens = 0
    mock_response.usage.prompt_tokens = 10

    async def mock_generate(*args, **kwargs):
        return mock_response

    mock_client.responses.create = mock_generate

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_google_client():
    """Mock Google GenAI client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Test response"

    async def mock_generate(*args, **kwargs):
        return mock_response

    mock_client.models.generate_content = mock_generate

    with patch(
        "coffee_with_llm.providers.google.text_client.genai.Client", return_value=mock_client
    ):
        yield mock_client
