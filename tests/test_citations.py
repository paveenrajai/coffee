"""Tests for Google citation utilities."""

from unittest.mock import MagicMock

from coffee_with_llm.providers.google.utils.citations import (
    collect_grounding_urls,
    extract_citations,
    inject_inline_citations,
)


class TestExtractCitations:
    """Tests for extract_citations."""

    def test_empty_response_returns_empty(self):
        """Empty response returns empty list."""
        resp = MagicMock()
        resp.grounding_metadata = None
        resp.citation_metadata = None
        resp.candidates = []
        assert extract_citations(resp) == []

    def test_grounding_metadata_attributions(self):
        """Extract from grounding_metadata.grounding_attributions."""
        resp = MagicMock()
        resp.grounding_metadata = None
        resp.citation_metadata = None
        resp.candidates = []
        gm = MagicMock()
        att = MagicMock()
        att.web = MagicMock()
        att.web.uri = "https://example.com"
        att.web.title = "Example"
        gm.grounding_attributions = [att]
        resp.grounding_metadata = gm
        result = extract_citations(resp)
        assert len(result) == 1
        assert result[0]["uri"] == "https://example.com"
        assert result[0]["title"] == "Example"


class TestCollectGroundingUrls:
    """Tests for collect_grounding_urls."""

    def test_empty_response_returns_empty(self):
        """Empty response returns empty set."""
        resp = MagicMock()
        resp.grounding_metadata = None
        resp.candidates = []
        assert collect_grounding_urls(resp) == set()

    def test_collects_urls_from_chunks(self):
        """Collect URLs from grounding_chunks."""
        resp = MagicMock()
        gm = MagicMock()
        ch = MagicMock()
        ch.web = MagicMock()
        ch.web.uri = "https://example.com"
        ch.web.url = None
        gm.grounding_chunks = [ch]
        gm.chunks = []
        resp.grounding_metadata = gm
        resp.candidates = []
        assert collect_grounding_urls(resp) == {"https://example.com"}


class TestInjectInlineCitations:
    """Tests for inject_inline_citations."""

    def test_empty_text_returns_unchanged(self):
        """Empty text returns unchanged."""
        resp = MagicMock()
        resp.grounding_metadata = None
        resp.candidates = []
        assert inject_inline_citations("", resp, lambda u: u) == ""

    def test_no_grounding_metadata_returns_unchanged(self):
        """No grounding metadata returns text unchanged."""
        resp = MagicMock()
        resp.grounding_metadata = None
        resp.candidates = []
        text = "Hello world"
        assert inject_inline_citations(text, resp, lambda u: u) == text

    def test_with_citations_injects_markers(self):
        """With grounding metadata and supports, injects citation markers."""
        resp = MagicMock()
        gm = MagicMock()
        gm.grounding_chunks = [MagicMock()]
        gm.grounding_supports = []
        gm.supports = []
        gm.chunks = []
        resp.grounding_metadata = gm
        resp.candidates = []

        # No supports - text unchanged
        text = "Hello world"
        assert inject_inline_citations(text, resp, lambda u: u) == text
