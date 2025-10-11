"""
Unit tests for Gemini provider classes

Tests the Gemini provider functionality including:
- SyncGeminiProvider and AsyncGeminiProvider
- Document formatting for Gemini API
- Provider type handling
"""

import pytest
from unittest.mock import Mock, AsyncMock
from parallellm.provider.gemini import (
    SyncGeminiProvider,
    AsyncGeminiProvider,
    _fix_docs_for_gemini,
    GeminiProvider,
)
from parallellm.core.backend.sync_backend import SyncBackend
from parallellm.core.backend.async_backend import AsyncBackend
from parallellm.core.response import PendingLLMResponse, ReadyLLMResponse
from parallellm.types import CallIdentifier


@pytest.fixture
def call_id() -> CallIdentifier:
    """Fixture to create mock call identifiers for testing"""
    return {
        "agent_name": "test_agent",
        "checkpoint": "test_checkpoint",
        "doc_hash": "test_hash",
        "seq_id": 1,
        "session_id": 1,
        "provider_type": "google",
    }


class TestGeminiDocumentFormatting:
    """Test document formatting for Gemini API"""

    def test_fix_docs_single_string(self):
        """Test fixing a single string document"""
        result = _fix_docs_for_gemini("Hello world")

        assert result == "Hello world"  # Single strings are passed directly

    def test_fix_docs_list_of_strings(self):
        """Test fixing a list of string documents"""
        docs = ["First message", "Second message"]
        result = _fix_docs_for_gemini(docs)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == "First message"
        assert result[1] == "Second message"


class TestGeminiProviders:
    """Test Gemini provider classes"""

    def test_provider_type(self):
        """Test that GeminiProvider has correct provider type"""
        assert GeminiProvider.provider_type == "google"

    def test_sync_gemini_provider_submit_query(self, call_id):
        """Test SyncGeminiProvider query submission"""
        # Mock client and backend
        mock_client = Mock()
        mock_models = Mock()
        mock_client.models = mock_models

        mock_response = Mock()
        mock_response.text = "Test response from Gemini"
        mock_models.generate_content.return_value = mock_response

        mock_backend = Mock(spec=SyncBackend)
        mock_backend.submit_sync_call.return_value = (
            "Test response from Gemini",
            None,
            {},
        )

        # Create provider
        provider = SyncGeminiProvider(client=mock_client, backend=mock_backend)

        # Submit query
        response = provider.submit_query_to_provider(
            instructions="Test instructions",
            documents=["Test document"],
            call_id=call_id,
        )

        # Verify response
        assert isinstance(response, ReadyLLMResponse)
        assert response.call_id == call_id
        assert response.value == "Test response from Gemini"

        # Verify backend was called
        mock_backend.submit_sync_call.assert_called_once()
        assert mock_backend.submit_sync_call.call_args[0][0] == call_id
