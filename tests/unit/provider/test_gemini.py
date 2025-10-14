"""
Unit tests for Gemini provider classes

Tests the Gemini provider functionality including:
- SyncGeminiProvider and AsyncGeminiProvider
- Document formatting for Gemini API
- Provider type handling
"""

from dotenv import load_dotenv
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
from parallellm.provider.schemas import guess_schema
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

    # @pytest.mark.skip("costs real money")
    def test_sync_gemini_provider_prepare_sync_call(self, call_id):
        """Test SyncGeminiProvider prepares sync callable correctly"""
        load_dotenv()
        from google import genai

        client = genai.Client()
        provider = SyncGeminiProvider(client=client)

        # Test prepare_sync_call returns a callable
        sync_callable = provider.prepare_sync_call(
            instructions="Please respond exactly with Hello World!",
            documents=["Test document"],
            llm=Mock(model_name="gemini-2.5-flash"),
        )

        # Verify it returns a callable
        assert callable(sync_callable)

        # Execute the callable
        result = sync_callable()
        parsed, _, _ = guess_schema(result)

        # Verify result
        assert parsed == "Hello World!"
