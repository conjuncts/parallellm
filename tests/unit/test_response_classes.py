"""
Unit tests for core response classes

Tests the LLMResponse hierarchy including:
- LLMIdentity creation and provider guessing
- PendingLLMResponse state management and resolve()
- ReadyLLMResponse immediate resolution
- Serialization/deserialization (__getstate__, __setstate__)
"""

import pytest
from unittest.mock import Mock, MagicMock
from parallellm.core.identity import LLMIdentity
from parallellm.core.response import (
    LLMResponse,
    PendingLLMResponse,
    ReadyLLMResponse,
)
from parallellm.types import CallIdentifier


class TestLLMIdentity:
    """Test LLM identity creation and provider handling"""

    def test_identity_creation_basic(self):
        """Test basic identity creation"""
        identity = LLMIdentity("gpt-4")
        assert identity.identity == "gpt-4"
        assert identity.provider == "openai"  # Should be guessed

    def test_identity_with_explicit_provider(self):
        """Test identity with explicit provider"""
        identity = LLMIdentity("claude-3", provider="anthropic")
        assert identity.identity == "claude-3"
        assert identity.provider == "anthropic"

    def test_to_str_openai_default(self):
        """Test string conversion with OpenAI defaults"""
        identity = LLMIdentity("gpt-4")
        assert identity.to_str("openai") == "gpt-4"

    def test_to_str_with_none_identity(self):
        """Test string conversion when identity is None"""
        identity = LLMIdentity(None)
        assert identity.to_str("openai") == "gpt-4.1-nano"  # Default

    def test_to_str_no_provider_specified(self):
        """Test string conversion using identity's provider"""
        identity = LLMIdentity("custom-model", provider="custom")
        assert identity.to_str() == "custom-model"


class TestReadyLLMResponse:
    """Test ReadyLLMResponse (already resolved responses)"""

    def test_ready_response_creation(self):
        """Test creating a ready response"""
        call_id = self._create_mock_call_id()
        response = ReadyLLMResponse(call_id=call_id, value="Test response content")

        assert response.call_id == call_id
        assert response.resolve() == "Test response content"

    def test_ready_response_immediate_resolution(self):
        """Test that ready responses resolve immediately"""
        call_id = self._create_mock_call_id()
        response = ReadyLLMResponse(call_id=call_id, value="Immediate result")

        # Should return value immediately without backend calls
        result = response.resolve()
        assert result == "Immediate result"

    def test_ready_response_serialization(self):
        """Test ready response can be serialized"""
        call_id = self._create_mock_call_id()
        response = ReadyLLMResponse(call_id=call_id, value="Serializable content")

        # Test __getstate__
        state = response.__getstate__()
        assert "call_id" in state
        assert "value" not in state

        # Test __setstate__
        # ReadyLLMResponse needs the help of the backend to retrieve the true value

    def _create_mock_call_id(self) -> CallIdentifier:
        """Helper to create mock call identifiers"""
        return {
            "agent_name": "test_agent",
            "checkpoint": None,
            "doc_hash": "test_hash_123",
            "seq_id": 1,
            "session_id": 1,
            "provider_type": "openai",
        }


class TestPendingLLMResponse:
    """Test PendingLLMResponse (lazy-loaded responses)"""

    def test_pending_response_creation(self):
        """Test creating a pending response"""
        call_id = self._create_mock_call_id()
        mock_backend = Mock()

        response = PendingLLMResponse(call_id=call_id, backend=mock_backend)

        assert response.call_id == call_id
        assert response._backend == mock_backend

    def test_pending_response_resolve_calls_backend(self):
        """Test that pending responses call backend.retrieve()"""
        call_id = self._create_mock_call_id()
        mock_backend = Mock()
        mock_backend.retrieve.return_value = "Backend response"

        response = PendingLLMResponse(call_id=call_id, backend=mock_backend)

        result = response.resolve()

        assert result == "Backend response"
        mock_backend.retrieve.assert_called_once_with(call_id)

    def test_pending_response_resolve_caching(self):
        """Test that pending responses cache results after first resolve"""
        call_id = self._create_mock_call_id()
        mock_backend = Mock()
        mock_backend.retrieve.return_value = "Cached response"

        response = PendingLLMResponse(call_id=call_id, backend=mock_backend)

        # First call should hit backend
        result1 = response.resolve()
        assert result1 == "Cached response"

        # Second call should use cached value
        result2 = response.resolve()
        assert result2 == "Cached response"

        # Backend should only be called once
        assert mock_backend.retrieve.call_count == 1

    def test_pending_response_serialization(self):
        """Test pending response serialization handling"""
        call_id = self._create_mock_call_id()
        mock_backend = Mock()

        response = PendingLLMResponse(call_id=call_id, backend=mock_backend)

        # Test __getstate__ removes backend
        state = response.__getstate__()
        assert "call_id" in state
        assert "_backend" not in state  # Should be excluded

        # Test __setstate__ restores call_id but backend is None
        new_response = PendingLLMResponse.__new__(PendingLLMResponse)
        new_response.__setstate__(state)
        assert new_response.call_id == call_id
        assert new_response._backend is None

    def test_pending_response_backend_none_handling(self):
        """Test behavior when backend is None (after deserialization)"""
        call_id = self._create_mock_call_id()

        response = PendingLLMResponse(
            call_id=call_id,
            backend=None,  # Simulates post-deserialization state
        )

        # Should handle None backend gracefully
        # (The exact behavior depends on implementation)
        with pytest.raises((AttributeError, RuntimeError)):
            response.resolve()

    def _create_mock_call_id(self) -> CallIdentifier:
        """Helper to create mock call identifiers"""
        return {
            "agent_name": "test_agent",
            "checkpoint": "test_checkpoint",
            "doc_hash": "test_hash_456",
            "seq_id": 2,
            "session_id": 1,
            "provider_type": "openai",
        }


class TestLLMResponseHierarchy:
    """Test interactions between different response types"""

    def test_response_type_consistency(self):
        """Test that all response types implement the same interface"""
        call_id = self._create_mock_call_id()
        mock_backend = Mock()
        mock_backend.retrieve.return_value = "Test content"

        # All types should have resolve() method
        ready = ReadyLLMResponse(call_id=call_id, value="Ready content")
        pending = PendingLLMResponse(call_id=call_id, backend=mock_backend)

        assert hasattr(ready, "resolve")
        assert hasattr(pending, "resolve")
        assert callable(ready.resolve)
        assert callable(pending.resolve)

        # All should return strings
        assert isinstance(ready.resolve(), str)
        assert isinstance(pending.resolve(), str)

    def test_call_id_structure(self):
        """Test that call IDs have required fields"""
        call_id = self._create_mock_call_id()

        required_fields = {
            "agent_name",
            "checkpoint",
            "doc_hash",
            "seq_id",
            "session_id",
            "provider_type",
        }

        assert all(field in call_id for field in required_fields)
        assert isinstance(call_id["seq_id"], int)
        assert isinstance(call_id["session_id"], int)
        assert isinstance(call_id["doc_hash"], str)

    def _create_mock_call_id(self) -> CallIdentifier:
        """Helper to create mock call identifiers"""
        return {
            "agent_name": "hierarchy_test",
            "checkpoint": None,
            "doc_hash": "hierarchy_hash",
            "seq_id": 999,
            "session_id": 1,
            "provider_type": "openai",
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
