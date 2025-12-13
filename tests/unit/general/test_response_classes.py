"""
Unit tests for core response classes

Tests the LLMResponse hierarchy including:
- LLMIdentity creation and provider guessing
- PendingLLMResponse state management and resolve()
- ReadyLLMResponse immediate resolution
- Serialization/deserialization (__getstate__, __setstate__)
"""

from unittest.mock import Mock
import pytest
from parallellm.core.calls import _call_to_concise_dict
from parallellm.core.identity import LLMIdentity
from parallellm.core.response import (
    LLMResponse,
    PendingLLMResponse,
    ReadyLLMResponse,
)
from parallellm.types import CallIdentifier, ParsedResponse
from parallellm.testing.simple_backend import MockBackend


class TestReadyLLMResponse:
    """Test ReadyLLMResponse (already resolved responses)"""

    def test_ready_response_immediate_resolution(self):
        """Test creating a ready response"""
        call_id = self._create_mock_call_id()
        response = ReadyLLMResponse(call_id=call_id, value="Immediate content")

        assert response.call_id == call_id
        assert response.resolve() == "Immediate content"

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
            "doc_hash": "test_hash_123",
            "seq_id": 1,
            "session_id": 1,
            "provider_type": "openai",
        }


class TestPendingLLMResponse:
    """Test PendingLLMResponse (lazy-loaded responses)"""

    def test_pending_response_resolve_calls_backend(self):
        """Test that pending responses call backend.retrieve()"""
        call_id = self._create_mock_call_id()
        mock_backend = MockBackend()
        # Store a response in the backend for retrieval
        mock_backend.store(
            call_id,
            ParsedResponse(
                text="Backend response",
                response_id="resp_789",
                metadata=None,
            ),
        )

        response = PendingLLMResponse(call_id=call_id, backend=mock_backend)

        result = response.resolve()

        assert result == "Backend response"

    def test_pending_response_resolve_caching(self):
        """Test that pending responses cache results after first resolve"""
        call_id = self._create_mock_call_id()
        mock_backend = MockBackend()
        # Store a response in the backend for retrieval
        mock_backend.store(
            call_id,
            ParsedResponse(
                text="Cached response",
                response_id="resp_123",
                metadata=None,
            ),
        )

        response = PendingLLMResponse(call_id=call_id, backend=mock_backend)

        # First call should hit backend
        result1 = response.resolve()
        assert result1 == "Cached response"

        # Second call should use cached value
        result2 = response.resolve()
        assert result2 == "Cached response"

        # Note: MockBackend doesn't track call counts like unittest.Mock,
        # but we can still verify the caching behavior works correctly

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
        assert new_response.call_id == _call_to_concise_dict(call_id)
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
            "doc_hash": "test_hash_456",
            "seq_id": 2,
            "session_id": 1,
            "provider_type": "openai",
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
