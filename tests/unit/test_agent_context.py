"""
Unit tests for AgentContext class

Tests the core agent functionality including:
- Context manager behavior (__enter__, __exit__)
- Checkpoint management (when_checkpoint, goto_checkpoint)
- Counter systems (anonymous vs checkpoint modes)
- ask_llm method and cache integration
- Exception handling and checkpoint transitions
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from parallellm.core.agent.agent import AgentContext, AgentDashboardContext
from parallellm.core.exception import NotAvailable, WrongCheckpoint, GotoCheckpoint
from parallellm.core.response import ReadyLLMResponse, PendingLLMResponse
from parallellm.types import CallIdentifier


class TestAgentContextBasics:
    """Test basic AgentContext functionality"""

    def test_context_manager_enter_exit(self, mock_orchestrator):
        """Test context manager protocol"""
        agent = AgentContext("test_agent", mock_orchestrator)

        # Should work without exception
        with agent:
            pass

    def test_exit_with_parallellm_exceptions(self, mock_orchestrator):
        """Test __exit__ handles ParalleLLM exceptions correctly"""
        agent = AgentContext("test_agent", mock_orchestrator)

        suppressed_exceptions = [NotAvailable, WrongCheckpoint, GotoCheckpoint]

        for exc_type in suppressed_exceptions:
            with agent:
                raise exc_type("test exception")
        # Exceptions should have been suppressed

    def test_exit_with_other_exceptions(self, mock_orchestrator):
        """Test __exit__ doesn't suppress other exceptions"""
        agent = AgentContext("test_agent", mock_orchestrator)

        # Other exceptions should not be suppressed
        with pytest.raises(ValueError):
            with agent:
                raise ValueError("non-ParalleLLM exception")

        with pytest.raises(AssertionError):
            with agent:
                raise AssertionError("Should not suppress asserts!")


def test_counter_independence(mock_orchestrator):
    """Test that anonymous and checkpoint counters are independent"""
    agent = AgentContext("test_agent", mock_orchestrator)

    with agent:
        agent.ask_llm("anonymous 1")
        agent.ask_llm("anonymous 2")
        assert agent._anonymous_counter == 2

    # Switch to checkpoint mode
    with agent:
        agent.when_checkpoint("checkpoint_1")
        agent.ask_llm("checkpoint 1")
        agent.ask_llm("checkpoint 2")
        agent.ask_llm("checkpoint 3")
        assert agent._checkpoint_counter == 3
        assert agent._anonymous_counter == 2  # Should remain unchanged


class TestAskLLMMethod:
    """Test the ask_llm method functionality"""

    def test_ask_llm_basic_call(self, mock_orchestrator):
        """Test basic ask_llm call"""
        agent = AgentContext("test_agent", mock_orchestrator)

        with agent:
            response = agent.ask_llm("Test prompt")

            assert isinstance(response, (ReadyLLMResponse, PendingLLMResponse))
            # Now backend.submit_query is called instead of provider.submit_query_to_provider
            mock_orchestrator._backend.submit_query.assert_called_once()

    def test_ask_llm_with_cache_hit(self, mock_orchestrator):
        """Test ask_llm when response is cached"""
        mock_orchestrator._backend.retrieve.return_value = "Cached response"

        agent = AgentContext("test_agent", mock_orchestrator)

        with agent:
            response = agent.ask_llm("Cached prompt")

            assert isinstance(response, ReadyLLMResponse)
            assert response.resolve() == "Cached response"
            # Backend submit_query should not be called for cached responses
            mock_orchestrator._backend.submit_query.assert_not_called()

    def test_ask_llm_call_id_generation(self, mock_orchestrator):
        """Test that ask_llm generates correct call IDs"""
        agent = AgentContext("test_agent", mock_orchestrator)

        with agent:
            agent.ask_llm("Test prompt")

            # Verify backend was called with correct call_id structure
            call_args = mock_orchestrator._backend.submit_query.call_args
            call_id = call_args.kwargs["call_id"]

            assert call_id["agent_name"] == "test_agent"
            assert call_id["checkpoint"] is None  # Anonymous mode
            assert call_id["seq_id"] == 0  # First call
            assert call_id["session_id"] == 1
            assert call_id["provider_type"] == "openai"

    def test_ask_llm_checkpoint_call_id(self, mock_orchestrator):
        """Test call ID generation in checkpoint mode"""
        agent = AgentContext("test_agent", mock_orchestrator)

        with agent:
            agent.when_checkpoint("test_checkpoint")
            agent.ask_llm("Checkpoint prompt")

            call_args = mock_orchestrator._backend.submit_query.call_args
            call_id = call_args.kwargs["call_id"]

            assert call_id["checkpoint"] == "test_checkpoint"
            assert call_id["seq_id"] == 0  # First checkpoint call

    @patch("parallellm.core.agent.agent.compute_hash")
    def test_ask_llm_hash_computation(self, mock_compute_hash, mock_orchestrator):
        """Test that ask_llm computes hashes correctly"""
        mock_compute_hash.return_value = "test_hash_123"

        agent = AgentContext("test_agent", mock_orchestrator)

        with agent:
            agent.ask_llm("Test prompt", instructions="Test instructions")

            mock_compute_hash.assert_called_once_with(
                "Test instructions", ["Test prompt"]
            )

    def test_context_manager_checkpoint_cleanup(self, mock_orchestrator):
        """Test that context manager properly cleans up checkpoint state"""
        agent = AgentContext("test_agent", mock_orchestrator)

        with agent:
            agent.when_checkpoint("test_checkpoint")
            agent.ask_llm("checkpoint prompt")

            # Verify checkpoint is active during context
            assert agent.active_checkpoint == "test_checkpoint"
            assert agent._checkpoint_counter is not None

        # After context exit, checkpoint should be cleaned up
        assert agent.active_checkpoint is None
        assert agent._checkpoint_counter is None

    def test_context_manager_preserves_anonymous_counter(self, mock_orchestrator):
        """Test that context manager preserves anonymous counter across contexts"""
        agent = AgentContext("test_agent", mock_orchestrator)

        with agent:
            agent.ask_llm("first call")
            agent.ask_llm("second call")
            assert agent._anonymous_counter == 2

        # Second context block - counter should continue
        with agent:
            agent.ask_llm("third call")
            assert agent._anonymous_counter == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
