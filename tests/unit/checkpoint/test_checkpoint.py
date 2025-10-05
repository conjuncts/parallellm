import pytest
from parallellm.core.agent.agent import AgentContext
from parallellm.core.exception import GotoCheckpoint


class TestCheckpointManagement:
    """Test checkpoint functionality"""

    def test_checkpoint_entry(self, mock_orchestrator):
        """Test when_checkpoint enters the correct checkpoint"""
        # Pre-populate agent metadata in the orchestrator
        mock_orchestrator._fm.metadata["agents"]["test_agent"] = {
            "checkpoint_counter": 5,
            "latest_checkpoint": "active-checkpoint",
        }

        agent = AgentContext("test_agent", mock_orchestrator)

        with agent:
            agent.when_checkpoint("inactive-checkpoint")
            assert False  # Should not reach here

        with agent:
            agent.when_checkpoint("active-checkpoint")

            assert agent.active_checkpoint == "active-checkpoint"
            assert agent._checkpoint_counter == 5  # Loaded from metadata

    def test_goto_checkpoint(self, mock_orchestrator):
        """Test goto_checkpoint initializes counter for new checkpoints"""
        # The agent metadata will be empty initially (no previous checkpoints)
        agent = AgentContext("test_agent", mock_orchestrator)
        mock_orchestrator._fm.metadata["agents"]["test_agent"] = {
            "checkpoint_counter": 5,
            "latest_checkpoint": "active",
        }

        with agent:
            agent.when_checkpoint("next")
            assert False

        with agent:
            agent.when_checkpoint("active")
            agent.ask_llm("Some question")
            agent.goto_checkpoint("next")
            assert False

        successes = 0
        with agent:
            agent.when_checkpoint("next")
            successes += 1
        assert successes == 1

    def test_checkpoint_counter(self, mock_orchestrator):
        """Test goto_checkpoint initializes counter for new checkpoints"""
        # The agent metadata will be empty initially (no previous checkpoints)
        agent = AgentContext("test_agent", mock_orchestrator)
        mock_orchestrator._fm.metadata["agents"]["test_agent"] = {
            "checkpoint_counter": 5,
            "latest_checkpoint": "active",
        }

        successes = 0
        with agent:
            agent.when_checkpoint("active")
            assert agent._checkpoint_counter == 5

            agent.ask_llm("Some question")
            assert agent._checkpoint_counter == 6

            successes += 1
            agent.goto_checkpoint("next")
            assert False

        with agent:
            agent.when_checkpoint("next")

            # Checkpoint counter should resume from previous
            assert agent._checkpoint_counter == 6
            successes += 1
        assert successes == 2

    def test_exit_checkpoint_clears_state(self, mock_orchestrator):
        """Test _exit_checkpoint clears checkpoint state"""
        # Set up agent metadata for checkpoint
        mock_orchestrator._fm.metadata["agents"]["test_agent"] = {
            "checkpoint_counter": 5,
            "latest_checkpoint": "active",
        }

        agent = AgentContext("test_agent", mock_orchestrator)

        successes = 0
        with agent:
            agent.when_checkpoint("active")
            assert agent.active_checkpoint == "active"
            assert agent._checkpoint_counter == 5
            successes += 1

        # Verify checkpoint is cleared
        assert agent.active_checkpoint is None
        assert agent._checkpoint_counter is None
