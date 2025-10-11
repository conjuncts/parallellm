"""
Unit tests for userdata persistence and FileManager

Tests the userdata persistence functionality including:
- FileManager initialization and metadata handling
- save_userdata and load_userdata operations
- Agent metadata management
- File sanitization and directory allocation
- Session counter and lock file handling
"""

import pytest
import tempfile
import os
import json
import pickle
from pathlib import Path
from unittest.mock import Mock, patch
from parallellm.file_io.file_manager import FileManager
from parallellm.core.agent.orchestrator import AgentOrchestrator
from parallellm.core.response import ReadyLLMResponse, PendingLLMResponse
from parallellm.types import WorkingMetadata, AgentMetadata, CallIdentifier


class TestFileManagerBasics:
    """Test basic FileManager functionality"""

    def test_file_manager_creation(self):
        """Test creating a FileManager"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            assert fm.directory == Path(temp_dir)
            assert fm.metadata_file == Path(temp_dir) / "metadata.json"
            assert fm.lock_file == Path(temp_dir) / ".filemanager.lock"

            # Directory should be created
            assert Path(temp_dir).exists()

            # Lock file should be created
            assert fm.lock_file.exists()

    def test_metadata_initialization(self):
        """Test metadata is properly initialized"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            # Should have default metadata structure
            assert "agents" in fm.metadata
            assert "session_counter" in fm.metadata
            assert "default-agent" in fm.metadata["agents"]

            # Default agent should have proper structure
            default_agent = fm.metadata["agents"]["default-agent"]
            assert "latest_checkpoint" in default_agent
            assert "checkpoint_counter" in default_agent
            assert default_agent["latest_checkpoint"] is None
            assert default_agent["checkpoint_counter"] == 0

    def test_session_counter_increments(self):
        """Test session counter increments on each new FileManager"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First instance
            fm1 = FileManager(temp_dir)
            session1 = fm1.metadata["session_counter"]
            fm1.persist()

            # Second instance should increment
            fm2 = FileManager(temp_dir)
            session2 = fm2.metadata["session_counter"]

            assert session2 == session1 + 1

    def test_metadata_persistence_across_instances(self):
        """Test that metadata persists across FileManager instances"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First instance - modify metadata
            fm1 = FileManager(temp_dir)
            fm1.metadata["agents"]["test_agent"] = {
                "latest_checkpoint": "test_checkpoint",
                "checkpoint_counter": 5,
            }
            fm1.persist()

            # Second instance should load persisted metadata
            fm2 = FileManager(temp_dir)
            assert "test_agent" in fm2.metadata["agents"]
            assert (
                fm2.metadata["agents"]["test_agent"]["latest_checkpoint"]
                == "test_checkpoint"
            )
            assert fm2.metadata["agents"]["test_agent"]["checkpoint_counter"] == 5

    def test_lock_file_cleanup(self):
        """Test lock file is cleaned up"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)
            lock_file = fm.lock_file

            assert lock_file.exists()

            # Cleanup should remove lock file
            fm._cleanup()
            assert not lock_file.exists()


class TestSanitization:
    """Test input sanitization functionality"""

    def test_sanitize_basic(self):
        """Test basic string sanitization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            result = fm._sanitize("simple_name")
            assert "simple_name" in result
            assert len(result.split("-")) == 2  # name-hash format

    def test_sanitize_none_input(self):
        """Test sanitization with None input"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            result = fm._sanitize(None)
            assert result == "default"

            result_custom = fm._sanitize(None, default="custom")
            assert result_custom == "custom"

    def test_sanitize_special_characters(self):
        """Test sanitization removes special characters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            result = fm._sanitize("test/name!@#$%")
            # Should replace special chars with underscores
            assert "/" not in result
            assert "!" not in result
            assert "@" not in result

    def test_sanitize_no_hash(self):
        """Test sanitization without hash"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            result = fm._sanitize("test_name", add_hash=False)
            assert result == "test_name"
            assert "-" not in result

    def test_sanitize_long_string(self):
        """Test sanitization with long strings"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            long_name = "a" * 100  # 100 characters
            result = fm._sanitize(long_name)

            # Should be truncated to 64 chars + hash
            name_part = result.split("-")[0]
            assert len(name_part) <= 64

    def test_sanitize_empty_string(self):
        """Test sanitization with empty string"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            result = fm._sanitize("")
            assert "checkpoint" in result  # Default fallback


class TestUserdataPersistence:
    """Test userdata save/load functionality"""

    def test_save_and_load_simple_data(self):
        """Test saving and loading simple data types"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            # Test different data types
            test_data = {
                "string": "Hello World",
                "number": 42,
                "list": [1, 2, 3],
                "dict": {"key": "value"},
            }

            for key, value in test_data.items():
                fm.save_userdata(key, value)
                loaded_value = fm.load_userdata(key)
                assert loaded_value == value

    def test_save_userdata_creates_directory(self):
        """Test that save_userdata creates userdata directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            userdata_dir = fm.directory / "userdata"
            assert not userdata_dir.exists()

            fm.save_userdata("test_key", "test_value")

            assert userdata_dir.exists()
            assert userdata_dir.is_dir()

    def test_load_nonexistent_data(self):
        """Test loading nonexistent data raises FileNotFoundError"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            with pytest.raises(FileNotFoundError):
                fm.load_userdata("nonexistent_key")

    def test_save_userdata_overwrite_protection(self):
        """Test overwrite protection in save_userdata"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            # Save initial data
            fm.save_userdata("test_key", "original_value")

            # No overwrite (should not change)
            fm.save_userdata("test_key", "new_value", overwrite=False)
            loaded_value = fm.load_userdata("test_key")
            assert loaded_value == "original_value"

            # Overwrite (should change)
            fm.save_userdata("test_key", "new_value", overwrite=True)
            loaded_value = fm.load_userdata("test_key")
            assert loaded_value == "new_value"

    def test_userdata_file_naming(self):
        """Test that userdata files are named correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            fm.save_userdata("test/key", "test_value")

            # Check that file was created with sanitized name
            userdata_dir = fm.directory / "userdata"
            pkl_files = list(userdata_dir.glob("*.pkl"))

            assert len(pkl_files) == 1
            assert "test_key" in pkl_files[0].name  # Should be sanitized


class TestDatastoreAllocation:
    """Test datastore directory allocation"""

    def test_allocate_datastore_basic(self):
        """Test basic datastore directory retrieval"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            path = fm.allocate_datastore()

            # Should create base datastore directory
            expected_path = Path(temp_dir) / "datastore"

            assert path.exists()
            assert path.is_dir()
            assert path == expected_path

    def test_get_datastore_directory_creates_parents(self):
        """Test that get_datastore_directory creates the directory if it doesn't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            datastore_base = Path(temp_dir) / "datastore"
            assert not datastore_base.exists()

            path = fm.allocate_datastore()

            assert datastore_base.exists()
            assert path == datastore_base


class TestAgentOrchestratorIntegration:
    """Test integration with AgentOrchestrator"""

    def test_orchestrator_userdata_operations(self):
        """Test userdata operations through AgentOrchestrator"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)
            mock_backend = Mock()
            mock_provider = Mock()
            mock_logger = Mock()
            mock_dash_logger = Mock()

            orchestrator = AgentOrchestrator(
                file_manager=fm,
                backend=mock_backend,
                provider=mock_provider,
                logger=mock_logger,
                dash_logger=mock_dash_logger,
            )

            # Test save/load through orchestrator
            test_data = {"key": "value", "number": 42}
            orchestrator.save_userdata("test_data", test_data)

            loaded_data = orchestrator.load_userdata("test_data")
            assert loaded_data == test_data

    def test_orchestrator_response_injection(self):
        """Test that orchestrator injects backend into loaded responses"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)
            mock_backend = Mock()
            mock_provider = Mock()
            mock_logger = Mock()
            mock_dash_logger = Mock()

            orchestrator = AgentOrchestrator(
                file_manager=fm,
                backend=mock_backend,
                provider=mock_provider,
                logger=mock_logger,
                dash_logger=mock_dash_logger,
            )

            call_id = self._create_mock_call_id()
            pending_response = PendingLLMResponse(call_id=call_id, backend=None)
            ready_response = ReadyLLMResponse(call_id=call_id, value="test_value")

            # Save responses
            orchestrator.save_userdata("pending", pending_response)
            orchestrator.save_userdata("ready", ready_response)

            # Load and verify backend injection
            loaded_pending = orchestrator.load_userdata("pending")
            loaded_ready = orchestrator.load_userdata("ready")

            assert isinstance(loaded_pending, PendingLLMResponse)
            assert loaded_pending._backend == mock_backend

            assert isinstance(loaded_ready, ReadyLLMResponse)
            mock_backend.retrieve.assert_called_with(ready_response.call_id)

    def test_orchestrator_ignore_cache_parameter(self):
        """Test that ignore_cache parameter works correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)
            mock_backend = Mock()
            mock_provider = Mock()
            mock_logger = Mock()
            mock_dash_logger = Mock()

            # Set up mock backend to return cached response
            mock_backend.retrieve.return_value = "cached_response"

            mock_call_id = self._create_mock_call_id()
            mock_provider.submit_query_to_provider.return_value = ReadyLLMResponse(
                call_id=mock_call_id, value="fresh_response"
            )

            # Create orchestrator with ignore_cache=True
            orchestrator = AgentOrchestrator(
                file_manager=fm,
                backend=mock_backend,
                provider=mock_provider,
                logger=mock_logger,
                dash_logger=mock_dash_logger,
                ignore_cache=True,
            )

            with orchestrator.agent() as agent:
                response = agent.ask_llm("Test prompt")

                # Backend retrieve should NOT be called
                mock_backend.retrieve.assert_not_called()

                mock_provider.submit_query_to_provider.assert_called_once()

    def _create_mock_call_id(self) -> CallIdentifier:
        """Helper to create mock call identifiers"""
        return {
            "agent_name": "test_agent",
            "checkpoint": "test_checkpoint",
            "doc_hash": "test_hash",
            "seq_id": 1,
            "session_id": 1,
            "provider_type": "openai",
        }


class TestCheckpointLogging:
    """Test checkpoint event logging"""

    def test_log_checkpoint_event(self):
        """Test checkpoint event logging"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            fm.log_checkpoint_event("enter", "test_agent", "test_checkpoint", 5)

            log_file = fm.directory / "logs" / "checkpoint_events.tsv"
            assert log_file.exists()

            # Read and verify log content
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            assert len(lines) == 2  # Header + one event
            assert "session_id\tevent_type\tagent_name\tcheckpoint\tseq_id" in lines[0]
            assert "enter\ttest_agent\ttest_checkpoint\t5" in lines[1]

    def test_log_multiple_checkpoint_events(self):
        """Test logging multiple checkpoint events"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            # Log multiple events
            events = [
                ("enter", "agent1", "checkpoint1", 0),
                ("switch", "agent1", "checkpoint2", 3),
                ("exit", "agent1", "checkpoint2", 5),
            ]

            for event_type, agent, checkpoint, seq_id in events:
                fm.log_checkpoint_event(event_type, agent, checkpoint, seq_id)

            # Verify all events are logged
            log_file = fm.directory / "logs" / "checkpoint_events.tsv"
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            assert len(lines) == 4  # Header + 3 events
            for i, (event_type, agent, checkpoint, seq_id) in enumerate(events, 1):
                assert event_type in lines[i]
                assert agent in lines[i]
                assert checkpoint in lines[i]


class TestFileManagerPersistence:
    """Test FileManager persist functionality"""

    def test_persist_saves_metadata(self):
        """Test that persist saves metadata to disk"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            # Modify metadata
            fm.metadata["agents"]["new_agent"] = {
                "latest_checkpoint": "new_checkpoint",
                "checkpoint_counter": 10,
            }

            fm.persist()

            # Verify metadata file contains changes
            with open(fm.metadata_file, "r") as f:
                saved_metadata = json.load(f)

            assert "new_agent" in saved_metadata["agents"]
            assert (
                saved_metadata["agents"]["new_agent"]["latest_checkpoint"]
                == "new_checkpoint"
            )
            assert saved_metadata["agents"]["new_agent"]["checkpoint_counter"] == 10

    def test_persist_idempotent(self):
        """Test that multiple persist calls are safe"""
        with tempfile.TemporaryDirectory() as temp_dir:
            fm = FileManager(temp_dir)

            # Multiple persist calls should not cause issues
            fm.persist()
            fm.persist()
            fm.persist()

            # Metadata should still be valid
            assert "agents" in fm.metadata
            assert "session_counter" in fm.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
