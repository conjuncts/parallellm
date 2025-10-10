"""
Tests for HybridSQLiteDatastore
"""

import pytest
import tempfile
from pathlib import Path

from parallellm.core.datastore.semi_sql_parquet import SQLiteParquetDatastore
from parallellm.file_io.file_manager import FileManager
from parallellm.types import CallIdentifier


class TestHybridSQLiteDatastore:
    def test_store_and_retrieve_workflow(self):
        """Test the basic store -> persist -> retrieve workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup
            file_manager = FileManager(Path(temp_dir))
            datastore = SQLiteParquetDatastore(file_manager)

            # Test data
            call_id: CallIdentifier = {
                "checkpoint": None,
                "doc_hash": "test_hash_123",
                "seq_id": 1,
                "session_id": 100,
                "agent_name": "test_agent",
                "provider_type": "openai",
            }

            response = "This is a test response"
            response_id = "resp_123"
            metadata = {"usage": {"tokens": 50}, "model": "gpt-4"}

            # Store data (should go to SQLite first)
            seq_id = datastore.store(call_id, response, response_id)
            assert seq_id == 1

            # Store metadata
            datastore.store_metadata(call_id, response_id, metadata)

            # Retrieve from SQLite (before persist)
            retrieved_response = datastore.retrieve(call_id)
            assert retrieved_response == response

            retrieved_metadata = datastore.retrieve_metadata(call_id)
            assert retrieved_metadata == metadata

            # Persist (should transfer to Parquet)
            datastore.persist()

            # Retrieve from Parquet (after persist)
            retrieved_response_after_persist = datastore.retrieve(call_id)
            assert retrieved_response_after_persist == response

            retrieved_metadata_after_persist = datastore.retrieve_metadata(call_id)
            assert retrieved_metadata_after_persist == metadata

            # Verify parquet files exist
            parquet_paths = datastore._get_parquet_paths()
            assert parquet_paths["anon_responses"].exists()
            assert parquet_paths["metadata"].exists()

            # Cleanup
            datastore.close()

    def test_checkpoint_workflow(self):
        """Test workflow with checkpoint data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup
            file_manager = FileManager(Path(temp_dir))
            datastore = SQLiteParquetDatastore(file_manager)

            # Test data with checkpoint
            call_id: CallIdentifier = {
                "checkpoint": "checkpoint_v1",
                "doc_hash": "test_hash_456",
                "seq_id": 2,
                "session_id": 200,
                "agent_name": "checkpoint_agent",
                "provider_type": "openai",
            }

            response = "This is a checkpoint response"
            response_id = "resp_456"

            # Store and persist
            datastore.store(call_id, response, response_id)
            datastore.persist()

            # Retrieve from Parquet
            retrieved_response = datastore.retrieve(call_id)
            assert retrieved_response == response

            # Verify checkpoint parquet file exists
            parquet_paths = datastore._get_parquet_paths()
            assert parquet_paths["chk_responses"].exists()

            # Cleanup
            datastore.close()

    def test_mixed_sqlite_parquet_retrieve(self):
        """Test retrieving from both SQLite and Parquet"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup
            file_manager = FileManager(Path(temp_dir))
            datastore = SQLiteParquetDatastore(file_manager)

            # Store and persist first response
            call_id_1: CallIdentifier = {
                "checkpoint": None,
                "doc_hash": "hash_1",
                "seq_id": 1,
                "session_id": 100,
                "agent_name": "agent_1",
                "provider_type": "openai",
            }

            datastore.store(call_id_1, "response_1", "resp_1")
            datastore.persist()  # This goes to Parquet

            # Store second response (not yet persisted)
            call_id_2: CallIdentifier = {
                "checkpoint": None,
                "doc_hash": "hash_2",
                "seq_id": 2,
                "session_id": 100,
                "agent_name": "agent_1",
                "provider_type": "openai",
            }

            datastore.store(call_id_2, "response_2", "resp_2")
            # Don't persist yet - this stays in SQLite

            # Retrieve both
            response_1 = datastore.retrieve(call_id_1)  # Should come from Parquet
            response_2 = datastore.retrieve(call_id_2)  # Should come from SQLite

            assert response_1 == "response_1"
            assert response_2 == "response_2"

            # Cleanup
            datastore.close()
