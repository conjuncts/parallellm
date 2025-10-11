"""
Tests for HybridSQLiteDatastore
"""

import pytest
import tempfile
from pathlib import Path

from parallellm.core.datastore.semi_sql_parquet import SQLiteParquetDatastore
from parallellm.file_io.file_manager import FileManager
from parallellm.types import CallIdentifier


def test_mixed_sqlite_parquet_retrieve():
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

        call_id_chk: CallIdentifier = {
            "checkpoint": "checkpoint_v1",
            "doc_hash": "test_hash_456",
            "seq_id": 2,
            "session_id": 200,
            "agent_name": "checkpoint_agent",
            "provider_type": "openai",
        }

        datastore.store(call_id_1, "response_1", "resp_1")
        datastore.store(call_id_chk, "This is a checkpoint response", "resp_456")

        datastore.persist()
        # del datastore

        # datastore = SQLiteParquetDatastore(file_manager)

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
        response_2 = datastore.retrieve(call_id_2)  # Should come from SQLite

        response_1 = datastore.retrieve(call_id_1)  # Should come from Parquet
        response_chk = datastore.retrieve(call_id_chk)  # Should come from Parquet

        assert response_1 == "response_1"
        assert response_chk == "This is a checkpoint response"
        assert response_2 == "response_2"

        assert datastore._get_parquet_paths()["chk_responses"].exists()

        # Cleanup
        datastore.close()
