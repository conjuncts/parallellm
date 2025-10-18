import json
import sqlite3
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import polars as pl

from parallellm.core.datastore.base import Datastore
from parallellm.core.datastore.sqlite import SQLiteDatastore, _sql_table_to_dataframe
from parallellm.core.lake.sequester import sequester_df_to_parquet
from parallellm.file_io.file_manager import FileManager
from parallellm.types import CallIdentifier, ParsedResponse


class SQLiteParquetDatastore(Datastore):
    """
    Hybrid SQLite + Parquet Datastore implementation.

    Uses SQLite for transactional operations and transfers all data to Parquet
    upon persist() for efficient read-only datalake operations. On retrieve(),
    checks Parquet first, then falls back to SQLite for uncommitted data.
    """

    def __init__(self, file_manager: FileManager):
        """
        Initialize Hybrid SQLite + Parquet Datastore.

        :param file_manager: FileManager instance to handle file I/O operations
        """
        self.file_manager = file_manager
        self._sqlite_datastore = SQLiteDatastore(file_manager)

        # Cache for loaded parquet dataframes. Eagerly load
        self._eager_tables = {}
        self._load_parquet_cache()

    def _get_parquet_paths(self) -> dict[str, Path]:
        """Get paths for parquet files."""
        datastore_dir = self.file_manager.allocate_datastore()
        parquet_dir = datastore_dir / "datalake"
        parquet_dir.mkdir(parents=True, exist_ok=True)

        return {
            "anon_responses": parquet_dir / "anon_responses.parquet",
            "chk_responses": parquet_dir / "chk_responses.parquet",
            "metadata": parquet_dir / "metadata.parquet",
        }

    def _load_parquet_cache(self):
        """Load a parquet dataframe with caching."""

        parquet_paths = self._get_parquet_paths()
        for table_name, parquet_path in parquet_paths.items():
            if parquet_path.exists():
                try:
                    df = pl.read_parquet(parquet_path)
                    self._eager_tables[table_name] = df
                except Exception as e:
                    print(f"Warning: Failed to load {table_name} parquet: {e}")

    def _retrieve_row_from_parquet(self, call_id: CallIdentifier) -> Optional[dict]:
        """Retrieve response from parquet files."""
        checkpoint = call_id["checkpoint"]
        doc_hash = call_id["doc_hash"]
        seq_id = call_id["seq_id"]
        agent_name = call_id["agent_name"]

        # Determine which table to check
        table_name = "chk_responses" if checkpoint is not None else "anon_responses"
        df = self._eager_tables.get(table_name)

        if df is None or df.is_empty():
            return None

        filters = [pl.col("doc_hash") == doc_hash]

        if agent_name is not None:
            filters.append(pl.col("agent_name") == agent_name)
        else:
            filters.append(pl.col("agent_name").is_null())

        if checkpoint is not None:
            filters.append(pl.col("checkpoint") == checkpoint)

        specific_filters = filters + [pl.col("seq_id") == seq_id]
        result = df.filter(pl.all_horizontal(specific_filters))

        if not result.is_empty():
            return result.row(0, named=True)

        # Fallback: try without seq_id (less specific)
        result = df.filter(pl.all_horizontal(filters))
        if not result.is_empty():
            return result.row(0, named=True)

        return None

    def _retrieve_metadata_from_parquet_by_response_id(
        self, response_id: str
    ) -> Optional[dict]:
        """Retrieve metadata from parquet files using response_id."""

        # Get metadata using response_id
        metadata_df = self._load_parquet_dataframe("metadata")
        if metadata_df is None or metadata_df.is_empty():
            return None

        metadata_result = metadata_df.filter(pl.col("response_id") == response_id)
        if not metadata_result.is_empty():
            metadata_json = metadata_result.select("metadata").to_series().to_list()[0]
            if metadata_json:
                return json.loads(metadata_json)

        return None

    def _retrieve_metadata_from_parquet(
        self, call_id: CallIdentifier
    ) -> Optional[dict]:
        """Retrieve metadata from parquet files."""

        row = self._retrieve_row_from_parquet(call_id)

        if row is None:
            return None

        response_id = row["response_id"]
        if response_id is None:
            return None

        return self._retrieve_metadata_from_parquet_by_response_id(response_id)

    def retrieve(
        self, call_id: CallIdentifier, metadata=False
    ) -> Optional[ParsedResponse]:
        """
        Retrieve a response, checking parquet first, then SQLite.

        :param call_id: The task identifier containing checkpoint, doc_hash, and seq_id.
        :param metadata: Whether to include metadata in the response.
        :returns: The retrieved ParsedResponse containing text, response_id, and optionally metadata.
        """
        row = self._retrieve_row_from_parquet(call_id)
        if row is not None:
            response_text = row["response"]
            response_id = row.get("response_id")

            # Get metadata if requested
            response_metadata = None
            if metadata and response_id:
                response_metadata = self._retrieve_metadata_from_parquet_by_response_id(
                    response_id
                )

            return ParsedResponse(
                text=response_text, response_id=response_id, metadata=response_metadata
            )

        return self._sqlite_datastore.retrieve(call_id, metadata=metadata)

    def retrieve_metadata(self, response_id: str) -> Optional[dict]:
        """
        Retrieve metadata, checking parquet first, then SQLite.

        :param response_id: The response ID to look up metadata for.
        :returns: The retrieved metadata as a dictionary, or None if not found.
        """
        metadata = self._retrieve_metadata_from_parquet_by_response_id(response_id)
        if metadata is not None:
            return metadata

        return self._sqlite_datastore.retrieve_metadata(response_id)

    def store(
        self,
        call_id: CallIdentifier,
        parsed_response: "ParsedResponse",
    ) -> Optional[int]:
        """
        Store a response in SQLite (transactional).

        :param call_id: The task identifier containing checkpoint, doc_hash, seq_id, and session_id.
        :param parsed_response: The parsed response object containing text, response_id, and metadata.
        :returns: The seq_id where the response was stored.
        """

        # Store in SQLite for transactional integrity
        return self._sqlite_datastore.store(call_id, parsed_response)

    def _transfer_tables_to_parquet(self) -> None:
        """
        Transfer all response tables from SQLite to Parquet files.

        This method:
        1. Extracts all data from anon_responses, chk_responses, and metadata tables
        2. Merges with existing Parquet data (if any)
        3. Writes to temporary files, then swaps them atomically
        4. Removes transferred data from SQLite only on success
        """
        conn = self._sqlite_datastore._get_connection(None)
        parquet_paths = self._get_parquet_paths()

        try:
            tables_to_process = [
                ("anon_responses", "anon_responses"),
                ("chk_responses", "chk_responses"),
                ("metadata", "metadata"),
            ]

            data_to_delete = {}

            for table_name, parquet_key in tables_to_process:
                parquet_path = parquet_paths[parquet_key]

                df = _sql_table_to_dataframe(conn, f"SELECT * FROM {table_name}")
                transferred_ids = sequester_df_to_parquet(df, table_name, parquet_path)

                if transferred_ids:
                    data_to_delete[table_name] = transferred_ids

            # Delete transferred data from SQLite
            for table, ids in data_to_delete.items():
                if not ids:
                    continue

                if table in ["anon_responses", "chk_responses"]:
                    # Delete by ID
                    placeholders = ",".join(["?" for _ in ids])
                    conn.execute(
                        f"DELETE FROM {table} WHERE id IN ({placeholders})", ids
                    )
                elif table == "metadata":
                    # Delete by response_id
                    placeholders = ",".join(["?" for _ in ids])
                    conn.execute(
                        f"DELETE FROM {table} WHERE response_id IN ({placeholders})",
                        ids,
                    )

            conn.commit()

            # Clear parquet cache to force reload on next access
            self._eager_tables.clear()

        except Exception as e:
            raise RuntimeError(f"Error transferring tables to Parquet: {e}")

    def persist(self) -> None:
        """
        Persist changes to Parquet datalake.

        Transfers all SQLite data to Parquet files and then calls the underlying
        SQLite datastore's persist method for metadata transfer.
        """
        try:
            self._transfer_tables_to_parquet()
            self._load_parquet_cache()

        except Exception as e:
            print(f"Warning: Failed to transfer data to Parquet: {e}")
        finally:
            self._sqlite_datastore.persist()

    def close(self, checkpoint: Optional[str] = None) -> None:
        """
        Close connections and clear caches.

        :param checkpoint: The checkpoint connection to close (if None, close all connections).
        """
        self._eager_tables.clear()
        self._sqlite_datastore.close(checkpoint)

    def __del__(self):
        """
        Cleanup: close all connections and clear caches when the object is destroyed.
        """
        try:
            self.close()
        except Exception:
            pass
