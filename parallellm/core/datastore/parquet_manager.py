"""
Parquet Manager for handling parquet file operations.

This component provides centralized parquet read/write functionality
that can be shared across different datastore implementations.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any

import polars as pl

from parallellm.file_io.file_manager import FileManager
from parallellm.types import CallIdentifier


class ParquetManager:
    """
    Manages parquet file operations for datastore implementations.

    Provides centralized functionality for:
    - Loading parquet files into memory
    - Retrieving metadata from parquet
    - Managing parquet file paths
    """

    def __init__(self, file_manager: FileManager):
        """
        Initialize ParquetManager.

        :param file_manager: FileManager instance to handle file I/O operations
        """
        self.file_manager = file_manager
        self._eager_tables: Dict[str, pl.DataFrame] = {}
        self._metadata_cache: Dict[str, dict] = {}
        self._load_parquet_cache()

    def get_parquet_paths(self) -> Dict[str, Path]:
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
        """Load parquet dataframes and metadata into cache for fast access."""
        parquet_paths = self.get_parquet_paths()

        # Load all parquet files into eager tables
        for table_name, parquet_path in parquet_paths.items():
            if parquet_path.exists():
                try:
                    df = pl.read_parquet(parquet_path)
                    self._eager_tables[table_name] = df
                except Exception as e:
                    print(f"Warning: Failed to load {table_name} parquet: {e}")

        # Build metadata cache for O(1) lookup
        self._build_metadata_cache()

    def _build_metadata_cache(self):
        """Build metadata cache from loaded parquet data."""
        self._metadata_cache.clear()

        metadata_df = self._eager_tables.get("metadata")
        if metadata_df is not None and not metadata_df.is_empty():
            try:
                for row in metadata_df.iter_rows(named=True):
                    response_id = row["response_id"]
                    metadata_json = row["metadata"]
                    if response_id and metadata_json:
                        self._metadata_cache[response_id] = json.loads(metadata_json)
            except Exception as e:
                print(f"Warning: Failed to build metadata cache: {e}")

    def reload_cache(self):
        """Reload parquet cache from disk (e.g., after sequestering)."""
        self._load_parquet_cache()

    def get_metadata(self, response_id: str) -> Optional[dict]:
        """
        Retrieve metadata by response_id from cache.

        :param response_id: The response ID to look up metadata for.
        :returns: The retrieved metadata as a dictionary, or None if not found.
        """
        return self._metadata_cache.get(response_id)

    def retrieve_response_row(self, call_id: CallIdentifier) -> Optional[dict]:
        """
        Retrieve response row from parquet files using call_id.

        :param call_id: The task identifier containing checkpoint, doc_hash, and seq_id.
        :returns: The retrieved row as a dictionary, or None if not found.
        """
        checkpoint = call_id["checkpoint"]
        doc_hash = call_id["doc_hash"]
        seq_id = call_id["seq_id"]
        agent_name = call_id["agent_name"]

        # Determine which table to check
        table_name = "chk_responses" if checkpoint is not None else "anon_responses"
        df = self._eager_tables.get(table_name)

        if df is None or df.is_empty():
            return None

        # Build filters
        filters = [pl.col("doc_hash") == doc_hash]

        if agent_name is not None:
            filters.append(pl.col("agent_name") == agent_name)
        else:
            filters.append(pl.col("agent_name").is_null())

        if checkpoint is not None:
            filters.append(pl.col("checkpoint") == checkpoint)

        # Try with seq_id first (most specific)
        specific_filters = filters + [pl.col("seq_id") == seq_id]
        result = df.filter(pl.all_horizontal(specific_filters))

        if not result.is_empty():
            return result.row(0, named=True)

        # Fallback: try without seq_id (less specific)
        result = df.filter(pl.all_horizontal(filters))
        if not result.is_empty():
            return result.row(0, named=True)

        return None
