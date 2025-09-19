from abc import ABC
from typing import Optional, Union
import sqlite3
from pathlib import Path

import polars as pl

from parallellm.file_io.file_manager import FileManager


class DataStore(ABC):
    """
    Stores responses
    """

    def retrieve(self, checkpoint: str, doc_hash: str, seq_id: int) -> Optional[str]:
        """
        Retrieve a response from the backend.

        :param checkpoint: The checkpoint of the response.
        :param doc_hash: The document hash of the response.
        :param seq_id: The sequential ID of the response.
        :returns: The retrieved LLMResponse.
        """
        raise NotImplementedError

    def store(
        self,
        checkpoint: str,
        doc_hash: str,
        seq_id: int,
        response: str,
        response_id: str,
        *,
        save_to_file: bool = True,
    ) -> Optional[int]:
        """
        Store a response in the backend.

        :param checkpoint: The checkpoint of the response.
        :param doc_hash: The document hash of the response.
        :param seq_id: The sequential ID of the response.
        :param response: The response content to store.
        :param response_id: The response ID to store.
        :param save_to_file: Whether to save the updated data back to the file.
        :returns: The seq_id where the response was stored (if applicable).
        """
        raise NotImplementedError

    def store_metadata(
        self,
        checkpoint: str,
        doc_hash: str,
        seq_id: int,
        response_id: str,
        metadata: dict,
    ) -> None:
        """
        Store metadata in the backend.

        :param checkpoint: The checkpoint of the metadata.
        :param doc_hash: The document hash of the response.
        :param seq_id: The sequential ID of the response.
        :param response_id: The response ID to store.
        :param metadata: The metadata to store.
        """
        raise NotImplementedError

    def persist(self, checkpoint: Optional[str] = None) -> None:
        """
        Persist changes to file(s).

        :param checkpoint: The checkpoint to persist (if None, persist all checkpoints with changes).
        """
        raise NotImplementedError


# class DataFrameBackend(BaseBackend):
#     """
#     Backed by a polars DataFrame
#     """

#     def __init__(self, checkpoint_to_resource: dict[str, str]):
#         """

#         :param checkpoint_to_resource: A mapping from checkpoint names to file paths to parquet files

#         dataframe should have columns:
#             - doc_hash: str
#             - response: str
#             - seq_id: int (optional)
#         """
#         self.checkpoint_to_resource = checkpoint_to_resource
#         self._dataframes: dict[str, pl.DataFrame] = {}

#     def _load_checkpoint(self, checkpoint: str) -> pl.DataFrame:
#         if checkpoint not in self._dataframes:
#             if checkpoint not in self.checkpoint_to_resource:
#                 raise ValueError(f"Checkpoint {checkpoint} not found in checkpoint_to_resource mapping")
#             path = self.checkpoint_to_resource[checkpoint]
#             self._dataframes[checkpoint] = pl.read_parquet(path)
#         return self._dataframes[checkpoint]

#     def retrieve(self, checkpoint: str, doc_hash: str, seq_id: Optional[int] = None) -> Optional[str]:
#         df = self._load_checkpoint(checkpoint)
#         # First, try an O(1) lookup just to see if it works

#         if seq_id is not None:
#             item = df.row(row=seq_id)
#             if item is not None:
#                 if item['doc_hash'] == doc_hash:
#                     return item['response']
#             # otherwise - oops, sequential id is not reliable

#         # Fallback to a O(n) lookup
#         # And we also know the seq_id is no good (ignore it)
#         filtered = df.filter(pl.col("doc_hash") == doc_hash)
#         if filtered.height == 0:
#             return None

#         return filtered.item(0, "response")

#     def store(self, checkpoint: str, doc_hash: str, response: str, seq_id: Optional[int] = None, save_to_file: bool = True) -> None:
#         """
#         Store a response in the backend.

#         :param checkpoint: The checkpoint of the response.
#         :param doc_hash: The document hash of the response.
#         :param response: The response content to store.
#         :param seq_id: The sequential ID of the response (optional).
#         :param save_to_file: Whether to save the updated DataFrame back to the parquet file.
#         """
#         # Load or initialize the DataFrame for this checkpoint
#         try:
#             df = self._load_checkpoint(checkpoint)
#         except ValueError:
#             # Checkpoint doesn't exist yet, create a new DataFrame
#             if checkpoint not in self.checkpoint_to_resource:
#                 raise ValueError(f"Checkpoint {checkpoint} not found in checkpoint_to_resource mapping")

#             # Create empty DataFrame with correct schema
#             if seq_id is not None:
#                 df = pl.DataFrame({
#                     "doc_hash": [],
#                     "response": [],
#                     "seq_id": []
#                 }, schema={"doc_hash": pl.Utf8, "response": pl.Utf8, "seq_id": pl.Int64})
#             else:
#                 df = pl.DataFrame({
#                     "doc_hash": [],
#                     "response": []
#                 }, schema={"doc_hash": pl.Utf8, "response": pl.Utf8})

#             self._dataframes[checkpoint] = df

#         # Check if the doc_hash already exists
#         existing = df.filter(pl.col("doc_hash") == doc_hash)
#         if existing.height > 0:
#             # Update existing record
#             if "seq_id" in df.columns and seq_id is not None:
#                 new_row = pl.DataFrame({
#                     "doc_hash": [doc_hash],
#                     "response": [response],
#                     "seq_id": [seq_id]
#                 })
#             else:
#                 new_row = pl.DataFrame({
#                     "doc_hash": [doc_hash],
#                     "response": [response]
#                 })

#             # Remove old record and add new one
#             df = df.filter(pl.col("doc_hash") != doc_hash)
#             df = pl.concat([df, new_row])
#         else:
#             # Add new record
#             if "seq_id" in df.columns and seq_id is not None:
#                 new_row = pl.DataFrame({
#                     "doc_hash": [doc_hash],
#                     "response": [response],
#                     "seq_id": [seq_id]
#                 })
#             else:
#                 new_row = pl.DataFrame({
#                     "doc_hash": [doc_hash],
#                     "response": [response]
#                 })

#             df = pl.concat([df, new_row])

#         # Update the cached DataFrame
#         self._dataframes[checkpoint] = df

#         # Save to file if requested
#         if save_to_file:
#             path = self.checkpoint_to_resource[checkpoint]
#             df.write_parquet(path)


# class DictBackend(BaseBackend):
#     """
#     Backed by hashmaps for fast lookups, with polars DataFrame/parquet for serialization
#     """

#     def __init__(self, checkpoint_to_resource: dict[str, str]):
#         """

#         :param checkpoint_to_resource: A mapping from checkpoint names to file paths to parquet files

#         dataframe should have columns:
#             - doc_hash: str
#             - response: str
#             - seq_id: int (optional)
#         """
#         self.checkpoint_to_resource = checkpoint_to_resource
#         self._hashmaps: dict[str, dict[str, str]] = {}

#     def _load_checkpoint(self, checkpoint: str) -> dict[str, str]:
#         if checkpoint not in self._hashmaps:
#             if checkpoint not in self.checkpoint_to_resource:
#                 raise ValueError(f"Checkpoint {checkpoint} not found in checkpoint_to_resource mapping")
#             path = self.checkpoint_to_resource[checkpoint]
#             df = pl.read_parquet(path)

#             # Convert DataFrame to hashmap for O(1) lookups
#             hashmap = {}
#             for row in df.iter_rows(named=True):
#                 hashmap[row['doc_hash']] = row['response']

#             self._hashmaps[checkpoint] = hashmap
#         return self._hashmaps[checkpoint]

#     def retrieve(self, checkpoint: str, doc_hash: str, seq_id: Optional[int] = None) -> Optional[str]:
#         hashmap = self._load_checkpoint(checkpoint)
#         return hashmap.get(doc_hash)


class ListDataStore(DataStore):
    """
    Backed by lists for O(1) operations with hash-to-index mapping
    """

    def __init__(self, file_manager: FileManager):
        """

        :param file_manager: FileManager instance to handle file I/O operations
        """
        self.file_manager = file_manager
        # data: checkpoint -> list of responses (seq_id is the index)
        self._data: dict[str, list[str]] = {}
        # hashes: checkpoint -> (doc_hash -> seq_id)
        self._hashes: dict[
            str, dict[str, int]
        ] = {}  # Track which checkpoints have unsaved changes
        self._dirty_checkpoints: set[str] = set()

    def _load_checkpoint(self, checkpoint: str) -> None:
        """Load checkpoint data from parquet into lists and hash mappings"""
        if checkpoint not in self._data:
            parquet_file = (
                self.file_manager.allocate_datastore(checkpoint) / "datastore.parquet"
            )
            try:
                # Initialize empty structures
                data_list = []
                hash_map = {}

                try:
                    df = pl.read_parquet(parquet_file)

                    if "seq_id" in df.columns:
                        df = df.sort("seq_id")

                        for row in df.iter_rows(named=True):
                            seq_id = row["seq_id"]
                            doc_hash = row["doc_hash"]
                            response = row["response"]

                            # Extend list if necessary to accommodate the seq_id
                            while len(data_list) <= seq_id:
                                data_list.append(None)

                            data_list[seq_id] = response
                            hash_map[doc_hash] = seq_id
                    else:
                        # If no seq_id column, use row order as seq_id
                        for i, row in enumerate(df.iter_rows(named=True)):
                            doc_hash = row["doc_hash"]
                            response = row["response"]

                            data_list.append(response)
                            hash_map[doc_hash] = i

                except FileNotFoundError:
                    # File doesn't exist yet, return empty structures
                    pass
                self._data[checkpoint] = data_list
                self._hashes[checkpoint] = hash_map
            except Exception as e:
                # Initialize empty structures if loading fails
                self._data[checkpoint] = []
                self._hashes[checkpoint] = {}

    def retrieve(self, checkpoint: str, doc_hash: str, seq_id: int) -> Optional[str]:
        self._load_checkpoint(checkpoint)

        # Try direct access with seq_id
        data_list = self._data[checkpoint]
        if 0 <= seq_id < len(data_list) and data_list[seq_id] is not None:
            # Verify the hash matches (as a safety check)
            hash_map = self._hashes[checkpoint]
            if doc_hash in hash_map and hash_map[doc_hash] == seq_id:
                return data_list[seq_id]

        # Fallback to hash lookup
        hash_map = self._hashes[checkpoint]
        if doc_hash in hash_map:
            found_seq_id = hash_map[doc_hash]
            return self._data[checkpoint][found_seq_id]

        return None

    def store(
        self,
        checkpoint: str,
        doc_hash: str,
        seq_id: int,
        response: str,
        response_id: str,
        *,
        save_to_file: bool = True,
    ) -> int:
        """
        Store a response in the backend.

        :param checkpoint: The checkpoint of the response.
        :param doc_hash: The document hash of the response.
        :param seq_id: The sequential ID of the response.
        :param response: The response content to store.
        :param response_id: The response ID to store.
        :param save_to_file: Whether to save the updated data back to the parquet file.
        :returns: The seq_id where the response was stored.
        """
        self._load_checkpoint(checkpoint)

        data_list = self._data[checkpoint]
        hash_map = self._hashes[checkpoint]

        # Check if doc_hash already exists
        if doc_hash in hash_map:
            # Update existing record
            existing_id = hash_map[doc_hash]
            data_list[existing_id] = response
            actual_seq_id = existing_id
        else:
            # Add new record at specified seq_id
            while len(data_list) <= seq_id:
                data_list.append(None)
            data_list[seq_id] = response
            hash_map[doc_hash] = seq_id
            actual_seq_id = seq_id
        # Mark checkpoint as dirty (has unsaved changes)
        self._dirty_checkpoints.add(checkpoint)
        # Save to file if requested
        if save_to_file:
            self._save_checkpoint(checkpoint)

        return actual_seq_id

    def persist(self, checkpoint: Optional[str] = None) -> None:
        """
        Persist changes to file(s).

        :param checkpoint: The checkpoint to persist (if None, persist all checkpoints with changes).
        """
        if checkpoint is not None:
            # Persist specific checkpoint
            if checkpoint in self._dirty_checkpoints:
                self._save_checkpoint(checkpoint)
        else:
            # Persist all dirty checkpoints
            for dirty_checkpoint in list(self._dirty_checkpoints):
                self._save_checkpoint(dirty_checkpoint)

    def _save_checkpoint(self, checkpoint: str) -> None:
        """Save checkpoint data to parquet file"""
        data_list = self._data[checkpoint]
        hash_map = self._hashes[checkpoint]
        # Build reverse mapping for doc_hashes
        id_to_hash = {v: k for k, v in hash_map.items()}
        doc_hashes = []
        responses = []
        for i, response in enumerate(data_list):
            if response is not None and i in id_to_hash:
                doc_hashes.append(id_to_hash[i])
                responses.append(response)
        df = pl.DataFrame(
            {
                "doc_hash": doc_hashes,
                "response": responses,
            }
        ).with_row_index("seq_id")
        # Use file manager to save the DataFrame
        directory = self.file_manager.allocate_datastore(checkpoint)
        df.write_parquet(directory / "datastore.parquet")

        # Mark checkpoint as clean (saved)
        self._dirty_checkpoints.discard(checkpoint)
