import sqlite3
import json
import threading
from typing import Optional
from pathlib import Path

import polars as pl

from parallellm.core.datastore.base import Datastore
from parallellm.core.datalake.metadata_sinks import openai_metadata_sinker
from parallellm.file_io.file_manager import FileManager
from parallellm.types import CallIdentifier


class SQLiteDatastore(Datastore):
    """
    SQLite-backed Datastore implementation
    """

    def __init__(self, file_manager: FileManager):
        """
        Initialize SQLite Datastore.

        :param file_manager: FileManager instance to handle file I/O operations
        """
        self.file_manager = file_manager
        # Use threading.local to ensure each thread has its own connections
        self._local = threading.local()
        self._dirty_checkpoints = set()

    def _get_connections(self) -> dict[str, sqlite3.Connection]:
        """Get the connections dictionary for the current thread"""
        if not hasattr(self._local, "connections"):
            self._local.connections = {}
        return self._local.connections

    def _get_connection(self, checkpoint: Optional[str]) -> sqlite3.Connection:
        """Get or create SQLite connection for a checkpoint in the current thread"""
        connections = self._get_connections()

        if checkpoint not in connections:
            # Create database file using file manager
            directory = self.file_manager.allocate_datastore(checkpoint)
            db_path = directory / "datastore.db"

            # Create connection
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row  # Enable dict-like access to rows

            # Create table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    seq_id INTEGER NOT NULL,
                    session_id INTEGER NOT NULL,
                    doc_hash TEXT NOT NULL,
                    response TEXT NOT NULL,
                    response_id TEXT,
                    provider_type TEXT,
                    UNIQUE(doc_hash)
                )
            """)

            # Create metadata table with response_id as join key
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    response_id TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    provider_type TEXT,
                    UNIQUE(response_id)
                )
            """)

            # Create index for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_hash ON responses(doc_hash)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id ON responses(session_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_seq_id ON responses(seq_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_response_id ON responses(response_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_provider_type ON responses(provider_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metadata_response_id ON metadata(response_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metadata_provider_type ON metadata(provider_type)
            """)

            # Migrate existing schema if needed
            self._migrate_schema(conn)

            conn.commit()
            connections[checkpoint] = conn

        self._dirty_checkpoints.add(checkpoint)
        return connections[checkpoint]

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """
        Migrate database schema from old format to new format.

        Moves metadata from responses table to separate metadata table.
        """
        try:
            # Check if old schema exists (responses table with metadata column)
            cursor = conn.execute("PRAGMA table_info(responses)")
            columns = [row[1] for row in cursor.fetchall()]

            if "provider_type" not in columns:
                print("Migrating SQLite...")
                # Add provider_type column to responses table (default "openai")
                conn.execute("ALTER TABLE responses ADD COLUMN provider_type TEXT")
                conn.execute(
                    "UPDATE responses SET provider_type = 'openai' WHERE provider_type IS NULL"
                )

                # Add provider_type column to metadata table
                conn.execute("ALTER TABLE metadata ADD COLUMN provider_type TEXT")
                conn.execute(
                    "UPDATE metadata SET provider_type = 'openai' WHERE provider_type IS NULL"
                )
                conn.commit()

        except sqlite3.Error:
            # If migration fails, continue - tables will be created fresh
            pass

    def retrieve(self, call_id: CallIdentifier) -> Optional[str]:
        """
        Retrieve a response from SQLite.

        :param call_id: The task identifier containing checkpoint, doc_hash, and seq_id.
        :returns: The retrieved response content.
        """
        checkpoint = call_id["checkpoint"]
        doc_hash = call_id["doc_hash"]
        seq_id = call_id["seq_id"]
        # session_id NOT relevant for lookup

        conn = self._get_connection(checkpoint)

        # Try direct lookup using seq_id and doc_hash
        cursor = conn.execute(
            "SELECT response FROM responses WHERE seq_id = ? AND doc_hash = ?",
            (seq_id, doc_hash),
        )
        row = cursor.fetchone()
        if row:
            return row["response"]

        # Fallback to doc_hash lookup
        cursor = conn.execute(
            "SELECT response FROM responses WHERE doc_hash = ?", (doc_hash,)
        )
        row = cursor.fetchone()
        return row["response"] if row else None

    def retrieve_metadata(self, call_id: CallIdentifier) -> Optional[dict]:
        """
        Retrieve metadata from SQLite.

        :param call_id: The task identifier containing checkpoint, doc_hash, and seq_id.
        :returns: The retrieved metadata as a dictionary, or None if not found.
        """
        checkpoint = call_id["checkpoint"]
        doc_hash = call_id["doc_hash"]
        seq_id = call_id["seq_id"]

        conn = self._get_connection(checkpoint)

        # First get the response_id from the responses table
        cursor = conn.execute(
            "SELECT response_id FROM responses WHERE seq_id = ? AND doc_hash = ?",
            (seq_id, doc_hash),
        )
        row = cursor.fetchone()

        if not row or not row["response_id"]:
            # Fallback to doc_hash lookup for response_id
            cursor = conn.execute(
                "SELECT response_id FROM responses WHERE doc_hash = ?", (doc_hash,)
            )
            row = cursor.fetchone()

        if row and row["response_id"]:
            response_id = row["response_id"]
            # Now get metadata using response_id
            cursor = conn.execute(
                "SELECT metadata FROM metadata WHERE response_id = ?", (response_id,)
            )
            metadata_row = cursor.fetchone()
            if metadata_row and metadata_row["metadata"]:
                return json.loads(metadata_row["metadata"])

        return None

    def store(
        self,
        call_id: CallIdentifier,
        response: str,
        response_id: str,
        *,
        save_to_file: bool = True,
    ) -> Optional[int]:
        """
        Store a response in SQLite.

        :param call_id: The task identifier containing checkpoint, doc_hash, seq_id, and session_id.
        :param response: The response content to store.
        :param response_id: The response ID to store.
        :param provider_type: The name of the provider (e.g., "openai").
        :param save_to_file: Whether to commit the transaction immediately (ignored - always commits).
        :returns: The seq_id where the response was stored.
        """
        checkpoint = call_id["checkpoint"]
        doc_hash = call_id["doc_hash"]
        seq_id = call_id["seq_id"]
        session_id = call_id["session_id"]
        provider_type = call_id.get("provider_type")

        conn = self._get_connection(checkpoint)

        try:
            # Check if doc_hash already exists
            cursor = conn.execute(
                "SELECT seq_id FROM responses WHERE doc_hash = ?", (doc_hash,)
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing record, setting seq_id and session_id if provided
                conn.execute(
                    "UPDATE responses SET response = ?, response_id = ?, seq_id = ?, session_id = ?, provider_type = ? WHERE doc_hash = ?",
                    (
                        response,
                        response_id,
                        seq_id,
                        session_id,
                        provider_type,
                        doc_hash,
                    ),
                )
                actual_seq_id = seq_id
            else:
                # Insert new record with seq_id and session_id
                cursor = conn.execute(
                    "INSERT INTO responses (seq_id, session_id, doc_hash, response, response_id, provider_type) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        seq_id,
                        session_id,
                        doc_hash,
                        response,
                        response_id,
                        provider_type,
                    ),
                )
                actual_seq_id = seq_id

            # Always commit immediately for thread safety
            conn.commit()
            return actual_seq_id

        except sqlite3.Error as e:
            conn.rollback()
            raise RuntimeError(f"SQLite error while storing response: {e}")

    def store_metadata(
        self,
        call_id: CallIdentifier,
        response_id: str,
        metadata: dict,
    ) -> None:
        """
        Store metadata in SQLite.

        :param call_id: The task identifier containing checkpoint, doc_hash, seq_id, and session_id.
        :param response_id: The response ID to store.
        :param metadata: The metadata to store.
        :param provider_type: The name of the provider (e.g., "openai").
        """
        checkpoint = call_id["checkpoint"]
        conn = self._get_connection(checkpoint)

        provider_type = call_id.get("provider_type", None)

        try:
            # Serialize metadata to JSON
            metadata_json = json.dumps(metadata) if metadata else None

            if metadata_json:
                # Insert or replace metadata for this response_id
                conn.execute(
                    "INSERT OR REPLACE INTO metadata (response_id, metadata, provider_type) VALUES (?, ?, ?)",
                    (response_id, metadata_json, provider_type),
                )

            # Always commit immediately for thread safety
            conn.commit()

        except sqlite3.Error as e:
            conn.rollback()
            raise RuntimeError(f"SQLite error while storing metadata: {e}")

    def _transfer_metadata_to_parquet(self, checkpoint: Optional[str]) -> None:
        """
        Transfer OpenAI metadata from SQLite to Parquet files.

        This method:
        1. Extracts all OpenAI metadata from the database
        2. Processes it using openai_metadata_sinker
        3. Merges with existing Parquet data (if any)
        4. Writes to temporary files, then swaps them atomically
        5. Removes transferred data from SQLite only on success

        :param checkpoint: The checkpoint to transfer metadata for
        """
        conn = self._get_connection(checkpoint)

        try:
            # Get all OpenAI metadata from the database
            cursor = conn.execute("""
                SELECT m.response_id, m.metadata 
                FROM metadata m 
                WHERE m.provider_type = 'openai' OR m.provider_type IS NULL
            """)
            metadata_rows = cursor.fetchall()

            if not metadata_rows:
                return  # Nothing to transfer

            # Extract metadata strings for processing
            metadata_strings = []
            response_ids_to_delete = []

            for row in metadata_rows:
                response_id = row["response_id"]
                metadata_json = row["metadata"]

                if metadata_json:
                    metadata_strings.append(metadata_json)
                    response_ids_to_delete.append(response_id)

            if not metadata_strings:
                return  # No valid metadata to transfer

            # Process metadata using the existing sinker function
            processed_data = openai_metadata_sinker(metadata_strings)
            responses_df = processed_data["responses"]
            messages_df = processed_data["messages"]

            if responses_df.is_empty() and messages_df.is_empty():
                return  # No data to save

            # Create metadata directory
            datastore_dir = self.file_manager.allocate_datastore(checkpoint)
            metadata_dir = datastore_dir / "apimeta"
            metadata_dir.mkdir(exist_ok=True)

            # Define file paths
            responses_parquet = metadata_dir / "responses.parquet"
            messages_parquet = metadata_dir / "messages.parquet"
            responses_tmp = metadata_dir / "responses.parquet.tmp"
            messages_tmp = metadata_dir / "messages.parquet.tmp"

            try:
                # Handle responses dataframe
                if not responses_df.is_empty():
                    final_responses_df = responses_df

                    # Merge with existing responses data if file exists
                    if responses_parquet.exists():
                        existing_responses = pl.read_parquet(responses_parquet)
                        # Concatenate and deduplicate based on all columns
                        final_responses_df = pl.concat(
                            [existing_responses, responses_df]
                        ).unique()

                    # Write to temporary file
                    final_responses_df.write_parquet(responses_tmp)

                # Handle messages dataframe
                if not messages_df.is_empty():
                    final_messages_df = messages_df

                    # Merge with existing messages data if file exists
                    if messages_parquet.exists():
                        existing_messages = pl.read_parquet(messages_parquet)
                        # Concatenate and deduplicate based on all columns
                        final_messages_df = pl.concat(
                            [existing_messages, messages_df]
                        ).unique()

                    # Write to temporary file
                    final_messages_df.write_parquet(messages_tmp)

                # Atomic swap: move temporary files to final locations
                if responses_tmp.exists():
                    if responses_parquet.exists():
                        responses_parquet.unlink()  # Remove old file
                    responses_tmp.rename(responses_parquet)

                if messages_tmp.exists():
                    if messages_parquet.exists():
                        messages_parquet.unlink()  # Remove old file
                    messages_tmp.rename(messages_parquet)

                # Success! Now remove the transferred metadata from SQLite
                if response_ids_to_delete:
                    placeholders = ",".join(["?" for _ in response_ids_to_delete])
                    conn.execute(
                        f"DELETE FROM metadata WHERE response_id IN ({placeholders})",
                        response_ids_to_delete,
                    )
                    conn.commit()

                    # VACUUM database for debugging (reclaim space after deletion)
                    # print(f"Debug: VACUUMing database for checkpoint '{checkpoint}' after deleting {len(response_ids_to_delete)} metadata records")
                    # conn.execute("VACUUM")

            except Exception as e:
                # Clean up temporary files on error
                for tmp_file in [responses_tmp, messages_tmp]:
                    if tmp_file.exists():
                        try:
                            tmp_file.unlink()
                        except Exception:
                            pass  # Ignore cleanup errors
                raise RuntimeError(f"Error transferring metadata to Parquet: {e}")

        except sqlite3.Error as e:
            raise RuntimeError(f"SQLite error during metadata transfer: {e}")

    def persist(self, checkpoint: Optional[str] = None) -> None:
        """
        Persist (commit) changes to SQLite database(s) and transfer metadata to Parquet files.

        This method now also transfers OpenAI metadata from SQLite to Parquet files
        for better storage efficiency.

        :param checkpoint: The checkpoint to persist (if None, process all active checkpoints).
        """
        for active_checkpoint in self._dirty_checkpoints:
            try:
                pass
                self._transfer_metadata_to_parquet(active_checkpoint)
            except Exception as e:
                # Log the error but don't fail the persist operation
                print(
                    f"Warning: Failed to transfer metadata to Parquet for checkpoint '{active_checkpoint}': {e}"
                )

        # Note: SQLite implementation always commits immediately, so this is a no-op
        # for the actual database persistence

    def close(self, checkpoint: Optional[str] = None) -> None:
        """
        Close SQLite connection(s) for the current thread.

        :param checkpoint: The checkpoint connection to close (if None, close all connections).
        """
        connections = self._get_connections()

        if checkpoint is not None:
            if checkpoint in connections:
                connections[checkpoint].close()
                del connections[checkpoint]
        else:
            # Close all connections for current thread
            for conn in connections.values():
                conn.close()
            connections.clear()

    def __del__(self):
        """
        Cleanup: close all connections when the object is destroyed.
        Note: Only closes connections from the current thread to avoid threading issues.
        """
        try:
            # Only try to close connections if we're in a thread that has them
            if hasattr(self, "_local") and hasattr(self._local, "connections"):
                self.close()
        except Exception:
            # Ignore any errors during cleanup in destructor
            pass
