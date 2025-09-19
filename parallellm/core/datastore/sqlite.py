import sqlite3
import json
import threading
from typing import Optional

from parallellm.core.datastore.base import DataStore
from parallellm.file_io.file_manager import FileManager


class SQLiteDataStore(DataStore):
    """
    SQLite-backed DataStore implementation
    """

    def __init__(self, file_manager: FileManager):
        """
        Initialize SQLite DataStore.

        :param file_manager: FileManager instance to handle file I/O operations
        """
        self.file_manager = file_manager
        # Use threading.local to ensure each thread has its own connections
        self._local = threading.local()

    def _get_connections(self) -> dict[str, sqlite3.Connection]:
        """Get the connections dictionary for the current thread"""
        if not hasattr(self._local, "connections"):
            self._local.connections = {}
        return self._local.connections

    def _get_connection(self, checkpoint: str) -> sqlite3.Connection:
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
                    doc_hash TEXT NOT NULL,
                    response TEXT NOT NULL,
                    response_id TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(doc_hash)
                )
            """)

            # Create index for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_hash ON responses(doc_hash)
            """)

            conn.commit()
            connections[checkpoint] = conn

        return connections[checkpoint]

    def retrieve(self, checkpoint: str, doc_hash: str, seq_id: int) -> Optional[str]:
        """
        Retrieve a response from SQLite.

        :param checkpoint: The checkpoint of the response.
        :param doc_hash: The document hash of the response.
        :param seq_id: The sequential ID of the response.
        :returns: The retrieved response content.
        """
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

    def retrieve_metadata(
        self, checkpoint: str, doc_hash: str, seq_id: int
    ) -> Optional[dict]:
        """
        Retrieve metadata from SQLite.

        :param checkpoint: The checkpoint of the response.
        :param doc_hash: The document hash of the response.
        :param seq_id: The sequential ID of the response.
        :returns: The retrieved metadata as a dictionary, or None if not found.
        """
        conn = self._get_connection(checkpoint)

        # Try direct lookup using seq_id and doc_hash
        cursor = conn.execute(
            "SELECT metadata FROM responses WHERE seq_id = ? AND doc_hash = ?",
            (seq_id, doc_hash),
        )
        row = cursor.fetchone()
        if row and row["metadata"]:
            return json.loads(row["metadata"])

        # Fallback to doc_hash lookup
        cursor = conn.execute(
            "SELECT metadata FROM responses WHERE doc_hash = ?", (doc_hash,)
        )
        row = cursor.fetchone()
        if row and row["metadata"]:
            return json.loads(row["metadata"])
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
    ) -> Optional[int]:
        """
        Store a response in SQLite.

        :param checkpoint: The checkpoint of the response.
        :param doc_hash: The document hash of the response.
        :param response: The response content to store.
        :param response_id: The response ID to store.
        :param seq_id: The sequential ID of the response.
        :param save_to_file: Whether to commit the transaction immediately (ignored - always commits).
        :returns: The seq_id where the response was stored.
        """
        conn = self._get_connection(checkpoint)

        try:
            # Check if doc_hash already exists
            cursor = conn.execute(
                "SELECT seq_id FROM responses WHERE doc_hash = ?", (doc_hash,)
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing record, setting seq_id if provided
                conn.execute(
                    "UPDATE responses SET response = ?, response_id = ?, seq_id = ? WHERE doc_hash = ?",
                    (response, response_id, seq_id, doc_hash),
                )
                actual_seq_id = seq_id
            else:
                # Insert new record with seq_id
                cursor = conn.execute(
                    "INSERT INTO responses (seq_id, doc_hash, response, response_id) VALUES (?, ?, ?, ?)",
                    (seq_id, doc_hash, response, response_id),
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
        checkpoint: str,
        doc_hash: str,
        seq_id: int,
        response_id: str,
        metadata: dict,
    ) -> None:
        """
        Store metadata in SQLite.

        :param checkpoint: The checkpoint of the metadata.
        :param doc_hash: The document hash of the response.
        :param seq_id: The sequential ID of the response.
        :param response_id: The response ID to store.
        :param metadata: The metadata to store.
        """
        conn = self._get_connection(checkpoint)

        try:
            # Serialize metadata to JSON
            metadata_json = json.dumps(metadata) if metadata else None

            # Update by seq_id and doc_hash
            cursor = conn.execute(
                "UPDATE responses SET metadata = ? WHERE seq_id = ? AND doc_hash = ?",
                (metadata_json, seq_id, doc_hash),
            )
            if cursor.rowcount == 0:
                # Record doesn't exist, create it with seq_id and metadata
                conn.execute(
                    "INSERT INTO responses (seq_id, doc_hash, response, response_id, metadata) VALUES (?, ?, '', ?, ?)",
                    (seq_id, doc_hash, response_id, metadata_json),
                )

            # Always commit immediately for thread safety
            conn.commit()

        except sqlite3.Error as e:
            conn.rollback()
            raise RuntimeError(f"SQLite error while storing metadata: {e}")

    def persist(self, checkpoint: Optional[str] = None) -> None:
        """
        Persist (commit) changes to SQLite database(s).

        Note: This implementation always commits immediately, so this method is a no-op.

        :param checkpoint: The checkpoint to persist (ignored - always commits immediately).
        """
        # No-op since we always commit immediately
        pass

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
