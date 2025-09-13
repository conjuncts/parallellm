import sqlite3
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
        self._connections: dict[str, sqlite3.Connection] = {}
        self._dirty_stages: set[str] = set()

    def _get_connection(self, stage: str) -> sqlite3.Connection:
        """Get or create SQLite connection for a stage"""
        if stage not in self._connections:
            # Create database file using file manager
            directory = self.file_manager.allocate_datastore(stage)
            db_path = directory / "datastore.db"

            # Create connection
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row  # Enable dict-like access to rows

            # Create table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS responses (
                    seq_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_hash TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(doc_hash)
                )
            """)

            # Create index for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_hash ON responses(doc_hash)
            """)

            conn.commit()
            self._connections[stage] = conn

        return self._connections[stage]

    def retrieve(
        self, stage: str, doc_hash: str, seq_id: Optional[int] = None
    ) -> Optional[str]:
        """
        Retrieve a response from SQLite.

        :param stage: The stage of the response.
        :param doc_hash: The document hash of the response.
        :param seq_id: The sequential ID of the response (optional).
        :returns: The retrieved response content.
        """
        conn = self._get_connection(stage)

        # If seq_id is provided, try direct lookup first for optimization
        if seq_id is not None:
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

    def store(
        self,
        stage: str,
        doc_hash: str,
        response: str,
        seq_id: Optional[int] = None,
        save_to_file: bool = True,
    ) -> Optional[int]:
        """
        Store a response in SQLite.

        :param stage: The stage of the response.
        :param doc_hash: The document hash of the response.
        :param response: The response content to store.
        :param seq_id: The sequential ID of the response (optional).
        :param save_to_file: Whether to commit the transaction immediately.
        :returns: The seq_id where the response was stored.
        """
        conn = self._get_connection(stage)

        try:
            if seq_id is not None:
                # Try to insert with specific seq_id
                cursor = conn.execute(
                    "INSERT OR REPLACE INTO responses (seq_id, doc_hash, response) VALUES (?, ?, ?)",
                    (seq_id, doc_hash, response),
                )
                actual_seq_id = seq_id
            else:
                # Check if doc_hash already exists
                cursor = conn.execute(
                    "SELECT seq_id FROM responses WHERE doc_hash = ?", (doc_hash,)
                )
                existing = cursor.fetchone()

                if existing:
                    # Update existing record
                    conn.execute(
                        "UPDATE responses SET response = ? WHERE doc_hash = ?",
                        (response, doc_hash),
                    )
                    actual_seq_id = existing["seq_id"]
                else:
                    # Insert new record (seq_id will be auto-generated)
                    cursor = conn.execute(
                        "INSERT INTO responses (doc_hash, response) VALUES (?, ?)",
                        (doc_hash, response),
                    )
                    actual_seq_id = cursor.lastrowid

            # Mark stage as dirty
            self._dirty_stages.add(stage)

            # Commit immediately if save_to_file is True
            if save_to_file:
                conn.commit()
                self._dirty_stages.discard(stage)

            return actual_seq_id

        except sqlite3.Error as e:
            conn.rollback()
            raise RuntimeError(f"SQLite error while storing response: {e}")

    def persist(self, stage: Optional[str] = None) -> None:
        """
        Persist (commit) changes to SQLite database(s).

        :param stage: The stage to persist (if None, persist all stages with changes).
        """
        if stage is not None:
            # Persist specific stage
            if stage in self._dirty_stages and stage in self._connections:
                self._connections[stage].commit()
                self._dirty_stages.discard(stage)
        else:
            # Persist all dirty stages
            for dirty_stage in list(self._dirty_stages):
                if dirty_stage in self._connections:
                    self._connections[dirty_stage].commit()
                self._dirty_stages.discard(dirty_stage)

    def close(self, stage: Optional[str] = None) -> None:
        """
        Close SQLite connection(s).

        :param stage: The stage connection to close (if None, close all connections).
        """
        if stage is not None:
            if stage in self._connections:
                self._connections[stage].close()
                del self._connections[stage]
                self._dirty_stages.discard(stage)
        else:
            # Close all connections
            for conn in self._connections.values():
                conn.close()
            self._connections.clear()
            self._dirty_stages.clear()

    def __del__(self):
        """Cleanup: close all connections when the object is destroyed"""
        self.close()
