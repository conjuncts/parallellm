import sqlite3
import json
import threading
from typing import Optional
from pathlib import Path

import polars as pl

from parallellm.core.datastore.base import Datastore
from parallellm.core.datastore.sql_migrate import (
    _check_and_migrate,
    _migrate_sql_schema,
)
from parallellm.core.lake.sequester import sequester_openai_metadata
from parallellm.file_io.file_manager import FileManager
from parallellm.types import CallIdentifier


def _sql_table_to_dataframe(
    conn: sqlite3.Connection, sql_query: str, params: tuple = ()
) -> Optional[pl.DataFrame]:
    """
    Extract data from a SQL query and convert to a Polars DataFrame.

    :param conn: SQLite connection
    :param sql_query: SQL query to execute
    :param params: Parameters for the SQL query
    :returns: Polars DataFrame with the query results, or None if no data
    """
    cursor = conn.execute(sql_query, params)
    rows = cursor.fetchall()

    if not rows:
        return None

    # Convert to DataFrame
    columns = [description[0] for description in cursor.description]
    data = [dict(zip(columns, row)) for row in rows]
    df = pl.DataFrame(data)

    return df if not df.is_empty() else None


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
        self._is_dirty = False

        # Check if migration is needed (only on first initialization)
        self._check_and_migrate()

    def _check_and_migrate(self) -> None:
        """
        Check if old directory-based structure exists and migrate if needed.
        Only runs once per datastore directory.
        """
        return _check_and_migrate(self)

    def _get_connections(self) -> dict[str, sqlite3.Connection]:
        """Get the connections dictionary for the current thread"""
        if not hasattr(self._local, "connections"):
            self._local.connections = {}
        return self._local.connections

    def _get_connection(self, db_name: Optional[str] = None) -> sqlite3.Connection:
        """
        Get or create SQLite connection for a database.

        :param db_name: Name identifier for the connection (None for main database, or custom names for additional connections)
        :returns: SQLite connection for the specified database
        """
        connections = self._get_connections()

        # Use "main" as key for None db_name
        connection_key = db_name if db_name is not None else "main"

        if connection_key not in connections:
            # Get the base datastore directory
            datastore_dir = self.file_manager.allocate_datastore()

            # Use main datastore file for None/main, or custom named files for others
            if db_name is None:
                db_path = datastore_dir / "datastore.db"
            else:
                db_path = datastore_dir / f"{db_name}-datastore.db"

            # Create connection
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row  # Enable dict-like access to rows

            # For main database (db_name is None), create both tables
            if db_name is None:
                # Anonymous responses table: no checkpoint column, agent_name can be NULL
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS anon_responses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_name TEXT,
                        seq_id INTEGER NOT NULL,
                        session_id INTEGER NOT NULL,
                        doc_hash TEXT NOT NULL,
                        response TEXT NOT NULL,
                        response_id TEXT,
                        provider_type TEXT,
                        UNIQUE(agent_name, doc_hash)
                    )
                """)

                # Checkpoint responses table: checkpoint is NOT NULL
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chk_responses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_name TEXT,
                        checkpoint TEXT NOT NULL,
                        seq_id INTEGER NOT NULL,
                        session_id INTEGER NOT NULL,
                        doc_hash TEXT NOT NULL,
                        response TEXT NOT NULL,
                        response_id TEXT,
                        provider_type TEXT,
                        UNIQUE(agent_name, checkpoint, doc_hash)
                    )
                """)

                # Create metadata table (shared between both response tables)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        response_id TEXT,
                        metadata TEXT NOT NULL,
                        provider_type TEXT,
                        UNIQUE(response_id)
                    )
                """)

                # Migrate existing schema if needed
                _migrate_sql_schema(conn, None)

                # Create indexes for anon_responses table
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_anon_agent_name ON anon_responses(agent_name)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_anon_agent_doc_hash ON anon_responses(agent_name, doc_hash)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_anon_doc_hash ON anon_responses(doc_hash)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_anon_session_id ON anon_responses(session_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_anon_seq_id ON anon_responses(seq_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_anon_response_id ON anon_responses(response_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_anon_provider_type ON anon_responses(provider_type)
                """)

                # Create indexes for chk_responses table
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chk_agent_checkpoint ON chk_responses(agent_name, checkpoint)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chk_agent_checkpoint_doc_hash ON chk_responses(agent_name, checkpoint, doc_hash)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chk_doc_hash ON chk_responses(doc_hash)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chk_session_id ON chk_responses(session_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chk_seq_id ON chk_responses(seq_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chk_response_id ON chk_responses(response_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chk_provider_type ON chk_responses(provider_type)
                """)

                # Create indexes for metadata table
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metadata_response_id ON metadata(response_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metadata_provider_type ON metadata(provider_type)
                """)
            else:
                # For custom named databases, create a simple responses table (legacy support)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS responses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_name TEXT,
                        seq_id INTEGER NOT NULL,
                        session_id INTEGER NOT NULL,
                        doc_hash TEXT NOT NULL,
                        response TEXT NOT NULL,
                        response_id TEXT,
                        provider_type TEXT,
                        UNIQUE(agent_name, doc_hash)
                    )
                """)

                # Migrate existing schema if needed
                _migrate_sql_schema(conn, db_name)

            conn.commit()
            connections[connection_key] = conn

        self._is_dirty = True
        return connections[connection_key]

    def retrieve(self, call_id: CallIdentifier) -> Optional[str]:
        """
        Retrieve a response from SQLite.

        :param call_id: The task identifier containing checkpoint, doc_hash, and seq_id.
        :returns: The retrieved response content.
        """
        checkpoint = call_id["checkpoint"]
        doc_hash = call_id["doc_hash"]
        seq_id = call_id["seq_id"]
        agent_name = call_id["agent_name"]
        # session_id NOT relevant for lookup

        # Get main database connection
        conn = self._get_connection(None)

        # Determine table and build WHERE clause components
        table_name = "chk_responses" if checkpoint is not None else "anon_responses"

        # Base WHERE conditions (always present)
        where_conditions = ["doc_hash = ?"]
        params = [doc_hash]

        # Add agent_name condition
        if agent_name is not None:
            where_conditions.append("agent_name = ?")
            params.append(agent_name)
        else:
            where_conditions.append("agent_name IS NULL")

        # Add checkpoint condition if using checkpoint table
        if checkpoint is not None:
            where_conditions.append("checkpoint = ?")
            params.append(checkpoint)

        # Try with seq_id first (most specific)
        full_where = " AND ".join(where_conditions + ["seq_id = ?"])
        full_params = params + [seq_id]

        cursor = conn.execute(
            f"SELECT response FROM {table_name} WHERE {full_where}", full_params
        )
        row = cursor.fetchone()
        if row:
            return row["response"]

        # Fallback: try without seq_id (less specific)
        base_where = " AND ".join(where_conditions)
        cursor = conn.execute(
            f"SELECT response FROM {table_name} WHERE {base_where}", params
        )
        row = cursor.fetchone()
        return row["response"] if row else None

    def retrieve_metadata(self, call_id: CallIdentifier) -> Optional[dict]:
        """
        Retrieve metadata from SQLite.

        :param call_id: The task identifier containing checkpoint, doc_hash, and seq_id.
        :returns: The retrieved metadata as a dictionary, or None if not found.
        """
        # TODO: does not work because does not read from parquet

        checkpoint = call_id["checkpoint"]
        doc_hash = call_id["doc_hash"]
        seq_id = call_id["seq_id"]
        agent_name = call_id["agent_name"]

        # Get main database connection
        conn = self._get_connection(None)

        # Determine table and build WHERE clause components
        table_name = "chk_responses" if checkpoint is not None else "anon_responses"

        # Base WHERE conditions (always present)
        where_conditions = ["doc_hash = ?"]
        params = [doc_hash]

        # Add agent_name condition
        if agent_name is not None:
            where_conditions.append("agent_name = ?")
            params.append(agent_name)
        else:
            where_conditions.append("agent_name IS NULL")

        # Add checkpoint condition if using checkpoint table
        if checkpoint is not None:
            where_conditions.append("checkpoint = ?")
            params.append(checkpoint)

        # Try with seq_id first (most specific)
        full_where = " AND ".join(where_conditions + ["seq_id = ?"])
        full_params = params + [seq_id]

        cursor = conn.execute(
            f"SELECT response_id FROM {table_name} WHERE {full_where}", full_params
        )
        row = cursor.fetchone()

        # If not found, try without seq_id (less specific)
        if not row or not row["response_id"]:
            base_where = " AND ".join(where_conditions)
            cursor = conn.execute(
                f"SELECT response_id FROM {table_name} WHERE {base_where}", params
            )
            row = cursor.fetchone()

        # If we have a response_id, get the metadata
        if row and row["response_id"]:
            response_id = row["response_id"]
            cursor = conn.execute(
                "SELECT metadata FROM metadata WHERE response_id = ?",
                (response_id,),
            )
            metadata_row = cursor.fetchone()
            if metadata_row and metadata_row["metadata"]:
                return json.loads(metadata_row["metadata"])

        return None

    def store(
        self, call_id: CallIdentifier, response: str, response_id: str
    ) -> Optional[int]:
        """
        Store a response in SQLite.

        :param call_id: The task identifier containing checkpoint, doc_hash, seq_id, and session_id.
        :param response: The response content to store.
        :param response_id: The response ID to store.
        :returns: The seq_id where the response was stored.
        """
        checkpoint = call_id["checkpoint"]
        doc_hash = call_id["doc_hash"]
        seq_id = call_id["seq_id"]
        session_id = call_id["session_id"]
        agent_name = call_id["agent_name"]
        provider_type = call_id.get("provider_type")

        # Get main database connection
        conn = self._get_connection(None)

        # Determine table and build WHERE/INSERT clause components
        table_name = "chk_responses" if checkpoint is not None else "anon_responses"

        # Build WHERE conditions for checking existing records
        where_conditions = ["doc_hash = ?"]
        where_params = [doc_hash]

        # Add agent_name condition
        if agent_name is not None:
            where_conditions.append("agent_name = ?")
            where_params.append(agent_name)
        else:
            where_conditions.append("agent_name IS NULL")

        # Add checkpoint condition if using checkpoint table
        if checkpoint is not None:
            where_conditions.append("checkpoint = ?")
            where_params.append(checkpoint)

        where_clause = " AND ".join(where_conditions)

        try:
            # Check if record already exists
            cursor = conn.execute(
                f"SELECT seq_id FROM {table_name} WHERE {where_clause}", where_params
            )
            existing = cursor.fetchone()

            # Prepare column names and values for INSERT/UPDATE
            columns = [
                "agent_name",
                "seq_id",
                "session_id",
                "doc_hash",
                "response",
                "response_id",
                "provider_type",
            ]
            values = [
                agent_name,
                seq_id,
                session_id,
                doc_hash,
                response,
                response_id,
                provider_type,
            ]

            if checkpoint is not None:
                columns.insert(1, "checkpoint")  # Insert after agent_name
                values.insert(1, checkpoint)

            if existing:
                # Update existing record
                set_clauses = [f"{col} = ?" for col in columns]
                update_sql = f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE {where_clause}"
                conn.execute(update_sql, values + where_params)
            else:
                # Insert new record
                placeholders = ", ".join(["?" for _ in columns])
                insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                conn.execute(insert_sql, values)

            # Always commit immediately for thread safety
            conn.commit()
            return seq_id

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
        """
        provider_type = call_id.get("provider_type", None)

        # Get main database connection
        conn = self._get_connection(None)

        try:
            # Serialize metadata to JSON
            metadata_json = json.dumps(metadata) if metadata else None

            if metadata_json:
                # Insert or replace metadata for this response_id (response_id is unique UUID)
                conn.execute(
                    "INSERT OR REPLACE INTO metadata (response_id, metadata, provider_type) VALUES (?, ?, ?)",
                    (response_id, metadata_json, provider_type),
                )

            # Always commit immediately for thread safety
            conn.commit()

        except sqlite3.Error as e:
            conn.rollback()
            raise RuntimeError(f"SQLite error while storing metadata: {e}")

    def _transfer_metadata_to_parquet(self) -> None:
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
        conn = self._get_connection(None)

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
            sequestered = sequester_openai_metadata(metadata_rows, self.file_manager)
            if sequestered:
                placeholders = ",".join(["?" for _ in sequestered])
                conn.execute(
                    f"DELETE FROM metadata WHERE response_id IN ({placeholders})",
                    sequestered,
                )
                conn.commit()

                # VACUUM database for debugging (reclaim space after deletion)
                # print(f"Debug: VACUUMing database for checkpoint '{checkpoint}' after deleting {len(succeeded)} metadata records")
                # conn.execute("VACUUM")
        except sqlite3.Error as e:
            raise RuntimeError(f"SQLite error during metadata transfer: {e}")

    def persist(self) -> None:
        """
        Persist (commit) changes to SQLite database and transfer metadata to Parquet files.

        And closes all connections. After this call, the datastore is still usable, 
        but current connections are closed (connections will be recreated on demand).

        Also, OpenAI metadata is transferred from SQLite to Parquet files
        for better storage efficiency.
        """
        # Transfer all metadata to parquet
        if hasattr(self, "_local") and hasattr(self._local, "connections"):
            try:
                self._transfer_metadata_to_parquet()
            except Exception as e:
                # Log the error but don't fail the persist operation
                print(f"Warning: Failed to transfer metadata to Parquet: {e}")

        # Note: SQLite implementation always commits immediately, so this is a no-op
        # for the actual database persistence

        # Close all connections to ensure proper cleanup, especially important on Windows
        self.close()

    def close(self, checkpoint: Optional[str] = None) -> None:
        """
        Close SQLite connection(s) for the current thread.

        :param checkpoint: The checkpoint connection to close (if None, close all connections).
        """
        if hasattr(self, "_local") and hasattr(self._local, "connections"):
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
