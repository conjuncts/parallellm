import os
from typing import TYPE_CHECKING, Optional
from parallellm.file_io.file_manager import FileManager
from pathlib import Path

import sqlite3

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

if TYPE_CHECKING:
    from parallellm.core.datastore.sqlite import SQLiteDatastore


def _check_and_migrate(ds: "SQLiteDatastore") -> None:
    """
    Check if old directory-based structure exists and migrate if needed.
    Only runs once per datastore directory.
    """
    # datastore_dir = ds.file_manager.allocate_datastore()


def _migrate_sql_schema(conn: sqlite3.Connection, db_name: Optional[str]) -> None:
    """
    Migrate SQL schema for a database.

    :param conn: SQLite connection to migrate
    :param db_name: Name of the database (None for main database, or custom names for additional databases)
    """
    try:
        # Add tool_calls column to anon_responses table if it doesn't exist
        cursor = conn.execute("PRAGMA table_info(anon_responses)")
        columns = [row[1] for row in cursor.fetchall()]
        if "tool_calls" not in columns:
            conn.execute("ALTER TABLE anon_responses ADD COLUMN tool_calls TEXT")

        # Add is_pending column to batch_pending table if it doesn't exist
        cursor = conn.execute("PRAGMA table_info(batch_pending)")
        columns = [row[1] for row in cursor.fetchall()]
        if "is_pending" not in columns:
            conn.execute(
                "ALTER TABLE batch_pending ADD COLUMN is_pending BOOLEAN DEFAULT 1"
            )

        # Remove UNIQUE constraints by recreating tables without them
        # SQLite doesn't support ALTER TABLE DROP CONSTRAINT, so we need to recreate
        _remove_unique_constraints(conn)

    except sqlite3.Error as e:
        # If migration fails, continue - tables will be created fresh
        db_label = "main database" if db_name is None else f"{db_name} database"
        print(f"Warning: Schema migration failed for {db_label}: {e}")
        pass


def _remove_unique_constraints(conn: sqlite3.Connection) -> None:
    """
    Remove UNIQUE constraints from anon_responses and chk_responses tables.
    This is done by recreating the tables without the constraints.

    :param conn: SQLite connection
    """
    try:
        # Check if anon_responses has UNIQUE constraint by examining the schema
        cursor = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='anon_responses'"
        )
        row = cursor.fetchone()

        if row and "UNIQUE" in row[0]:
            # Table has UNIQUE constraint, need to recreate it
            conn.execute("BEGIN TRANSACTION")

            # Create new table without UNIQUE constraint
            conn.execute("""
                CREATE TABLE anon_responses_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT,
                    seq_id INTEGER NOT NULL,
                    session_id INTEGER NOT NULL,
                    doc_hash TEXT NOT NULL,
                    response TEXT NOT NULL,
                    response_id TEXT,
                    provider_type TEXT,
                    tool_calls TEXT
                )
            """)

            # Copy data from old table to new table
            conn.execute("""
                INSERT INTO anon_responses_new 
                SELECT id, agent_name, seq_id, session_id, doc_hash, response, response_id, provider_type, tool_calls
                FROM anon_responses
            """)

            # Drop old table and rename new table
            conn.execute("DROP TABLE anon_responses")
            conn.execute("ALTER TABLE anon_responses_new RENAME TO anon_responses")

            # Recreate indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_anon_agent_name ON anon_responses(agent_name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_anon_agent_doc_hash ON anon_responses(agent_name, doc_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_anon_doc_hash ON anon_responses(doc_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_anon_session_id ON anon_responses(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_anon_seq_id ON anon_responses(seq_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_anon_response_id ON anon_responses(response_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_anon_provider_type ON anon_responses(provider_type)"
            )

            conn.execute("COMMIT")

    except sqlite3.Error as e:
        # Rollback on error
        try:
            conn.execute("ROLLBACK")
        except:
            pass
        raise RuntimeError(f"Failed to remove UNIQUE constraints: {e}")


if __name__ == "__main__":
    root = ".pllm"
    for subdir in os.listdir(root):
        fm = FileManager(f"{root}/{subdir}")
        # _tmp_migrate_datastores(fm)
        pass
