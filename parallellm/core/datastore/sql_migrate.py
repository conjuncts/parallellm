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

        # Add tool_calls column to chk_responses table if it doesn't exist
        cursor = conn.execute("PRAGMA table_info(chk_responses)")
        columns = [row[1] for row in cursor.fetchall()]
        if "tool_calls" not in columns:
            conn.execute("ALTER TABLE chk_responses ADD COLUMN tool_calls TEXT")

    except sqlite3.Error as e:
        # If migration fails, continue - tables will be created fresh
        db_label = "main database" if db_name is None else f"{db_name} database"
        print(f"Warning: Schema migration failed for {db_label}: {e}")
        pass


if __name__ == "__main__":
    root = ".pllm"
    for subdir in os.listdir(root):
        fm = FileManager(f"{root}/{subdir}")
        # _tmp_migrate_datastores(fm)
        pass
