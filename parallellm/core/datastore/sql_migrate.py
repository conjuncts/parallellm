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


# Helper function to check if a table exists
def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
        (table_name,)
    )
    return cursor.fetchone() is not None


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
        if table_exists(conn, "batch_pending"):
            cursor = conn.execute("PRAGMA table_info(batch_pending)")
            columns = [row[1] for row in cursor.fetchall()]
            if "is_pending" not in columns:
                conn.execute(
                    "ALTER TABLE batch_pending ADD COLUMN is_pending BOOLEAN DEFAULT 1"
                )

        # Remove UNIQUE constraints by recreating tables without them
        # SQLite doesn't support ALTER TABLE DROP CONSTRAINT, so we need to recreate
        _remove_unique_constraints(conn)

        # Remove provider_type column from anon_responses table
        _remove_provider_type_from_anon_responses(conn)

        # Add agent_name, seq_id, session_id columns to metadata table
        _add_metadata_columns(conn)

        # Add tag column to metadata table if it doesn't exist
        cursor = conn.execute("PRAGMA table_info(metadata)")
        columns = [row[1] for row in cursor.fetchall()]
        if "tag" not in columns:
            conn.execute("ALTER TABLE metadata ADD COLUMN tag TEXT")

        # Add tag column to batch_pending table if it doesn't exist
        if table_exists(conn, "batch_pending"):
            cursor = conn.execute("PRAGMA table_info(batch_pending)")
            columns = [row[1] for row in cursor.fetchall()]
            if "tag" not in columns:
                conn.execute("ALTER TABLE batch_pending ADD COLUMN tag TEXT")

        # Migrate default-agent to empty string
        _migrate_default_agent_to_empty_string(conn)

    except sqlite3.Error as e:
        # If migration fails, continue - tables will be created fresh
        db_label = "main database" if db_name is None else f"{db_name} database"
        print(f"Warning: Schema migration failed for {db_label}: {e}")
        return False
    return True


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
                    tool_calls TEXT
                )
            """)

            # Copy data from old table to new table
            conn.execute("""
                INSERT INTO anon_responses_new 
                SELECT id, agent_name, seq_id, session_id, doc_hash, response, response_id, tool_calls
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

            conn.execute("COMMIT")

    except sqlite3.Error as e:
        # Rollback on error
        try:
            conn.execute("ROLLBACK")
        except:
            pass
        raise RuntimeError(f"Failed to remove UNIQUE constraints: {e}")


def _remove_unique_constraint(conn: sqlite3.Connection, table_name: str, unique_columns: list[str]) -> None:
    """
    Remove UNIQUE constraint from a specified table by recreating it without the constraint.

    :param conn: SQLite connection
    :param table_name: Name of the table to modify
    :param unique_columns: List of column names that were part of the UNIQUE constraint
    """
    try:
        # Check if table has UNIQUE constraint by examining the schema
        cursor = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        row = cursor.fetchone()

        if not row or "UNIQUE" not in row[0]:
            return  # No UNIQUE constraint found, nothing to do

        # Get current table structure
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()
        
        # Build column definitions without UNIQUE constraint
        column_defs = []
        for col in columns_info:
            col_name, col_type, not_null, default_val, pk = col[1], col[2], col[3], col[4], col[5]
            col_def = f"{col_name} {col_type}"
            if pk:
                col_def += " PRIMARY KEY"
                if "AUTOINCREMENT" in row[0]:
                    col_def += " AUTOINCREMENT"
            elif not_null:
                col_def += " NOT NULL"
            if default_val is not None:
                col_def += f" DEFAULT {default_val}"
            column_defs.append(col_def)

        conn.execute("BEGIN TRANSACTION")

        # Create new table without UNIQUE constraint
        new_table_name = f"{table_name}_new"
        create_sql = f"CREATE TABLE {new_table_name} ({', '.join(column_defs)})"
        conn.execute(create_sql)

        # Copy data from old table to new table
        column_names = [col[1] for col in columns_info]
        columns_str = ', '.join(column_names)
        conn.execute(f"INSERT INTO {new_table_name} SELECT {columns_str} FROM {table_name}")

        # Drop old table and rename new table
        conn.execute(f"DROP TABLE {table_name}")
        conn.execute(f"ALTER TABLE {new_table_name} RENAME TO {table_name}")

        # Recreate indexes (excluding the UNIQUE constraint)
        cursor = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=?",
            (table_name,)
        )
        for index_row in cursor.fetchall():
            if index_row[0] and "UNIQUE" not in index_row[0]:
                # Recreate non-unique indexes
                conn.execute(index_row[0].replace(f"CREATE INDEX", "CREATE INDEX IF NOT EXISTS"))

        conn.execute("COMMIT")

    except sqlite3.Error as e:
        # Rollback on error
        try:
            conn.execute("ROLLBACK")
        except:
            pass
        raise RuntimeError(f"Failed to remove UNIQUE constraint from {table_name}: {e}")


def _drop_column(conn: sqlite3.Connection, table_name: str, column: str) -> None:
    """
    Drop a column from a specified table by recreating the table without the column.

    :param conn: SQLite connection
    :param table_name: Name of the table to modify
    :param column: Name of the column to drop
    """
    try:
        # Check if table exists
        if not table_exists(conn, table_name):
            return  # Table doesn't exist, nothing to do

        # Get current table structure
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()
        
        # Check if column exists
        column_names = [col[1] for col in columns_info]
        if column not in column_names:
            return  # Column doesn't exist, nothing to do

        # Filter out the column to be dropped
        remaining_columns = [col for col in columns_info if col[1] != column]
        
        # Build column definitions for remaining columns
        column_defs = []
        remaining_names = []
        for col in remaining_columns:
            col_name, col_type, not_null, default_val, pk = col[1], col[2], col[3], col[4], col[5]
            col_def = f"{col_name} {col_type}"
            if pk:
                col_def += " PRIMARY KEY"
                # Check if original table had AUTOINCREMENT
                cursor = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,)
                )
                table_sql = cursor.fetchone()
                if table_sql and "AUTOINCREMENT" in table_sql[0]:
                    col_def += " AUTOINCREMENT"
            elif not_null:
                col_def += " NOT NULL"
            if default_val is not None:
                col_def += f" DEFAULT {default_val}"
            column_defs.append(col_def)
            remaining_names.append(col_name)

        conn.execute("BEGIN TRANSACTION")

        # Create new table without the dropped column
        temp_table_name = f"{table_name}_temp"
        create_sql = f"CREATE TABLE {temp_table_name} ({', '.join(column_defs)})"
        conn.execute(create_sql)

        # Copy data from old table to new table (excluding dropped column)
        columns_str = ', '.join(remaining_names)
        conn.execute(f"INSERT INTO {temp_table_name} SELECT {columns_str} FROM {table_name}")

        # Drop old table and rename new table
        conn.execute(f"DROP TABLE {table_name}")
        conn.execute(f"ALTER TABLE {temp_table_name} RENAME TO {table_name}")

        # Recreate indexes (excluding any that referenced the dropped column)
        cursor = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=?",
            (table_name,)
        )
        for index_row in cursor.fetchall():
            if index_row[0] and column not in index_row[0]:
                # Only recreate indexes that don't reference the dropped column
                conn.execute(index_row[0].replace(f"CREATE INDEX", "CREATE INDEX IF NOT EXISTS"))

        conn.execute("COMMIT")

    except sqlite3.Error as e:
        # Rollback on error
        try:
            conn.execute("ROLLBACK")
        except:
            pass
        raise RuntimeError(f"Failed to drop column {column} from {table_name}: {e}")


def _add_columns(conn: sqlite3.Connection, table_name: str, columns: list[tuple[str, str, str]]) -> None:
    """
    Add new columns to a specified table.

    :param conn: SQLite connection
    :param table_name: Name of the table to modify
    :param columns: List of tuples (column_name, column_type, constraints) 
                   e.g., [("new_col1", "TEXT", ""), ("new_col2", "INTEGER", "NOT NULL DEFAULT 0")]
    """
    try:
        # Check if table exists
        if not table_exists(conn, table_name):
            raise RuntimeError(f"Table {table_name} does not exist")

        # Get current table structure
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        existing_columns = [row[1] for row in cursor.fetchall()]

        # Add each column if it doesn't already exist
        for column_name, column_type, constraints in columns:
            if column_name not in existing_columns:
                # Build the full column definition
                column_def = f"{column_name} {column_type}"
                if constraints.strip():
                    column_def += f" {constraints.strip()}"
                
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_def}")
                
                # Create index for the new column if it's a commonly indexed type
                if column_type.upper() in ["TEXT", "INTEGER", "REAL", "NUMERIC"] and "PRIMARY KEY" not in constraints.upper():
                    index_name = f"idx_{table_name}_{column_name}"
                    conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({column_name})")

    except sqlite3.Error as e:
        raise RuntimeError(f"Failed to add columns to {table_name}: {e}")

def _remove_provider_type_from_anon_responses(conn: sqlite3.Connection) -> None:
    """
    Remove provider_type column from anon_responses table.
    This is done by recreating the table without the column.

    :param conn: SQLite connection
    """
    try:
        # Check if anon_responses has provider_type column
        cursor = conn.execute("PRAGMA table_info(anon_responses)")
        columns = [row[1] for row in cursor.fetchall()]

        if "provider_type" not in columns:
            return  # Column already removed, nothing to do

        # Table has provider_type column, need to recreate it
        conn.execute("BEGIN TRANSACTION")

        # Create new table without provider_type column
        conn.execute("""
            CREATE TABLE anon_responses_temp (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT,
                seq_id INTEGER NOT NULL,
                session_id INTEGER NOT NULL,
                doc_hash TEXT NOT NULL,
                response TEXT NOT NULL,
                response_id TEXT,
                tool_calls TEXT
            )
        """)

        # Copy data from old table to new table (excluding provider_type)
        conn.execute("""
            INSERT INTO anon_responses_temp 
            SELECT id, agent_name, seq_id, session_id, doc_hash, response, response_id, tool_calls
            FROM anon_responses
        """)

        # Drop old table and rename new table
        conn.execute("DROP TABLE anon_responses")
        conn.execute("ALTER TABLE anon_responses_temp RENAME TO anon_responses")

        # Recreate indexes (without provider_type index)
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

        conn.execute("COMMIT")

    except sqlite3.Error as e:
        # Rollback on error
        try:
            conn.execute("ROLLBACK")
        except:
            pass
        raise RuntimeError(f"Failed to remove provider_type column: {e}")


def _add_metadata_columns(conn: sqlite3.Connection) -> None:
    """
    Add agent_name, seq_id, and session_id columns to metadata table.

    :param conn: SQLite connection
    """
    try:
        # Check what columns exist in metadata table
        cursor = conn.execute("PRAGMA table_info(metadata)")
        columns = [row[1] for row in cursor.fetchall()]

        # Add agent_name column if it doesn't exist
        if "agent_name" not in columns:
            conn.execute("ALTER TABLE metadata ADD COLUMN agent_name TEXT")

        # Add seq_id column if it doesn't exist
        if "seq_id" not in columns:
            conn.execute("ALTER TABLE metadata ADD COLUMN seq_id INTEGER")

        # Add session_id column if it doesn't exist
        if "session_id" not in columns:
            conn.execute("ALTER TABLE metadata ADD COLUMN session_id INTEGER")

        # Create indexes for new columns
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metadata_agent_name ON metadata(agent_name)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metadata_seq_id ON metadata(seq_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metadata_session_id ON metadata(session_id)"
        )

    except sqlite3.Error as e:
        raise RuntimeError(f"Failed to add metadata columns: {e}")


def _migrate_default_agent_to_empty_string(conn: sqlite3.Connection) -> None:
    """
    Migrate existing 'default-agent' values to empty strings in all relevant tables.
    
    :param conn: SQLite connection
    """


    try:
        # Update anon_responses table if it exists
        if table_exists(conn, "anon_responses"):
            conn.execute(
                "UPDATE anon_responses SET agent_name = '' WHERE agent_name = 'default-agent'"
            )
        
        # Update metadata table if it exists
        if table_exists(conn, "metadata"):
            conn.execute(
                "UPDATE metadata SET agent_name = '' WHERE agent_name = 'default-agent'"
            )
        
        # Update batch_pending table if it exists
        if table_exists(conn, "batch_pending"):
            conn.execute(
                "UPDATE batch_pending SET agent_name = '' WHERE agent_name = 'default-agent'"
            )
        
        # Update errors table if it exists
        if table_exists(conn, "errors"):
            conn.execute(
                "UPDATE errors SET agent_name = '' WHERE agent_name = 'default-agent'"
            )

    except sqlite3.Error as e:
        # Log but don't fail - migration should be non-destructive
        print(f"Warning: Failed to migrate some default-agent values: {e}")
        pass


def migrate_all_databases_in_directory(root_dir: str) -> None:
    """
    Utility to recursively migrate all datastore.db files in a given root directory.
    
    :param root_dir: Root directory to recursively search for datastore.db files
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"Directory {root_dir} does not exist")
        return
    
    migrated_count = 0
    failed_count = 0
    
    # Recursively walk through all directories looking for datastore.db files
    for db_path in root_path.rglob("datastore.db"):
        print(f"Migrating database: {db_path}")
        try:
            # Connect to the database and run migrations
            with sqlite3.connect(str(db_path)) as conn:
                # Use relative path from root as db_name for logging
                relative_path = db_path.relative_to(root_path)
                outcome = _migrate_sql_schema(conn, str(relative_path.parent))
                conn.commit()
            if outcome:
                print(f"  ✓ Successfully migrated {db_path}")
                migrated_count += 1
            else:
                print(f"  ✗ Failed to migrate {db_path}: {e}")
                failed_count += 1
        except Exception as e:
            print(f"  ✗ Failed to migrate {db_path}: {e}")
            failed_count += 1
    
    print(f"\nMigration complete: {migrated_count} succeeded, {failed_count} failed")


if __name__ == "__main__":
    # Utility to migrate all databases in a directory
    # migrate_all_databases_in_directory(".pllm")
    migrate_all_databases_in_directory("tests/data")

