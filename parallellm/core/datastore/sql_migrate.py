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


def _tmp_migrate_datastores(fm: FileManager) -> None:
    """
    Migrate from old datastore.db structure to new table-based structure.

    This handles the transition from the old unified structure to the new structure
    with separate anon_responses and chk_responses tables in the same database.

    :param fm: FileManager instance to handle file I/O operations
    """
    datastore_dir = fm.allocate_datastore()
    db_path = datastore_dir / "datastore.db"

    # Check if database exists
    if not db_path.exists():
        print("No datastore.db found - nothing to migrate")
        return

    print(f"Migrating datastore.db to new table structure...")

    # Connect to database
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        # Check if new table structure already exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        if "anon_responses" in tables and "chk_responses" in tables:
            print("New table structure already exists - skipping migration")
            return

        # Check if old responses table exists
        if "responses" not in tables:
            print("No responses table found - nothing to migrate")
            return

        # Create new table structure
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

        conn.commit()

        # Check if checkpoint column exists in old responses table
        cursor = conn.execute("PRAGMA table_info(responses)")
        columns = [row[1] for row in cursor.fetchall()]
        has_checkpoint = "checkpoint" in columns

        # Migrate responses using simple SQL queries
        if has_checkpoint:
            # ANON_RESPONSES = SELECT * FROM responses WHERE checkpoint IS NULL OR checkpoint = ''
            anon_cursor = conn.execute("""
                SELECT agent_name, seq_id, session_id, doc_hash, response, response_id, provider_type
                FROM responses 
                WHERE checkpoint IS NULL OR checkpoint = '' OR checkpoint = 'default'
            """)
            anon_rows = anon_cursor.fetchall()

            # CHK_RESPONSES = SELECT * FROM responses WHERE checkpoint IS NOT NULL AND checkpoint != ''
            chk_cursor = conn.execute("""
                SELECT agent_name, checkpoint, seq_id, session_id, doc_hash, response, response_id, provider_type
                FROM responses 
                WHERE checkpoint IS NOT NULL AND checkpoint != '' AND checkpoint != 'default'
            """)
            chk_rows = chk_cursor.fetchall()
        else:
            # No checkpoint column - all responses go to anonymous table
            anon_cursor = conn.execute("""
                SELECT agent_name, seq_id, session_id, doc_hash, response, response_id, provider_type
                FROM responses
            """)
            anon_rows = anon_cursor.fetchall()
            chk_rows = []

        # Insert into anonymous responses table
        anon_count = 0
        for row in anon_rows:
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO anon_responses 
                    (agent_name, seq_id, session_id, doc_hash, response, response_id, provider_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    tuple(row),
                )
                anon_count += 1
            except sqlite3.IntegrityError as e:
                print(f"  Warning: Skipping duplicate in anon_responses: {e}")

        # Insert into checkpoint responses table
        chk_count = 0
        for row in chk_rows:
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO chk_responses 
                    (agent_name, checkpoint, seq_id, session_id, doc_hash, response, response_id, provider_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    tuple(row),
                )
                chk_count += 1
            except sqlite3.IntegrityError as e:
                print(f"  Warning: Skipping duplicate in chk_responses: {e}")

        conn.commit()

        print(f"  ✓ Migrated {anon_count} responses to anon_responses table")
        print(f"  ✓ Migrated {chk_count} responses to chk_responses table")

        # Check if we should drop the old responses table
        if anon_count + chk_count > 0:
            print(
                f"  ✓ Migration successful - you can now safely drop the old 'responses' table"
            )
            # Optionally rename the old table instead of dropping it
            try:
                conn.execute("ALTER TABLE responses RENAME TO responses_old_backup")
                conn.commit()
                print(f"  ✓ Old 'responses' table renamed to 'responses_old_backup'")
            except sqlite3.Error as e:
                print(f"  Warning: Could not rename old table: {e}")

        print(f"\nMigration complete!")

    finally:
        conn.close()


def _check_and_migrate(ds: "SQLiteDatastore") -> None:
    """
    Check if old directory-based structure exists and migrate if needed.
    Only runs once per datastore directory.
    """
    datastore_dir = ds.file_manager.allocate_datastore()
    # migration_marker = datastore_dir / ".migrated"

    # # Skip if already migrated
    # if migration_marker.exists():
    #     return

    # Check if there are old-style subdirectories
    has_old_structure = False
    for item in datastore_dir.iterdir():
        if item.is_dir() and item.name not in [
            "apimeta",
            ".migrated",
            "anon-datastore.db",
            "chk-datastore.db",
        ]:
            # Check if it contains checkpoint subdirectories with databases
            for subitem in item.iterdir():
                if subitem.is_dir() and (subitem / "datastore.db").exists():
                    has_old_structure = True
                    break
            if has_old_structure:
                break

    # if has_old_structure:
    #     print("Detected old directory-based database structure. Starting migration...")
    #     migrate_old_directory_structure(ds.file_manager)
    #     # Create marker file to prevent re-migration
    #     migration_marker.touch()
    #     print("Migration marker created.")
    # else:
    # No old structure, just create the marker
    # migration_marker.touch()


def _migrate_sql_schema(conn: sqlite3.Connection, db_name: Optional[str]) -> None:
    """
    Migrate SQL schema for a database.

    :param conn: SQLite connection to migrate
    :param db_name: Name of the database (None for main database, or custom names for additional databases)
    """
    try:
        # For main database (db_name is None), check both response tables
        if db_name is None:
            # Check if anon_responses table exists and migrate
            try:
                cursor = conn.execute("PRAGMA table_info(anon_responses)")
                columns = [row[1] for row in cursor.fetchall()]

                # Add agent_name column if it doesn't exist
                if "agent_name" not in columns:
                    print(
                        "Migrating SQLite schema: adding agent_name column to anon_responses..."
                    )
                    conn.execute(
                        "ALTER TABLE anon_responses ADD COLUMN agent_name TEXT"
                    )
                    conn.commit()

                # Add provider_type column if it doesn't exist
                if "provider_type" not in columns:
                    print(
                        "Migrating SQLite schema: adding provider_type column to anon_responses..."
                    )
                    conn.execute(
                        "ALTER TABLE anon_responses ADD COLUMN provider_type TEXT"
                    )
                    conn.execute(
                        "UPDATE anon_responses SET provider_type = 'openai' WHERE provider_type IS NULL"
                    )
                    conn.commit()
            except sqlite3.OperationalError:
                # Table doesn't exist yet, skip
                pass

            # Check if chk_responses table exists and migrate
            try:
                cursor = conn.execute("PRAGMA table_info(chk_responses)")
                columns = [row[1] for row in cursor.fetchall()]

                # Add agent_name column if it doesn't exist
                if "agent_name" not in columns:
                    print(
                        "Migrating SQLite schema: adding agent_name column to chk_responses..."
                    )
                    conn.execute("ALTER TABLE chk_responses ADD COLUMN agent_name TEXT")
                    conn.commit()

                # Add checkpoint column if it doesn't exist
                if "checkpoint" not in columns:
                    print(
                        "Migrating SQLite schema: adding checkpoint column to chk_responses..."
                    )
                    conn.execute(
                        "ALTER TABLE chk_responses ADD COLUMN checkpoint TEXT NOT NULL DEFAULT ''"
                    )
                    conn.commit()

                # Add provider_type column if it doesn't exist
                if "provider_type" not in columns:
                    print(
                        "Migrating SQLite schema: adding provider_type column to chk_responses..."
                    )
                    conn.execute(
                        "ALTER TABLE chk_responses ADD COLUMN provider_type TEXT"
                    )
                    conn.execute(
                        "UPDATE chk_responses SET provider_type = 'openai' WHERE provider_type IS NULL"
                    )
                    conn.commit()
            except sqlite3.OperationalError:
                # Table doesn't exist yet, skip
                pass

        else:
            # For custom named databases, check legacy responses table
            try:
                cursor = conn.execute("PRAGMA table_info(responses)")
                columns = [row[1] for row in cursor.fetchall()]

                # Add agent_name column if it doesn't exist
                if "agent_name" not in columns:
                    print(
                        f"Migrating {db_name} SQLite schema: adding agent_name column..."
                    )
                    conn.execute("ALTER TABLE responses ADD COLUMN agent_name TEXT")
                    conn.commit()

                # Add provider_type column if it doesn't exist
                if "provider_type" not in columns:
                    print(
                        f"Migrating {db_name} SQLite schema: adding provider_type column..."
                    )
                    conn.execute("ALTER TABLE responses ADD COLUMN provider_type TEXT")
                    conn.execute(
                        "UPDATE responses SET provider_type = 'openai' WHERE provider_type IS NULL"
                    )
                    conn.commit()
            except sqlite3.OperationalError:
                # Table doesn't exist yet, skip
                pass

        # Check metadata table (exists in all databases)
        try:
            cursor = conn.execute("PRAGMA table_info(metadata)")
            metadata_columns = [row[1] for row in cursor.fetchall()]

            # Add provider_type column to metadata if it doesn't exist
            if "provider_type" not in metadata_columns:
                db_label = "main database" if db_name is None else f"{db_name} database"
                print(
                    f"Migrating {db_label} SQLite schema: adding provider_type to metadata..."
                )
                conn.execute("ALTER TABLE metadata ADD COLUMN provider_type TEXT")
                conn.execute(
                    "UPDATE metadata SET provider_type = 'openai' WHERE provider_type IS NULL"
                )
                conn.commit()
        except sqlite3.OperationalError:
            # Metadata table doesn't exist yet, skip
            pass

    except sqlite3.Error as e:
        # If migration fails, continue - tables will be created fresh
        db_label = "main database" if db_name is None else f"{db_name} database"
        print(f"Warning: Schema migration failed for {db_label}: {e}")
        pass


def _tmp_migrate_apimeta(fm: FileManager):
    path = fm.allocate_datastore() / "apimeta"

    num_moved = 0
    if (path / "responses.parquet").exists():
        # move to (path / "openai-responses.parquet")
        (path / "responses.parquet").rename(path / "openai-responses.parquet")
        num_moved += 1
    if (path / "messages.parquet").exists():
        (path / "messages.parquet").rename(path / "openai-messages.parquet")
        num_moved += 1
    if num_moved > 0:
        print(f"Migrated {num_moved} apimeta files")


if __name__ == "__main__":
    root = ".pllm"
    for subdir in os.listdir(root):
        fm = FileManager(f"{root}/{subdir}")
        # _tmp_migrate_datastores(fm)
        _tmp_migrate_apimeta(fm)
