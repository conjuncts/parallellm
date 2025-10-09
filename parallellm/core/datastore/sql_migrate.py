import os
from typing import TYPE_CHECKING
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


def migrate_old_directory_structure(file_manager: FileManager) -> None:
    """
    Migrate old directory-based SQLite structure to new unified database.

    Old structure: datastore/{agent_name}/{checkpoint}/datastore.db
    New structure: datastore/datastore.db (with agent_name and checkpoint columns)

    This function:
    1. Scans the datastore directory for old agent/checkpoint subdirectories
    2. Reads data from each old database
    3. Inserts it into the new unified database with agent_name and checkpoint columns
    4. Optionally backs up old databases before deletion

    :param file_manager: FileManager instance to handle file I/O operations
    """
    datastore_dir = file_manager.allocate_datastore()
    new_db_path = datastore_dir / "datastore.db"

    # Create new database connection
    new_conn = sqlite3.connect(str(new_db_path))
    new_conn.row_factory = sqlite3.Row

    try:
        # Create tables if they don't exist (in case this is a fresh database)
        new_conn.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                checkpoint TEXT,
                seq_id INTEGER NOT NULL,
                session_id INTEGER NOT NULL,
                doc_hash TEXT NOT NULL,
                response TEXT NOT NULL,
                response_id TEXT,
                provider_type TEXT,
                UNIQUE(agent_name, checkpoint, doc_hash)
            )
        """)

        new_conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                response_id TEXT NOT NULL,
                metadata TEXT NOT NULL,
                provider_type TEXT,
                UNIQUE(response_id)
            )
        """)
        new_conn.commit()

        # Track migration statistics
        total_responses = 0
        total_metadata = 0
        migrated_dbs = []
        total_parquet_responses = 0
        total_parquet_messages = 0

        # Setup new apimeta directory at the top level
        new_apimeta_dir = datastore_dir / "apimeta"
        new_apimeta_dir.mkdir(exist_ok=True)
        new_responses_parquet = new_apimeta_dir / "responses.parquet"
        new_messages_parquet = new_apimeta_dir / "messages.parquet"

        # Scan for old database files in agent/checkpoint subdirectories
        # Pattern: datastore/{agent_name}/{checkpoint_hash}/datastore.db
        targets = []
        for agent_dir in datastore_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            # Skip the new database file and apimeta directory
            if agent_dir.name in ["datastore.db", "apimeta", "datastore.db-journal"]:
                continue

            agent_name = agent_dir.name

            # Look for checkpoint subdirectories
            for checkpoint_dir in agent_dir.iterdir():
                if not checkpoint_dir.is_dir():
                    continue

                old_db_path = checkpoint_dir / "datastore.db"

                if not old_db_path.exists():
                    continue

                # Extract checkpoint name from directory (remove hash suffix if present)
                checkpoint_name = checkpoint_dir.name.split("-")[
                    0
                ]  # Take part before hash
                targets.append((agent_name, checkpoint_name, old_db_path))

        for agent_name, checkpoint_name, old_db_path in targets:
            print(
                f"Migrating: agent='{agent_name}', "
                f"checkpoint='{checkpoint_name}' "
                f"from {old_db_path}"
            )

            # Connect to old database
            old_conn = sqlite3.connect(str(old_db_path))
            old_conn.row_factory = sqlite3.Row

            try:
                # Migrate responses table
                cursor = old_conn.execute("SELECT * FROM responses")
                responses = cursor.fetchall()

                for row in responses:
                    # Insert into new database with agent_name and checkpoint
                    try:
                        new_conn.execute(
                            """
                            INSERT OR REPLACE INTO responses 
                            (agent_name, checkpoint, seq_id, session_id, doc_hash, response, response_id, provider_type)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                agent_name,
                                checkpoint_name,
                                row["seq_id"],
                                row["session_id"],
                                row["doc_hash"],
                                row["response"],
                                row["response_id"],
                                row["provider_type"],
                            ),
                        )
                        total_responses += 1
                    except sqlite3.IntegrityError as e:
                        print(f"  Warning: Skipping duplicate response: {e}")

                # Migrate metadata table if it exists
                try:
                    cursor = old_conn.execute("SELECT * FROM metadata")
                    metadata_rows = cursor.fetchall()

                    for row in metadata_rows:
                        # Insert into new database
                        try:
                            new_conn.execute(
                                """
                                INSERT OR REPLACE INTO metadata 
                                (response_id, metadata, provider_type)
                                VALUES (?, ?, ?)
                            """,
                                (
                                    row["response_id"],
                                    row["metadata"],
                                    row["provider_type"],
                                ),
                            )
                            total_metadata += 1
                        except sqlite3.IntegrityError as e:
                            print(f"  Warning: Skipping duplicate metadata: {e}")
                except sqlite3.OperationalError:
                    # Metadata table doesn't exist in old database
                    pass

                new_conn.commit()
                migrated_dbs.append(old_db_path)
                print(f"  ✓ Migrated {len(responses)} responses")

                # Migrate parquet files if they exist
                old_apimeta_dir = checkpoint_dir / "apimeta"
                if not old_apimeta_dir.exists() or HAS_POLARS:
                    old_responses_parquet = old_apimeta_dir / "responses.parquet"
                    old_messages_parquet = old_apimeta_dir / "messages.parquet"

                    # Migrate responses.parquet
                    if old_responses_parquet.exists():
                        try:
                            old_df = pl.read_parquet(old_responses_parquet)

                            # Merge with existing new parquet file if it exists
                            if new_responses_parquet.exists():
                                existing_df = pl.read_parquet(new_responses_parquet)
                                merged_df = pl.concat([existing_df, old_df]).unique()
                            else:
                                merged_df = old_df

                            merged_df.write_parquet(new_responses_parquet)
                            total_parquet_responses += len(old_df)
                            print(f"  ✓ Migrated {len(old_df)} parquet responses")
                        except Exception as e:
                            print(
                                f"  Warning: Failed to migrate responses.parquet: {e}"
                            )

                    # Migrate messages.parquet
                    if old_messages_parquet.exists():
                        try:
                            old_df = pl.read_parquet(old_messages_parquet)

                            # Merge with existing new parquet file if it exists
                            if new_messages_parquet.exists():
                                existing_df = pl.read_parquet(new_messages_parquet)
                                merged_df = pl.concat([existing_df, old_df]).unique()
                            else:
                                merged_df = old_df

                            merged_df.write_parquet(new_messages_parquet)
                            total_parquet_messages += len(old_df)
                            print(f"  ✓ Migrated {len(old_df)} parquet messages")
                        except Exception as e:
                            print(f"  Warning: Failed to migrate messages.parquet: {e}")

            finally:
                old_conn.close()

        print(f"\nMigration complete!")
        print(f"  Total responses migrated: {total_responses}")
        print(f"  Total metadata migrated: {total_metadata}")
        print(f"  Total parquet responses migrated: {total_parquet_responses}")
        print(f"  Total parquet messages migrated: {total_parquet_messages}")
        print(f"  Databases migrated: {len(migrated_dbs)}")

        # Ask user if they want to remove old databases
        if migrated_dbs:
            print("\nOld database files:")
            for db_path in migrated_dbs:
                print(f"  {db_path}")
            print(
                "\nYou can safely delete the old agent/checkpoint subdirectories if migration was successful."
            )

    finally:
        new_conn.close()


def _check_and_migrate(ds: "SQLiteDatastore") -> None:
    """
    Check if old directory-based structure exists and migrate if needed.
    Only runs once per datastore directory.
    """
    datastore_dir = ds.file_manager.allocate_datastore()
    migration_marker = datastore_dir / ".migrated"

    # Skip if already migrated
    if migration_marker.exists():
        return

    # Check if there are old-style subdirectories
    has_old_structure = False
    for item in datastore_dir.iterdir():
        if item.is_dir() and item.name not in ["apimeta", ".migrated"]:
            # Check if it contains checkpoint subdirectories with databases
            for subitem in item.iterdir():
                if subitem.is_dir() and (subitem / "datastore.db").exists():
                    has_old_structure = True
                    break
            if has_old_structure:
                break

    if has_old_structure:
        print("Detected old directory-based database structure. Starting migration...")
        migrate_old_directory_structure(ds.file_manager)
        # Create marker file to prevent re-migration
        migration_marker.touch()
        print("Migration marker created.")
    else:
        # No old structure, just create the marker
        migration_marker.touch()


def _migrate_sql_schema(conn: sqlite3.Connection) -> None:
    try:
        # Check if old schema exists (responses table without agent_name/checkpoint columns)
        cursor = conn.execute("PRAGMA table_info(responses)")
        columns = [row[1] for row in cursor.fetchall()]

        # Add agent_name column if it doesn't exist
        if "agent_name" not in columns:
            print("Migrating SQLite schema: adding agent_name column...")
            conn.execute("ALTER TABLE responses ADD COLUMN agent_name TEXT NOT NULL")
            conn.commit()

        # Add checkpoint column if it doesn't exist
        if "checkpoint" not in columns:
            print("Migrating SQLite schema: adding checkpoint column...")
            conn.execute("ALTER TABLE responses ADD COLUMN checkpoint TEXT")
            conn.commit()

        # Add provider_type column if it doesn't exist
        if "provider_type" not in columns:
            print("Migrating SQLite schema: adding provider_type column...")
            conn.execute("ALTER TABLE responses ADD COLUMN provider_type TEXT")
            conn.execute(
                "UPDATE responses SET provider_type = 'openai' WHERE provider_type IS NULL"
            )
            conn.commit()

        # Check metadata table
        cursor = conn.execute("PRAGMA table_info(metadata)")
        metadata_columns = [row[1] for row in cursor.fetchall()]

        # Add provider_type column to metadata if it doesn't exist
        if "provider_type" not in metadata_columns:
            print("Migrating SQLite schema: adding provider_type to metadata...")
            conn.execute("ALTER TABLE metadata ADD COLUMN provider_type TEXT")
            conn.execute(
                "UPDATE metadata SET provider_type = 'openai' WHERE provider_type IS NULL"
            )
            conn.commit()

    except sqlite3.Error as e:
        # If migration fails, continue - tables will be created fresh
        print(f"Warning: Schema migration failed: {e}")
        pass


if __name__ == "__main__":
    for subdir in os.listdir(".pllm"):
        fm = FileManager(f".pllm/{subdir}")
        migrate_old_directory_structure(fm)
