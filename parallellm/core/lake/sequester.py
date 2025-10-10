from pathlib import Path
from typing import Dict, Optional
import polars as pl

from parallellm.core.lake.metadata_sinks import openai_metadata_sinker


def sequester_df_to_parquet(
    df: pl.DataFrame,
    table_name: str,
    parquet_path: Path,
    *,
    return_key=False,
) -> Optional[list]:
    """
    Backup a df to parquet; merging with existing data if present.

    :param df: DataFrame to backup
    :param table_name: Name of the SQL table to backup
    :param parquet_path: Path where to save the parquet file
    :returns: List of IDs/keys of transferred records, or None if no data
    """
    new_df = df

    if new_df is None:
        return None

    # Ensure parent directory exists
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing data if parquet file exists
    final_df = new_df
    if parquet_path.exists():
        try:
            existing_df = pl.read_parquet(parquet_path)
            # Concatenate and deduplicate based on all columns
            final_df = pl.concat([existing_df, new_df], how="diagonal_relaxed")
        except Exception as e:
            print(f"Warning: Failed to read existing {parquet_path.name}: {e}")

    # Write to temporary file for atomic operation
    temp_path = parquet_path.with_suffix(".parquet.tmp")

    try:
        final_df.write_parquet(temp_path)

        # Atomic swap: move temporary file to final location
        if parquet_path.exists():
            parquet_path.unlink()  # Remove old file
        temp_path.rename(parquet_path)

        # Return list of transferred record identifiers for cleanup
        if return_key and return_key in new_df.columns:
            return new_df.select(return_key).to_series().to_list()
        elif "id" in new_df.columns:
            return new_df.select("id").to_series().to_list()
        elif "response_id" in new_df.columns:
            return new_df.select("response_id").to_series().to_list()
        else:
            # For tables without clear ID columns, return row count
            return list(range(len(new_df)))

    except Exception as e:
        # Clean up temporary file on error
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise RuntimeError(f"Error backing up {table_name} to parquet: {e}")


def sequester_openai_metadata(
    metadata_rows: list[Dict], file_manager
) -> Optional[list[str]]:
    """
    Sequester OpenAI metadata from SQLite rows to Parquet files.
    Returns a list of response_ids that were successfully transferred and can be deleted from SQLite.
    """
    # metadata_rows is actually a sqlite3.Row object

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
    datastore_dir = file_manager.allocate_datastore()
    metadata_dir = datastore_dir / "apimeta"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    responses_parquet = metadata_dir / "openai-responses.parquet"
    messages_parquet = metadata_dir / "openai-messages.parquet"
    responses_tmp = metadata_dir / "openai-responses.parquet.tmp"
    messages_tmp = metadata_dir / "openai-messages.parquet.tmp"

    try:
        # Handle responses dataframe
        if not responses_df.is_empty():
            final_responses_df = responses_df

            # Merge with existing responses data if file exists
            if responses_parquet.exists():
                existing_responses = pl.read_parquet(responses_parquet)
                # Concatenate and deduplicate based on all columns
                final_responses_df = pl.concat(
                    [existing_responses, responses_df], how="diagonal_relaxed"
                )

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
                    [existing_messages, messages_df], how="diagonal_relaxed"
                )

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
        return response_ids_to_delete

    except Exception as e:
        # Clean up temporary files on error
        for tmp_file in [responses_tmp, messages_tmp]:
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except Exception:
                    pass  # Ignore cleanup errors
        raise RuntimeError(f"Error transferring metadata to Parquet: {e}")
