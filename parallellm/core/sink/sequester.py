from pathlib import Path
from typing import Dict, Optional
import polars as pl

from parallellm.core.sink.to_parquet import write_to_parquet
from parallellm.provider.google._sink import google_metadata_sinker
from parallellm.provider.openai._sink import openai_metadata_sinker


def sequester_df_to_parquet(
    df: pl.DataFrame,
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

    write_to_parquet(parquet_path, new_df, mode="append")

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


def sequester_metadata(metadata_rows: list[Dict], file_manager) -> Optional[list[str]]:
    """
    Sequester OpenAI metadata from SQLite rows to Parquet files.
    Returns a list of response_ids that were successfully transferred and can be deleted from SQLite.
    """
    # metadata_rows is actually a sqlite3.Row object

    # Extract metadata strings for processing
    metadata_strings = {
        "openai": [],
        "google": [],
    }

    for row in metadata_rows:
        response_id = row["response_id"]
        metadata_json = row["metadata"]
        provider_type = row["provider_type"]

        agent_name = row["agent_name"]
        seq_id = row["seq_id"]
        session_id = row["session_id"]

        if metadata_json and provider_type in metadata_strings:
            metadata_strings[provider_type].append(
                (
                    {
                        "response_id": response_id,
                        "agent_name": agent_name,
                        "seq_id": seq_id,
                        "session_id": session_id,
                    },
                    metadata_json,
                )
            )

    if not metadata_strings:
        return

    # Process metadata using the existing sinker function
    response_ids_to_delete = []
    for provider_type, metas in metadata_strings.items():
        if provider_type == "openai":
            processed_dfs = openai_metadata_sinker(metas)

        elif provider_type == "google":
            processed_dfs = google_metadata_sinker(metas)
        else:
            # Unavailable
            continue

        sequester_dataframes(processed_dfs, file_manager, provider_type=provider_type)

        # Collect successfully sequestered response_ids
        response_df = processed_dfs["responses"]
        if not response_df.is_empty():
            response_ids_to_delete.extend(
                response_df.select("response_id").to_series().to_list()
            )

    return response_ids_to_delete


def sequester_dataframes(
    dfs: dict[str, pl.DataFrame], file_manager, provider_type: str
) -> Optional[list[str]]:
    # Create metadata directory
    datastore_dir = file_manager.allocate_datastore()
    metadata_dir = datastore_dir / "apimeta"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    tmp_files = []
    try:
        for df_name, df in dfs.items():
            if df.is_empty():
                continue
            df_parquet = metadata_dir / f"{provider_type}-{df_name}.parquet"
            df_tmp = metadata_dir / f"{provider_type}-{df_name}.parquet.tmp"
            tmp_files.append(df_tmp)

            # Handle responses dataframe
            final_df = df

            # Merge with existing responses data if file exists
            if df_parquet.exists():
                existing_responses = pl.read_parquet(df_parquet)
                final_df = pl.concat([existing_responses, df], how="diagonal_relaxed")

            # Write to temporary file
            final_df.write_parquet(df_tmp)

            # Atomic swap: move temporary files to final locations
            if df_tmp.exists():
                if df_parquet.exists():
                    df_parquet.unlink()
                df_tmp.rename(df_parquet)

    except Exception as e:
        # Clean up temporary files on error
        for tmp_file in tmp_files:
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except Exception:
                    pass  # Ignore cleanup errors
        raise RuntimeError(f"Error transferring metadata to Parquet: {e}")
