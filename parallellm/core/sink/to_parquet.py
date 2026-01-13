from pathlib import Path
from typing import Literal, Union

import polars as pl


def write_to_parquet(
    parquet_fpath: Path,
    data: Union[dict, list, pl.DataFrame],
    *,
    mode: Literal["append", "replace", "unique", "update"] = "append",
    on: list[str] = [],
):
    """
    Write data to a Parquet file.

    :param mode: How to write the data.
        Append: Adds new rows to the existing data.
        Replace: Replaces the existing data with the new data.
        Unique: Adds only new rows that don't exist in the existing data.
        Update: Updates existing rows with new data.
    :param on: Columns which collectively should serve as primary key.
        If mode is "unique", these are the columns compared.
        If mode is "update", these are the columns used to identify rows to update.
    """
    if isinstance(data, pl.DataFrame):
        commit = data
    else:
        commit = pl.DataFrame(data)

    parquet_fpath.parent.mkdir(parents=True, exist_ok=True)
    if parquet_fpath.exists():
        existing = pl.read_parquet(parquet_fpath)
        if mode == "append":
            commit = pl.concat([existing, commit], how="diagonal_relaxed")
        elif mode == "unique":
            commit = commit.join(existing, on=on, how="anti")
            commit = pl.concat([existing, commit], how="diagonal_relaxed")
        elif mode == "update":
            commit = existing.update(commit, on=on)
        elif mode == "replace":
            pass  # Nothing to do
        else:
            raise ValueError(f"Unknown mode: {mode}")

    tmp_fpath = parquet_fpath.with_suffix(".tmp")
    commit.write_parquet(tmp_fpath)
    tmp_fpath.replace(parquet_fpath)
    return commit
