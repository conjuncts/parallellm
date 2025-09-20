from abc import ABC
from typing import Optional, Union
import sqlite3
from pathlib import Path

import polars as pl

from parallellm.file_io.file_manager import FileManager
from parallellm.types import CallIdentifier


class DataStore(ABC):
    """
    Stores responses
    """

    def retrieve(self, call_id: CallIdentifier) -> Optional[str]:
        """
        Retrieve a response from the backend.

        :param call_id: The task identifier containing checkpoint, doc_hash, and seq_id.
        :returns: The retrieved LLMResponse.
        """
        raise NotImplementedError

    def store(
        self,
        call_id: CallIdentifier,
        response: str,
        response_id: str,
        *,
        save_to_file: bool = True,
    ) -> Optional[int]:
        """
        Store a response in the backend.

        :param call_id: The task identifier containing checkpoint, doc_hash, and seq_id.
        :param response: The response content to store.
        :param response_id: The response ID to store.
        :param save_to_file: Whether to save the updated data back to the file.
        :returns: The seq_id where the response was stored (if applicable).
        """
        raise NotImplementedError

    def store_metadata(
        self,
        call_id: CallIdentifier,
        response_id: str,
        metadata: dict,
    ) -> None:
        """
        Store metadata in the backend.

        :param checkpoint: The checkpoint of the metadata.
        :param doc_hash: The document hash of the response.
        :param seq_id: The sequential ID of the response.
        :param response_id: The response ID to store.
        :param metadata: The metadata to store.
        """
        raise NotImplementedError

    def persist(self, checkpoint: Optional[str] = None) -> None:
        """
        Persist changes to file(s).

        :param checkpoint: The checkpoint to persist (if None, persist all checkpoints with changes).
        """
        raise NotImplementedError
