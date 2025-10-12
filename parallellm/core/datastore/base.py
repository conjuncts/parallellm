from abc import ABC
from typing import Optional, Union
import sqlite3
from pathlib import Path

import polars as pl

from parallellm.file_io.file_manager import FileManager
from parallellm.types import CallIdentifier


class Datastore(ABC):
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

    def retrieve_metadata(self, call_id: CallIdentifier) -> Optional[dict]:
        """
        Retrieve metadata from the backend.

        :param call_id: The task identifier containing checkpoint, doc_hash, and seq_id.
        :returns: The retrieved metadata as a dictionary, or None if not found.
        """
        raise NotImplementedError

    def store(
        self,
        call_id: CallIdentifier,
        response: str,
        response_id: str,
        *,
        metadata: Optional[dict] = None,
    ) -> Optional[int]:
        """
        Store a response in the backend.

        :param call_id: The task identifier containing checkpoint, doc_hash, and seq_id.
        :param response: The response content to store.
        :param response_id: The response ID to store.
        :param metadata: Optional metadata dictionary to store alongside the response.
        :returns: The seq_id where the response was stored (if applicable).
        """
        raise NotImplementedError

    def persist(self) -> None:
        """
        Persist changes to file(s). Cleans up resources.
        """
        raise NotImplementedError
