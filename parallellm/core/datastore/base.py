from abc import ABC
from typing import Optional, Union
import sqlite3
from pathlib import Path

import polars as pl

from parallellm.file_io.file_manager import FileManager
from parallellm.types import CallIdentifier, ParsedResponse


class Datastore(ABC):
    """
    Stores responses
    """

    def retrieve(
        self, call_id: CallIdentifier, metadata=False
    ) -> Optional[ParsedResponse]:
        """
        Retrieve a response from the backend.

        :param call_id: The task identifier containing agent_name, doc_hash, and seq_id.
        :param metadata: Whether to include metadata in the response.
        :returns: The retrieved response content.
        """
        raise NotImplementedError

    def retrieve_metadata_legacy(self, response_id: str) -> Optional[dict]:
        """
        Retrieve metadata from the backend using response_id.

        :param response_id: The response ID to look up metadata for.
        :returns: The retrieved metadata as a dictionary, or None if not found.
        """
        raise NotImplementedError

    def store(
        self,
        call_id: CallIdentifier,
        parsed_response: ParsedResponse,
    ):
        """
        Store a response in the backend.

        :param call_id: The task identifier containing doc_hash and seq_id.
        :param parsed_response: The parsed response object containing text, response_id, and metadata.
        :returns: The seq_id where the response was stored (if applicable).
        """
        raise NotImplementedError

    def persist(self) -> None:
        """
        Persist changes to file(s). Cleans up resources.
        """
        raise NotImplementedError
