import asyncio
from typing import Optional
from parallellm.core.datastore.base import DataStore


class BaseBackend:
    """
    A backend is a data store, but also a way to poll
    """

    # def __init__(self, datastore: DataStore):
    #     super().__init__()
    #     self.datastore = datastore

    async def _poll_changes(self, stage: str, doc_hash: str, seq_id: int = None):
        """
        A chance to poll for changes and update the data store
        """
        raise NotImplementedError

    def persist(self):
        self.datastore.persist()

    def retrieve(self, stage: str, doc_hash: str, seq_id: int = None) -> Optional[str]:
        raise NotImplementedError

    def store(
        self, stage: str, doc_hash: str, response: str, seq_id: int = None
    ) -> int:
        raise NotImplementedError
