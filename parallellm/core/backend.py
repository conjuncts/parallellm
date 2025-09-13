from typing import Optional
from parallellm.core.datastore.base import DataStore


class BaseBackend(DataStore):
    """
    A backend is a data store, but also a way to poll
    """

    def __init__(self, datastore: DataStore):
        super().__init__()
        self.datastore = datastore

    def _poll_changes(self):
        """
        A chance to poll for changes and update the data store
        """
        pass

    def retrieve(self, stage: str, doc_hash: str, seq_id: int = None) -> Optional[str]:
        self._poll_changes()
        return self.datastore.retrieve(stage, doc_hash, seq_id)

    def store(self, stage: str, doc_hash: str, response: str) -> int:
        self._poll_changes()
        return self.datastore.store(stage, doc_hash, response)

    def persist(self):
        self.datastore.persist()
