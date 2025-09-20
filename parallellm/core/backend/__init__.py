import asyncio
from typing import Optional
from parallellm.core.datastore.base import Datastore
from parallellm.types import CallIdentifier


class BaseBackend:
    """
    A backend is a data store, but also a way to poll
    """

    async def _poll_changes(self, call_id: CallIdentifier):
        """
        A chance to poll for changes and update the data store
        """
        raise NotImplementedError

    def persist(self):
        pass

    def retrieve(self, call_id: CallIdentifier) -> Optional[str]:
        raise NotImplementedError

    def store(self, call_id: CallIdentifier, response: str) -> int:
        raise NotImplementedError
