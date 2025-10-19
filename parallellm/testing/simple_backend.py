from typing import Optional
from parallellm.core.backend import BaseBackend
from parallellm.core.calls import _call_to_concise_dict
from parallellm.types import CallIdentifier, ParsedResponse


class MockBackend(BaseBackend):
    """
    A simple in-memory backend for testing purposes.
    """

    def __init__(self):
        self._dict = {}

    async def _poll_changes(self, call_id: CallIdentifier):
        pass

    def persist(self):
        pass

    def retrieve(
        self, call_id: CallIdentifier, metadata=False
    ) -> Optional[ParsedResponse]:
        """
        Retrieve a response.

        :param call_id: The task identifier containing agent_name, checkpoint, doc_hash, and seq_id.
        :returns: The retrieved ParsedResponse.
        """
        return self._dict.get(tuple(_call_to_concise_dict(call_id).values()))

    def store(self, call_id: CallIdentifier, response: ParsedResponse):
        self._dict[tuple(_call_to_concise_dict(call_id).values())] = response
